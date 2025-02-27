import os
from typing import Optional

import torch
from torch import nn
from transformers import Trainer
from transformers.utils import logging
from transformers.modeling_utils import PreTrainedModel
import loratorch as lora


logger = logging.get_logger(__name__)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"

class FSTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        fs_args = kwargs.pop('fs_args')
        super().__init__(*args, **kwargs)

        self.use_lora = fs_args.use_lora

        if self.use_lora:
            self.replace_lora_modules(rank=fs_args.lora_rank)
        else:
            self.freeze_model(
                freeze_modules=fs_args.freeze_modules,
                unfreeze_modules=fs_args.unfreeze_modules,
                freeze_at=fs_args.freeze_at
            )

    def replace_lora_modules(self, rank=8):
        """
        Replace applicable layers in the model with LoRA versions.
        """
        self.replace_lora(self.model.backbone, rank=128)

        # Log all LoRA layers applied and their types
        print("\n🔹 Checking LoRA Layers in the Model:")
        for name, module in self.model.named_modules():
            if isinstance(module, lora.LoRALayer):
                print(f"✅ LoRA Applied to: {name} -> {type(module).__name__}")

        # Log trainable parameters
        print("\n🔹 Checking Trainable Parameters:")
        for name, param in self.model.named_parameters():
            print(f"{name}: Trainable={param.requires_grad}")

    def replace_lora(self, model, module_name="", rank=8):
        """
        Recursively replace Conv2d, MultiheadAttention, and Linear layers in the model with LoRA equivalents.
        """
        for sub_module_name in model._modules:
            current_module_name = sub_module_name if module_name == "" else module_name + "." + sub_module_name

            if len(model._modules[sub_module_name]._modules) > 1:
                self.replace_lora(model._modules[sub_module_name], current_module_name, rank=rank)
            else:
                if isinstance(model._modules[sub_module_name], nn.Conv2d):
                    model._modules[sub_module_name] = lora.Conv2d(
                        in_channels=model._modules[sub_module_name].in_channels,
                        out_channels=model._modules[sub_module_name].out_channels,
                        kernel_size=model._modules[sub_module_name].kernel_size[0],
                        stride=model._modules[sub_module_name].stride,
                        padding=model._modules[sub_module_name].padding,
                        padding_mode=model._modules[sub_module_name].padding_mode,
                        dilation=model._modules[sub_module_name].dilation,
                        groups=model._modules[sub_module_name].groups,
                        bias=model._modules[sub_module_name].bias is not None,
                        r=rank
                    ).to('cuda')
                elif isinstance(model._modules[sub_module_name], nn.MultiheadAttention):
                    model._modules[sub_module_name] = lora.MultiheadAttention(
                        model._modules[sub_module_name].embed_dim,
                        model._modules[sub_module_name].num_heads,
                        dropout=model._modules[sub_module_name].dropout,
                        r=rank
                    ).to('cuda')
                elif isinstance(model._modules[sub_module_name], nn.Linear):
                    model._modules[sub_module_name] = lora.Linear(
                        model._modules[sub_module_name].in_features,
                        model._modules[sub_module_name].out_features,
                        bias=model._modules[sub_module_name].bias is not None,
                        r=rank
                    ).to('cuda')
                else:
                    if len(model._modules[sub_module_name]._modules) > 0:
                        self.replace_lora(model._modules[sub_module_name], current_module_name, rank=rank)

    def freeze_model(self, freeze_modules, unfreeze_modules, freeze_at):
        """
        Freeze model parameters for various modules
        When backbone_freeze == 0, freeze all backbone parameters
        Otherwise freeze up to res_#backbone_freeze_at layer.
        """
        if len(freeze_at) == 0:
            freeze_at = [0] * max(len(freeze_modules), len(unfreeze_modules))
        else:
            try:
                freeze_at = [int(x) if x != 'half' else 'half' for x in freeze_at]
            except ValueError:
                raise ValueError(
                    f"Invalid value for 'freeze_at': expected an integer or the string 'half' received {set(map(type, freeze_at))}.")

        module_exists = False

        def freeze(model, freeze_module, freeze_level=0, unfreeze=False):
            nonlocal module_exists
            if hasattr(model, freeze_module):
                module_exists = True
                if freeze_level == 'half':
                    freeze_level = int(len(list(getattr(model, freeze_module).parameters())) / 2)
                if freeze_level == 0:
                    freeze_level = int(len(list(getattr(model, freeze_module).parameters())))
                for idx, param in enumerate(getattr(model, freeze_module).parameters()):
                    if freeze_level >= idx:
                        param.requires_grad_(unfreeze)
            elif len(list(model.children())) != 0:
                for sub_modules in model.children():
                    freeze(sub_modules, freeze_module, freeze_level)

        def freeze_bias(model, unfreeze=False):
            if unfreeze:
                model.requires_grad_(False)
            for module in model.modules():
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.requires_grad_(unfreeze)

        def freeze_norm(model, unfreeze=False):
            if unfreeze:
                model.requires_grad_(False)
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    if hasattr(module, 'weight') and module.bias is not None:
                        module.weight.requires_grad_(unfreeze)
                    if hasattr(module, 'bias') and module.bias is not None:
                        module.bias.requires_grad_(unfreeze)

        def freeze_model_process(model):
            nonlocal module_exists
            if len(freeze_modules) != 0 and len(unfreeze_modules) != 0:
                raise ValueError("Parameters 'freeze_modules' and 'unfreeze_modules' cannot be given at the same time")

            if len(freeze_modules) != 0:
                if len(freeze_modules) != len(freeze_at):
                    raise ValueError(
                        f"Length of 'freeze_modules' {len(freeze_modules)} and 'freeze_at' {len(freeze_at)} should be the same")

                for freeze_module, at in zip(freeze_modules, freeze_at):
                    if freeze_module == 'bias':
                        freeze_bias(model)
                    elif freeze_module == 'norm':
                        freeze_norm(model)
                    else:
                        module_exists = False
                        freeze(model, freeze_module, at)
                        if not module_exists:
                            raise ValueError(f"The specified module '{freeze_module}' was not found in the model. "
                                             "Please ensure the module name is correct and exists in the model's architecture.")

            if len(unfreeze_modules) != 0:
                if len(unfreeze_modules) != len(freeze_at):
                    raise ValueError(
                        f"Length of 'unfreeze_modules' {len(unfreeze_modules)} and 'freeze_at' {len(freeze_at)} should be the same")

                for unfreeze_module, at in zip(unfreeze_modules, freeze_at):
                    if unfreeze_module == 'bias':
                        freeze_bias(model, unfreeze=True)
                    elif unfreeze_module == 'norm':
                        freeze_norm(model, unfreeze=True)
                    else:
                        module_exists = False
                        model.requires_grad_(False)
                        freeze(model, unfreeze_module, at, unfreeze=True)
                        if not module_exists:
                            raise ValueError(f"The specified module '{unfreeze_module}' was not found in the model. "
                                             "Please ensure the module name is correct and exists in the model's architecture.")

        freeze_model_process(self.model.model if hasattr(self.model, 'model') else self.model)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Will save the lora wieghts if `use_lora=True`.

        Will only save from the main process.
        """
        if self.use_lora:
            # If we are executing this function, we are the process zero, so we don't check for that.
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Saving lora wieghts checkpoint to {output_dir}")

            supported_classes = (PreTrainedModel,)
            state_dict = lora.lora_state_dict(self.model)
            # Save a trained model and configuration using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            if not isinstance(self.model, supported_classes):
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
            else:
                self.model.save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )

            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        else:
            super().save_model(output_dir, _internal_call)
