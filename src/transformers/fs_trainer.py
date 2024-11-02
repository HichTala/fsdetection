from torch import nn
from transformers import Trainer


class FSTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        fs_args = kwargs.pop('fs_args')
        super().__init__(*args, **kwargs)

        self.freeze_model(
            freeze_modules=fs_args.freeze_modules,
            unfreeze_modules=fs_args.unfreeze_modules,
            freeze_at=fs_args.freeze_at
        )

    def freeze_model(self, freeze_modules, unfreeze_modules, freeze_at):
        """
        Freeze model parameters for various modules
        When backbone_freeze == 0, freeze all backbone parameters
        Otherwise freeze up to res_#backbone_freeze_at layer.
        """
        if len(freeze_at) == 0:
            freeze_at = [0] * max(len(freeze_modules), len(unfreeze_modules))

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

        freeze_model_process(self.model.model)

    def train(self, *args, **kwargs):
        # print model module names and if they are frozen or no
        for name, param in self.model.named_parameters():
            print(name, param.requires_grad)
        pass
