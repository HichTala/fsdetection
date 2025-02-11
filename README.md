# fsdetection

**fsdetection** is a Python package for few-shot object detection, inspired by the simplicity and flexibility of Hugging Face libraries. With `fsdetection`, you can quickly experiment with few-shot learning for object detection tasks, easily integrate it with popular frameworks, and customize detection models with minimal data.

## Features
- **Few-Shot Object Detection**: Fine-tune object detection models with only a few examples per class.
- **Cross-Domain Adaptation**: Effortlessly adapt models to new domains.
- **Modular Design**: Build and customize models with a clean, intuitive API.
- **Pre-trained Models**: Access a range of pre-trained models as a starting point for your tasks.

## Installation

Install `fsdetection` directly from PyPI:

```bash
pip install fsdetection
```

## LoRA script implementation

https://github.com/Baijiong-Lin/LoRA-Torch

```python
def replace_lora(model, module_name, rank):
    for sub_module_name in model._modules:
        cuurent_module_name = sub_module_name if module_name == "" else module_name + "." + sub_module_name

        if len(model._modules[sub_module_name]._modules) > 1:
            replace_lora(model._modules[sub_module_name], cuurent_module_name, rank=rank)
        else:
            if isinstance(model._modules[sub_module_name], nn.Conv2d):
                model._modules[sub_module_name] = LoraConv2d(
                    in_channels=model._modules[sub_module_name].in_channels,
                    out_channels=model._modules[sub_module_name].out_channels,
                    kernel_size=model._modules[sub_module_name].kernel_size[0],
                    stride=model._modules[sub_module_name].stride,
                    padding=model._modules[sub_module_name].padding,
                    padding_mode=model._modules[sub_module_name].padding_mode,
                    dilation=model._modules[sub_module_name].dilation,
                    groups=model._modules[sub_module_name].groups,
                    bias=model._modules[sub_module_name].bias is not None,
                    norm=model._modules[sub_module_name].norm,
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
                    replace_lora(model._modules[sub_module_name], cuurent_module_name, rank=rank)


class LoraTrainer(FineTuningTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    @classmethod
    def build_model(cls, cfg, is_finetuned=False):
        model = super().build_model(cfg, is_finetuned)
        replace_lora(model, "", rank=cfg.FINETUNE.LORA.RANK)
        return model
```
