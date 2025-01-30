
import torch.nn as nn
import loratorch as lora
import torch
import math

def set_param(curr_mod, name, param=None, mode='update', with_nn=False):
    r"""Refer to https://github.com/Baijiong-Lin/MOML/blob/main/MTL/utils.py"""
    if '.' in name:
        n = name.split('.')
        module_name = n[0]
        rest = '.'.join(n[1:])
        for name, mod in curr_mod.named_children():
            if module_name == name:
                return set_param(mod, rest, param, mode=mode, with_nn=with_nn)
    else:
        if mode == 'update':
            delattr(curr_mod, name)
            if with_nn:
                setattr(curr_mod, name, nn.Parameter(param))
            else:
                setattr(curr_mod, name, param)
        elif mode == 'get':
            if hasattr(curr_mod, name):
                p = getattr(curr_mod, name)
                return p

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        fan_in_fan_out: bool = False,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        if self.r > 0:
            self.scaling = self.lora_alpha / self.r
        # Mark the weight as unmerged
        self.merged = False
        # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        self.fan_in_fan_out = fan_in_fan_out
        # define params that require LoRA {'param_name': 'lora_name'}
        self.params_with_lora = {}

    def register_weight_after_backward(self):
        for param_name, _ in self.params_with_lora.items():
            p = set_param(self, param_name, mode='get')
            # print('+'*10, param_name, p.flatten()[:10])
            set_param(self, param_name, param=p, mode='update', with_nn=True)

    def register_lora_param(self):
        r"""Register LoRA matrix"""
        for param_name, lora_name in self.params_with_lora.items():
            assert len(eval(f'self.{param_name}').size()) == 2
            self.register_parameter(f'{lora_name}_lora_A', 
                nn.Parameter(eval(f'self.{param_name}').new_zeros((self.r, eval(f'self.{param_name}').size()[1])))
                )
            self.register_parameter(f'{lora_name}_lora_B', 
                nn.Parameter(eval(f'self.{param_name}').new_zeros((eval(f'self.{param_name}').size()[0], self.r)))
                )
            eval(f'self.{param_name}').requires_grad = False

    def init_lora_param(self):
        for param_name, lora_name in self.params_with_lora.items():
            if hasattr(self, f'{lora_name}_lora_A'):
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(eval(f'self.{lora_name}_lora_A'), a=math.sqrt(5))
                nn.init.zeros_(eval(f'self.{lora_name}_lora_B'))

    def transpose(self, w: torch.Tensor):
        return w.transpose(0, 1) if self.fan_in_fan_out else w

    def merge_BA(self, param_name: str):
        lora_name = self.params_with_lora[param_name]
        return self.transpose((eval(f'self.{lora_name}_lora_B') @ eval(f'self.{lora_name}_lora_A')).view(eval(f'self.{param_name}').shape))

    def merge_lora_param(self):
        r"""p_new = p + scaling * B @ A and keep differentiable to A and B"""
        for param_name, lora_name in self.params_with_lora.items():
            p = set_param(self, param_name, mode='get')
            # detach() is very important here
            p_new = p.detach() + self.merge_BA(param_name) * self.scaling
            set_param(self, param_name, param=p_new, mode='update')

    def add_lora_data(self):
        r"""NOT differentiable"""
        for param_name, lora_name in self.params_with_lora.items():
            eval(f'self.{param_name}').data += self.merge_BA(param_name) * self.scaling

    def sub_lora_data(self):
        r"""NOT differentiable"""
        for param_name, lora_name in self.params_with_lora.items():
            eval(f'self.{param_name}').data -= self.merge_BA(param_name) * self.scaling

    def lora_train(self, mode: bool = True):
        if mode:
            if self.merged and self.r > 0:
            # Make sure that the weights are not merged
                self.sub_lora_data()
            self.merged = False
        else:
            if not self.merged and self.r > 0:
            # Merge the weights and mark it
                self.add_lora_data()
            self.merged = True 
            
#transformer part
class MultiheadAttention(nn.MultiheadAttention, LoRALayer):
    # LoRA implemented in a MultiheadAttention layer
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        enable_lora: list = ['q', 'k', 'v', 'o'],
        r: int = 0, 
        lora_alpha: int = 1, 
        **kwargs
    ):
        nn.MultiheadAttention.__init__(self, embed_dim, num_heads, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha)

        # Actual trainable parameters
        if self.r > 0:
            if 'o' in enable_lora:
                self.params_with_lora.update({'out_proj.weight': 'o'})

            if not self._qkv_same_embed_dim:
                for n in ['q', 'k', 'v']:
                    if n in enable_lora:
                        self.params_with_lora.update({f'{n}_proj_weight': n})
                self.register_lora_param()
                nn.MultiheadAttention._reset_parameters(self)
                self.init_lora_param()
            else:
                lora_name, enable_lora_bool = '', []
                for n in ['q', 'k', 'v']:
                    if n in enable_lora:
                        lora_name += n
                        enable_lora_bool.append(True)
                    else:
                        enable_lora_bool.append(False)
                self.params_with_lora.update({'in_proj_weight': lora_name})
                self.register_lora_param()
                nn.MultiheadAttention._reset_parameters(self)
                if 'o' in enable_lora:
                    self.init_lora_param_o()
                self.init_lora_param_qkv(enable_lora_bool)

    def init_lora_param_o(self):
        param_name, lora_name = 'out_proj.weight', 'o'
        if hasattr(self, f'{lora_name}_lora_A'):
            nn.init.kaiming_uniform_(eval(f'self.{lora_name}_lora_A'), a=math.sqrt(5))
            nn.init.zeros_(eval(f'self.{lora_name}_lora_B'))

    def init_lora_param_qkv(self, enable_lora_bool):
        lora_name = self.params_with_lora['in_proj_weight']
        nn.init.zeros_(eval(f'self.{lora_name}_lora_B'))
        dim = int(self.in_proj_weight.size()[1] / 3)
        for idx, enable in zip(range(3), enable_lora_bool):
            if enable:
                nn.init.kaiming_uniform_(eval(f'self.{lora_name}_lora_A')[:,idx*dim:(idx+1)*dim], a=math.sqrt(5))
            else:
                nn.init.zeros_(eval(f'self.{lora_name}_lora_A')[:,idx*dim:(idx+1)*dim])

    def train(self, mode: bool = True):
        nn.MultiheadAttention.train(self, mode)
        self.lora_train(mode)     

    def forward(self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            **kwargs):

        if self.r > 0 and not self.merged:
            self.merge_lora_param()
            result = nn.MultiheadAttention.forward(self, query, key, value, **kwargs)
            self.sub_lora_data()
            return result
        else:
            return nn.MultiheadAttention.forward(self, query, key, value, **kwargs)


class Conv2d(nn.Conv2d, LoRALayer):
    # LoRA implemented in a Conv2d layer
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        r: int = 0, 
        lora_alpha: int = 1, 
        **kwargs
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha)

        assert type(kernel_size) is int
        #if Lora is used, then....
        # Actual trainable parameters
        self.params_with_lora = {'weight': 'w'}
        if r > 0:
            self.w_lora_A = nn.Parameter(
                self.weight.new_zeros((r*kernel_size, in_channels*kernel_size))
            )
            self.w_lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels//self.groups*kernel_size, r*kernel_size))
            )
            #.....
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        nn.Conv2d.reset_parameters(self)
        self.init_lora_param()

    def train(self, mode: bool = True):
        nn.Conv2d.train(self, mode)      
        self.lora_train(mode)

    def forward(self, x: torch.Tensor, **kwargs):

        if self.r > 0 and not self.merged:
            self.merge_lora_param()
            result = nn.Conv2d.forward(self, x, **kwargs)
            self.sub_lora_data()
            return result
        else:
            return nn.Conv2d.forward(self, x, **kwargs)

def replace_lora(model, module_name, rank):
    for sub_module_name in model._modules:
        cuurent_module_name = sub_module_name if module_name == "" else module_name + "." + sub_module_name

        if len(model._modules[sub_module_name]._modules) > 1:
            replace_lora(model._modules[sub_module_name], cuurent_module_name, rank=rank)
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
                    # norm=model._modules[sub_module_name].norm,
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


# class LoraTrainer(FineTuningTrainer):
#     def __init__(self, cfg):
#         super().__init__(cfg)

#     @classmethod
#     def build_model(cls, cfg, is_finetuned=False):
#         model = super().build_model(cfg, is_finetuned)
#         replace_lora(model, "", rank=cfg.FINETUNE.LORA.RANK)
#         return model