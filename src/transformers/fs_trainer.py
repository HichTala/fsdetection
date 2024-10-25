from transformers import Trainer


class FSTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def freeze_model(self,
                     freeze_modules=['backbone'],
                     freeze_mode='all',
                     backbone_freeze_at=0,
                     distributed=False):
        """
        Freeze model parameters for various modules
        When backbone_freeze == 0, freeze all backbone parameters
        Otherwise freeze up to res_#backbone_freeze_at layer.
        """

        def freeze(model, freeze_at=0):
            for idx, param in enumerate(model.parameters()):
                if freeze_at >= idx:
                    param.requires_grad = False

        def freeze_model_filtered(model, unfrozen_names=[], unfrozen_type=None):
            for param_name, param in model.named_parameters():
                if unfrozen_type is None:  # freeze all param but unfrozen_names
                    if all([(name not in param_name) for name in unfrozen_names]):
                        param.requires_grad = False
                else:  # keep only a type of params unfrozen within unfrozen_names, all others are frozen
                    if all([(name not in param_name) and (unfrozen_type not in param_name) for name in unfrozen_names]):
                        param.requires_grad = False

        def freeze_model_process(model):
            if 'backbone' in freeze_modules:
                if freeze_mode == 'all':
                    if backbone_freeze_at > 0:
                        freeze(model.backbone[0], backbone_freeze_at)
                    else:
                        freeze_model_filtered(model.backbone)
                elif freeze_mode == 'half':
                    freeze(model.backbone[0], int(len(list(model.backbone.parameters())) / 2))
                elif freeze_mode == 'bias':
                    freeze_model_filtered(model.backbone, ['bias'])
                elif freeze_mode == 'norm':
                    freeze_model_filtered(model.backbone, ['norm'])

        if distributed:
            freeze_model_process(self.model.module)
        else:
            freeze_model_process(self.model)

    def _get_train_sampler(self):
        breakpoint()
        return