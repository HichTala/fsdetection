from transformers import Trainer


class FSTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self):
        breakpoint()
        return