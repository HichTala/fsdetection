from datasets import Dataset

class FSDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)