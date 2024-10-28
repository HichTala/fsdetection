import random

import numpy as np
from datasets import Dataset
import pyarrow as pa
import pyarrow.compute as pc


class FSDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        shots = 5

        category_list = pc.struct_field(self._data['objects'], 'category')

        true_indices = {}
        for i, cats in enumerate(category_list):
            for cat in cats:
                py_cat = cat.as_py()
                if py_cat not in true_indices.keys():
                    true_indices[py_cat] = []
                true_indices[py_cat].append(i)

        selected_indices = set()
        for cat in true_indices.keys():
            selected_indices = selected_indices.union(set(random.sample(true_indices[cat], min(shots, len(true_indices[cat])))))
        print(selected_indices)

        class_table = np.full(len(self._data), False)
        for i in selected_indices:
            class_table[i] = True

        self._data = self._data.filter(pa.array(class_table))
