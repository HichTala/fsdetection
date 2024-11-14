import random
from typing import Optional, Union

import numpy as np
from datasets import Dataset
import pyarrow as pa
import pyarrow.compute as pc
from datasets.table import MemoryMappedTable, InMemoryTable


class FSDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shots = None

    def sampling(self, shots: int, seed: Optional[int] = None):

        if self._indices is None:
            self._indices = InMemoryTable.from_pydict({"indices": pa.array(np.arange(len(self)))})

        self.shots = shots
        category_list = pc.struct_field(self._data['objects'], 'category')

        indices = set(self._indices['indices'].to_pylist()) if self._indices is not None else None

        if seed is None:
            _, seed, pos, *_ = np.random.get_state()
            seed = seed[pos] if pos < 624 else seed[0]
            _ = np.random.random()  # do 1 step of rng
        generator = np.random.default_rng(seed)

        true_indices = {}
        for i, cats in enumerate(category_list):
            for cat in cats:
                py_cat = cat.as_py()
                if py_cat not in true_indices.keys():
                    true_indices[py_cat] = []
                if indices is None  or i in indices:
                    true_indices[py_cat].append(i)

        selected_indices = set()
        for cat in true_indices.keys():
            selected_indices = selected_indices.union(
                set(generator.choice(true_indices[cat], size=min(self.shots, len(true_indices[cat])), replace=False)))

        class_table = pc.is_in(self._indices['indices'], value_set=pa.array(selected_indices, type=pa.uint64()))

        self._indices = self._indices.filter(class_table)

    def _new_dataset_with_indices(
            self,
            indices_cache_file_name: Optional[str] = None,
            indices_buffer: Optional[pa.Buffer] = None,
            fingerprint: Optional[str] = None,
    ) -> "Union[Dataset, FSDataset]":
        """Return a new Dataset obtained by adding indices (provided in indices_cache_file_name or in a buffer) to the
        current Dataset.
        """

        if indices_cache_file_name is None and indices_buffer is None:
            raise ValueError("At least one of indices_cache_file_name or indices_buffer must be provided.")

        if fingerprint is None:
            raise ValueError("please specify a fingerprint for the dataset with indices")

        if indices_cache_file_name is not None:
            indices_table = MemoryMappedTable.from_file(indices_cache_file_name)
        else:
            indices_table = InMemoryTable.from_buffer(indices_buffer)

        # Return new Dataset object
        # don't forget to copy the objects
        if self.split == "train":
            return FSDataset(
                self._data,
                info=self.info.copy(),
                split=self.split,
                indices_table=indices_table,
                fingerprint=fingerprint,
            )

        return Dataset(
            self._data,
            info=self.info.copy(),
            split=self.split,
            indices_table=indices_table,
            fingerprint=fingerprint,
        )
