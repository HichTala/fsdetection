from typing import Union

from datasets import DatasetBuilder, Split, ArrowBasedBuilder, Dataset
from datasets.arrow_reader import ArrowReader, ReadInstruction

from src.datasets.fs_arrow_dataset import FSDataset


class FSDatasetBuilder(DatasetBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _as_dataset(self, split: Union[ReadInstruction, Split] = Split.TRAIN, in_memory: bool = False) -> Union[Dataset, FSDataset]:
        """Constructs a `Dataset`.

        This is the internal implementation to overwrite called when user calls
        `as_dataset`. It should read the pre-processed datasets files and generate
        the `Dataset` object.

        Args:
            split (`datasets.Split`):
                which subset of the data to read.
            in_memory (`bool`, defaults to `False`):
                Whether to copy the data in-memory.

        Returns:
            `Dataset`
        """
        cache_dir = self._fs._strip_protocol(self._output_dir)
        dataset_name = self.dataset_name
        if self._check_legacy_cache():
            dataset_name = self.name
        dataset_kwargs = ArrowReader(cache_dir, self.info).read(
            name=dataset_name,
            instructions=split,
            split_infos=self.info.splits.values(),
            in_memory=in_memory,
        )
        fingerprint = self._get_dataset_fingerprint(split)
        if split == 'train':
            return FSDataset(fingerprint=fingerprint, **dataset_kwargs)
        else:
            return Dataset(fingerprint=fingerprint, **dataset_kwargs)

class ArrowBasedFSBuilder(FSDatasetBuilder, ArrowBasedBuilder):
    pass
