from datasets.packaged_modules.parquet.parquet import Parquet

from ....datasets import ArrowBasedFSBuilder


class FSParquet(ArrowBasedFSBuilder, Parquet):
    pass