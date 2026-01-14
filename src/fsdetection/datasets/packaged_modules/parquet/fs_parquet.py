from datasets.packaged_modules.parquet.parquet import Parquet

from ...fs_builder import ArrowBasedFSBuilder


class FSParquet(ArrowBasedFSBuilder, Parquet):
    pass