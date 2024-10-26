import inspect

from datasets.packaged_modules import _PACKAGED_DATASETS_MODULES, _hash_python_lines

from src.datasets.packaged_modules.parquet import fs_parquet

_PACKAGED_DATASETS_MODULES['parquet'] = (fs_parquet.__name__, _hash_python_lines(inspect.getsource(fs_parquet).splitlines()))