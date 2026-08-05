import importlib
import inspect
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Union, Sequence, Mapping, Dict, Type

from datasets import Features, DownloadConfig, DownloadMode, Version, Split, VerificationMode, \
    DatasetDict, Dataset, IterableDatasetDict, IterableDataset, config
try:
    from datasets.features.features import _fix_for_backward_compatible_features
except ImportError:
    def _fix_for_backward_compatible_features(features):
        return features
from datasets.load import dataset_module_factory, DatasetModule

try:
    from datasets.load import configure_builder_class
except ImportError:
    # Fallback implementation for older versions of Hugging Face datasets
    def snakecase_to_camelcase(name: str) -> str:
        return "".join(word.capitalize() for word in re.split(r"[^a-zA-Z0-9]", name) if word)


    def configure_builder_class(
            builder_cls: Type,
            builder_configs: list,
            default_config_name: Optional[str],
            dataset_name: str,
    ) -> Type:
        """Dynamically create a builder class with custom builder configs."""

        class ConfiguredDatasetBuilder(builder_cls):
            BUILDER_CONFIGS = builder_configs
            DEFAULT_CONFIG_NAME = default_config_name
            __module__ = builder_cls.__module__

            def __reduce__(self):
                return (
                    _InitializeConfiguredDatasetBuilder(),
                    (builder_cls, self.BUILDER_CONFIGS, self.DEFAULT_CONFIG_NAME, dataset_name),
                    self.__dict__.copy(),
                )

        # Give the dynamically created class a readable name
        class_name = f"{builder_cls.__name__.capitalize()}{snakecase_to_camelcase(dataset_name)}"
        ConfiguredDatasetBuilder.__name__ = class_name
        ConfiguredDatasetBuilder.__qualname__ = class_name

        return ConfiguredDatasetBuilder


    class _InitializeConfiguredDatasetBuilder:
        def __call__(self, builder_cls, builder_configs, default_config_name, name):
            obj = _InitializeConfiguredDatasetBuilder()
            obj.__class__ = configure_builder_class(
                builder_cls,
                builder_configs=builder_configs,
                default_config_name=default_config_name,
                dataset_name=name,
            )
            return obj

from datasets.utils.info_utils import is_small_dataset
from datasets.utils.logging import get_logger

from .fs_builder import FSDatasetBuilder

from datasets.packaged_modules import (
    _EXTENSION_TO_MODULE,
)

from .packaged_modules import _PACKAGED_DATASETS_MODULES
from .fs_arrow_dataset import FSDataset

logger = get_logger(__name__)

def import_main_fs_class(module_path):
    """Import a module at module_path and return its main class: a DatasetBuilder"""
    module = importlib.import_module(module_path)
    # Find the main class in our imported module
    module_main_cls = None
    for name, obj in module.__dict__.items():
        if inspect.isclass(obj) and issubclass(obj, FSDatasetBuilder):
            if inspect.isabstract(obj):
                continue
            module_main_cls = obj
            obj_module = inspect.getmodule(obj)
            if obj_module is not None and module == obj_module:
                break

    return module_main_cls


def get_fs_dataset_builder_class(
    dataset_module, dataset_name = None
):
    builder_cls = import_main_fs_class(dataset_module.module_path)
    if getattr(dataset_module, "builder_configs_parameters.default_config_name", None):
        dataset_name = dataset_name or dataset_module.builder_kwargs.get("dataset_name")
        if dataset_name is None:
            raise ValueError("dataset_name should be specified but got None")
        builder_cls = configure_builder_class(
            builder_cls,
            builder_configs=dataset_module.builder_configs_parameters.builder_configs,
            default_config_name=dataset_module.builder_configs_parameters.default_config_name,
            dataset_name=dataset_name,
        )
    return builder_cls

def load_fs_dataset_builder(
    path,
    name = None,
    data_dir = None,
    data_files = None,
    cache_dir = None,
    features = None,
    download_config = None,
    download_mode = None,
    revision = None,
    token = None,
    storage_options = None,
    **config_kwargs,
) :
    """Load a dataset builder which can be used to:

    - Inspect general information that is required to build a dataset (cache directory, config, dataset info, features, data files, etc.)
    - Download and prepare the dataset as Arrow files in the cache
    - Get a streaming dataset without downloading or caching anything

    You can find the list of datasets on the [Hub](https://huggingface.co/datasets) or with [`huggingface_hub.list_datasets`].

    A dataset is a directory that contains some data files in generic formats (JSON, CSV, Parquet, etc.) and possibly
    in a generic structure (Webdataset, ImageFolder, AudioFolder, VideoFolder, etc.)

    Args:

        path (`str`):
            Path or name of the dataset.

            - if `path` is a dataset repository on the HF hub (list all available datasets with [`huggingface_hub.list_datasets`])
              -> load the dataset builder from supported files in the repository (csv, json, parquet, etc.)
              e.g. `'username/dataset_name'`, a dataset repository on the HF hub containing the data files.

            - if `path` is a local directory
              -> load the dataset builder from supported files in the directory (csv, json, parquet, etc.)
              e.g. `'./path/to/directory/with/my/csv/data'`.

            - if `path` is the name of a dataset builder and `data_files` or `data_dir` is specified
              (available builders are "json", "csv", "parquet", "arrow", "text", "xml", "webdataset", "imagefolder", "audiofolder", "videofolder")
              -> load the dataset builder from the files in `data_files` or `data_dir`
              e.g. `'parquet'`.

        name (`str`, *optional*):
            Defining the name of the dataset configuration.
        data_dir (`str`, *optional*):
            Defining the `data_dir` of the dataset configuration. If specified for the generic builders (csv, text etc.) or the Hub datasets and `data_files` is `None`,
            the behavior is equal to passing `os.path.join(data_dir, **)` as `data_files` to reference all the files in a directory.
        data_files (`str` or `Sequence` or `Mapping`, *optional*):
            Path(s) to source data file(s).
        cache_dir (`str`, *optional*):
            Directory to read/write data. Defaults to `"~/.cache/huggingface/datasets"`.
        features ([`Features`], *optional*):
            Set the features type to use for this dataset.
        download_config ([`DownloadConfig`], *optional*):
            Specific download configuration parameters.
        download_mode ([`DownloadMode`] or `str`, defaults to `REUSE_DATASET_IF_EXISTS`):
            Download/generate mode.
        revision ([`Version`] or `str`, *optional*):
            Version of the dataset to load.
            As datasets have their own git repository on the Datasets Hub, the default version "main" corresponds to their "main" branch.
            You can specify a different version than the default "main" by using a commit SHA or a git tag of the dataset repository.
        token (`str` or `bool`, *optional*):
            Optional string or boolean to use as Bearer token for remote files on the Datasets Hub.
            If `True`, or not specified, will get token from `"~/.huggingface"`.
        storage_options (`dict`, *optional*, defaults to `None`):
            **Experimental**. Key/value pairs to be passed on to the dataset file-system backend, if any.

            <Added version="2.11.0"/>

        **config_kwargs (additional keyword arguments):
            Keyword arguments to be passed to the [`BuilderConfig`]
            and used in the [`DatasetBuilder`].

    Returns:
        [`DatasetBuilder`]

    Example:

    ```py
    >>> from datasets import load_dataset_builder
    >>> ds_builder = load_dataset_builder('cornell-movie-review-data/rotten_tomatoes')
    >>> ds_builder.info.features
    {'label': ClassLabel(names=['neg', 'pos']),
     'text': Value('string')}
    ```
    """
    download_mode = DownloadMode(download_mode or DownloadMode.REUSE_DATASET_IF_EXISTS)
    if token is not None:
        download_config = download_config.copy() if download_config else DownloadConfig()
        download_config.token = token
    if storage_options is not None:
        download_config = download_config.copy() if download_config else DownloadConfig()
        download_config.storage_options.update(storage_options)
    if features is not None:
        features = _fix_for_backward_compatible_features(features)
    dataset_module = dataset_module_factory(
        path,
        revision=revision,
        download_config=download_config,
        download_mode=download_mode,
        data_dir=data_dir,
        data_files=data_files,
        cache_dir=cache_dir,
    )
    # Get dataset builder class
    builder_kwargs = dataset_module.builder_kwargs
    data_dir = builder_kwargs.pop("data_dir", data_dir)
    data_files = builder_kwargs.pop("data_files", data_files)
    config_name = builder_kwargs.pop(
        "config_name", name or getattr(dataset_module, "builder_configs_parameters.default_config_name", None)
    )
    dataset_name = builder_kwargs.pop("dataset_name", None)
    info = dataset_module.dataset_infos.get(config_name) if hasattr(dataset_module, "dataset_infos") and dataset_module.dataset_infos else None

    if (
        path in _PACKAGED_DATASETS_MODULES
        and data_files is None
        and dataset_module.builder_configs_parameters.builder_configs[0].data_files is None
    ):
        error_msg = f"Please specify the data files or data directory to load for the {path} dataset builder."
        example_extensions = [
            extension for extension in _EXTENSION_TO_MODULE if _EXTENSION_TO_MODULE[extension] == path
        ]
        if example_extensions:
            error_msg += f'\nFor example `data_files={{"train": "path/to/data/train/*.{example_extensions[0]}"}}`'
        raise ValueError(error_msg)

    builder_cls = get_fs_dataset_builder_class(dataset_module, dataset_name=dataset_name)
    # Instantiate the dataset builder
    if  "hash" in builder_kwargs:
        builder_kwargs.pop("hash")
    if  "info" in builder_kwargs:
        builder_kwargs.pop("info")
    if  "features" in builder_kwargs:
        builder_kwargs.pop("features")
    builder_instance: FSDatasetBuilder = builder_cls(
        cache_dir=cache_dir,
        dataset_name=dataset_name,
        config_name=config_name,
        data_dir=data_dir,
        data_files=data_files,
        hash=dataset_module.hash,
        info=info,
        features=features,
        token=token,
        storage_options=storage_options,
        **builder_kwargs,
        **config_kwargs,
    )
    if hasattr(builder_instance, "_use_legacy_cache_dir_if_possible"):
        builder_instance._use_legacy_cache_dir_if_possible(dataset_module)

    return builder_instance

def load_fs_dataset(
    path,
    name = None,
    data_dir = None,
    data_files = None,
    split = None,
    cache_dir = None,
    features = None,
    download_config = None,
    download_mode = None,
    verification_mode = None,
    keep_in_memory = None,
    save_infos = False,
    revision = None,
    token = None,
    streaming = False,
    num_proc = None,
    storage_options = None,
    **config_kwargs,
) :
    """Load a dataset from the Hugging Face Hub, or a local dataset.

    You can find the list of datasets on the [Hub](https://huggingface.co/datasets) or with [`huggingface_hub.list_datasets`].

    A dataset is a directory that contains some data files in generic formats (JSON, CSV, Parquet, etc.) and possibly
    in a generic structure (Webdataset, ImageFolder, AudioFolder, VideoFolder, etc.)

    This function does the following under the hood:

        1. Load a dataset builder:

            * Find the most common data format in the dataset and pick its associated builder (JSON, CSV, Parquet, Webdataset, ImageFolder, AudioFolder, etc.)
            * Find which file goes into which split (e.g. train/test) based on file and directory names or on the YAML configuration
            * It is also possible to specify `data_files` manually, and which dataset builder to use (e.g. "parquet").

        2. Run the dataset builder:

            In the general case:

            * Download the data files from the dataset if they are not already available locally or cached.
            * Process and cache the dataset in typed Arrow tables for caching.

                Arrow table are arbitrarily long, typed tables which can store nested objects and be mapped to numpy/pandas/python generic types.
                They can be directly accessed from disk, loaded in RAM or even streamed over the web.

            In the streaming case:

            * Don't download or cache anything. Instead, the dataset is lazily loaded and will be streamed on-the-fly when iterating on it.

        3. Return a dataset built from the requested splits in `split` (default: all).

    Args:

        path (`str`):
            Path or name of the dataset.

            - if `path` is a dataset repository on the HF hub (list all available datasets with [`huggingface_hub.list_datasets`])
              -> load the dataset from supported files in the repository (csv, json, parquet, etc.)
              e.g. `'username/dataset_name'`, a dataset repository on the HF hub containing the data files.

            - if `path` is a local directory
              -> load the dataset from supported files in the directory (csv, json, parquet, etc.)
              e.g. `'./path/to/directory/with/my/csv/data'`.

            - if `path` is the name of a dataset builder and `data_files` or `data_dir` is specified
              (available builders are "json", "csv", "parquet", "arrow", "text", "xml", "webdataset", "imagefolder", "audiofolder", "videofolder")
              -> load the dataset from the files in `data_files` or `data_dir`
              e.g. `'parquet'`.

        name (`str`, *optional*):
            Defining the name of the dataset configuration.
        data_dir (`str`, *optional*):
            Defining the `data_dir` of the dataset configuration. If specified for the generic builders (csv, text etc.) or the Hub datasets and `data_files` is `None`,
            the behavior is equal to passing `os.path.join(data_dir, **)` as `data_files` to reference all the files in a directory.
        data_files (`str` or `Sequence` or `Mapping`, *optional*):
            Path(s) to source data file(s).
        split (`Split` or `str`):
            Which split of the data to load.
            If `None`, will return a `dict` with all splits (typically `datasets.Split.TRAIN` and `datasets.Split.TEST`).
            If given, will return a single Dataset.
            Splits can be combined and specified like in tensorflow-datasets.
        cache_dir (`str`, *optional*):
            Directory to read/write data. Defaults to `"~/.cache/huggingface/datasets"`.
        features (`Features`, *optional*):
            Set the features type to use for this dataset.
        download_config ([`DownloadConfig`], *optional*):
            Specific download configuration parameters.
        download_mode ([`DownloadMode`] or `str`, defaults to `REUSE_DATASET_IF_EXISTS`):
            Download/generate mode.
        verification_mode ([`VerificationMode`] or `str`, defaults to `BASIC_CHECKS`):
            Verification mode determining the checks to run on the downloaded/processed dataset information (checksums/size/splits/...).

            <Added version="2.9.1"/>
        keep_in_memory (`bool`, defaults to `None`):
            Whether to copy the dataset in-memory. If `None`, the dataset
            will not be copied in-memory unless explicitly enabled by setting `datasets.config.IN_MEMORY_MAX_SIZE` to
            nonzero. See more details in the [improve performance](../cache#improve-performance) section.
        revision ([`Version`] or `str`, *optional*):
            Version of the dataset to load.
            As datasets have their own git repository on the Datasets Hub, the default version "main" corresponds to their "main" branch.
            You can specify a different version than the default "main" by using a commit SHA or a git tag of the dataset repository.
        token (`str` or `bool`, *optional*):
            Optional string or boolean to use as Bearer token for remote files on the Datasets Hub.
            If `True`, or not specified, will get token from `"~/.huggingface"`.
        streaming (`bool`, defaults to `False`):
            If set to `True`, don't download the data files. Instead, it streams the data progressively while
            iterating on the dataset. An [`IterableDataset`] or [`IterableDatasetDict`] is returned instead in this case.

            Note that streaming works for datasets that use data formats that support being iterated over like txt, csv, jsonl for example.
            Json files may be downloaded completely. Also streaming from remote zip or gzip files is supported but other compressed formats
            like rar and xz are not yet supported. The tgz format doesn't allow streaming.
        num_proc (`int`, *optional*, defaults to `None`):
            Number of processes when downloading and generating the dataset locally.
            Multiprocessing is disabled by default.

            <Added version="2.7.0"/>
        storage_options (`dict`, *optional*, defaults to `None`):
            **Experimental**. Key/value pairs to be passed on to the dataset file-system backend, if any.

            <Added version="2.11.0"/>
        **config_kwargs (additional keyword arguments):
            Keyword arguments to be passed to the `BuilderConfig`
            and used in the [`DatasetBuilder`].

    Returns:
        [`Dataset`] or [`DatasetDict`]:
        - if `split` is not `None`: the dataset requested,
        - if `split` is `None`, a [`~datasets.DatasetDict`] with each split.

        or [`IterableDataset`] or [`IterableDatasetDict`]: if `streaming=True`

        - if `split` is not `None`, the dataset is requested
        - if `split` is `None`, a [`~datasets.streaming.IterableDatasetDict`] with each split.

    Example:

    Load a dataset from the Hugging Face Hub:

    ```py
    >>> from datasets import load_dataset
    >>> ds = load_dataset('cornell-movie-review-data/rotten_tomatoes', split='train')

    # Load a subset or dataset configuration (here 'sst2')
    >>> from datasets import load_dataset
    >>> ds = load_dataset('nyu-mll/glue', 'sst2', split='train')

    # Manual mapping of data files to splits
    >>> data_files = {'train': 'train.csv', 'test': 'test.csv'}
    >>> ds = load_dataset('namespace/your_dataset_name', data_files=data_files)

    # Manual selection of a directory to load
    >>> ds = load_dataset('namespace/your_dataset_name', data_dir='folder_name')
    ```

    Load a local dataset:

    ```py
    # Load a CSV file
    >>> from datasets import load_dataset
    >>> ds = load_dataset('csv', data_files='path/to/local/my_dataset.csv')

    # Load a JSON file
    >>> from datasets import load_dataset
    >>> ds = load_dataset('json', data_files='path/to/local/my_dataset.json')
    ```

    Load an [`~datasets.IterableDataset`]:

    ```py
    >>> from datasets import load_dataset
    >>> ds = load_dataset('cornell-movie-review-data/rotten_tomatoes', split='train', streaming=True)
    ```

    Load an image dataset with the `ImageFolder` dataset builder:

    ```py
    >>> from datasets import load_dataset
    >>> ds = load_dataset('imagefolder', data_dir='/path/to/images', split='train')
    ```
    """
    if "trust_remote_code" in config_kwargs:
        if config_kwargs.pop("trust_remote_code"):
            logger.error(
                "`trust_remote_code` is not supported anymore.\n"
                f"Please check that the Hugging Face dataset '{path}' isn't based on a loading script and remove `trust_remote_code`.\n"
                "If the dataset is based on a loading script, please ask the dataset author to remove it and convert it to a standard format like Parquet."
            )
    if data_files is not None and not data_files:
        raise ValueError(f"Empty 'data_files': '{data_files}'. It should be either non-empty or None (default).")
    if Path(path, config.DATASET_STATE_JSON_FILENAME).exists():
        raise ValueError(
            "You are trying to load a dataset that was saved using `save_to_disk`. "
            "Please use `load_from_disk` instead."
        )

    if streaming and num_proc is not None:
        raise NotImplementedError(
            "Loading a streaming dataset in parallel with `num_proc` is not implemented. "
            "To parallelize streaming, you can wrap the dataset with a PyTorch DataLoader using `num_workers` > 1 instead."
        )

    download_mode = DownloadMode(download_mode or DownloadMode.REUSE_DATASET_IF_EXISTS)
    verification_mode = VerificationMode(
        (verification_mode or VerificationMode.BASIC_CHECKS) if not save_infos else VerificationMode.ALL_CHECKS
    )

    # Create a dataset builder
    builder_instance = load_fs_dataset_builder(
        path=path,
        name=name,
        data_dir=data_dir,
        data_files=data_files,
        cache_dir=cache_dir,
        features=features,
        download_config=download_config,
        download_mode=download_mode,
        revision=revision,
        token=token,
        storage_options=storage_options,
        **config_kwargs,
    )

    # Return iterable dataset in case of streaming
    if streaming:
        return builder_instance.as_streaming_dataset(split=split)

    builder_instance._cache_dir_root = str(config.HF_DATASETS_CACHE)
    builder_instance.config, builder_instance.config_id = builder_instance._create_builder_config(config_name=None, custom_features=features, **config_kwargs)
    builder_instance._cache_dir = builder_instance._build_cache_dir()

    # Download and prepare data
    builder_instance.download_and_prepare(
        download_config=download_config,
        download_mode=download_mode,
        verification_mode=verification_mode,
        num_proc=num_proc,
        storage_options=storage_options,
    )

    # Build dataset for splits
    keep_in_memory = (
        keep_in_memory if keep_in_memory is not None else is_small_dataset(builder_instance.info.dataset_size)
    )
    ds = builder_instance.as_dataset(split=split, verification_mode=verification_mode, in_memory=keep_in_memory)

    return ds