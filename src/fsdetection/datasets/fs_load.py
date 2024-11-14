import importlib
import inspect
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Union, Sequence, Mapping, Dict, Type

from datasets import Features, DownloadConfig, DownloadMode, Version, Split, VerificationMode, \
    DatasetDict, Dataset, IterableDatasetDict, IterableDataset, config
from datasets.load import dataset_module_factory, configure_builder_class, \
    DatasetModule
from datasets.utils.info_utils import is_small_dataset
from datasets.utils.py_utils import lock_importable_file

from src.fsdetection.datasets.fs_builder import FSDatasetBuilder

from datasets.packaged_modules import (
    _EXTENSION_TO_MODULE,
)

from src.fsdetection.datasets.packaged_modules import _PACKAGED_DATASETS_MODULES



def import_main_fs_class(module_path) -> Optional[Type[FSDatasetBuilder]]:
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
    dataset_module: "DatasetModule", dataset_name: Optional[str] = None
) -> Type[FSDatasetBuilder]:
    with lock_importable_file(
        dataset_module.importable_file_path
    ) if dataset_module.importable_file_path else nullcontext():
        builder_cls = import_main_fs_class(dataset_module.module_path)
    if dataset_module.builder_configs_parameters.builder_configs:
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
    path: str,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
    cache_dir: Optional[str] = None,
    features: Optional[Features] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[Union[DownloadMode, str]] = None,
    revision: Optional[Union[str, Version]] = None,
    token: Optional[Union[bool, str]] = None,
    storage_options: Optional[Dict] = None,
    trust_remote_code: Optional[bool] = None,
    _require_default_config_name=True,
    **config_kwargs,
) -> FSDatasetBuilder:
    """Load a dataset builder from the Hugging Face Hub, or a local dataset. A dataset builder can be used to inspect general information that is required to build a dataset (cache directory, config, dataset info, etc.)
    without downloading the dataset itself.

    You can find the list of datasets on the [Hub](https://huggingface.co/datasets) or with [`huggingface_hub.list_datasets`].

    A dataset is a directory that contains:

    - some data files in generic formats (JSON, CSV, Parquet, text, etc.)
    - and optionally a dataset script, if it requires some code to read the data files. This is used to load any kind of formats or structures.

    Note that dataset scripts can also download and read data files from anywhere - in case your data files already exist online.

    Args:

        path (`str`):
            Path or name of the dataset.
            Depending on `path`, the dataset builder that is used comes from a generic dataset script (JSON, CSV, Parquet, text etc.) or from the dataset script (a python file) inside the dataset directory.

            For local datasets:

            - if `path` is a local directory (containing data files only)
              -> load a generic dataset builder (csv, json, text etc.) based on the content of the directory
              e.g. `'./path/to/directory/with/my/csv/data'`.
            - if `path` is a local dataset script or a directory containing a local dataset script (if the script has the same name as the directory)
              -> load the dataset builder from the dataset script
              e.g. `'./dataset/squad'` or `'./dataset/squad/squad.py'`.

            For datasets on the Hugging Face Hub (list all available datasets with [`huggingface_hub.list_datasets`])

            - if `path` is a dataset repository on the HF hub (containing data files only)
              -> load a generic dataset builder (csv, text etc.) based on the content of the repository
              e.g. `'username/dataset_name'`, a dataset repository on the HF hub containing your data files.
            - if `path` is a dataset repository on the HF hub with a dataset script (if the script has the same name as the directory)
              -> load the dataset builder from the dataset script in the dataset repository
              e.g. `glue`, `squad`, `'username/dataset_name'`, a dataset repository on the HF hub containing a dataset script `'dataset_name.py'`.

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
            Version of the dataset script to load.
            As datasets have their own git repository on the Datasets Hub, the default version "main" corresponds to their "main" branch.
            You can specify a different version than the default "main" by using a commit SHA or a git tag of the dataset repository.
        token (`str` or `bool`, *optional*):
            Optional string or boolean to use as Bearer token for remote files on the Datasets Hub.
            If `True`, or not specified, will get token from `"~/.huggingface"`.
        storage_options (`dict`, *optional*, defaults to `None`):
            **Experimental**. Key/value pairs to be passed on to the dataset file-system backend, if any.

            <Added version="2.11.0"/>
        trust_remote_code (`bool`, defaults to `False`):
            Whether or not to allow for datasets defined on the Hub using a dataset script. This option
            should only be set to `True` for repositories you trust and in which you have read the code, as it will
            execute code present on the Hub on your local machine.

            <Added version="2.16.0"/>

            <Changed version="2.20.0">

            `trust_remote_code` defaults to `False` if not specified.

            </Changed>

        **config_kwargs (additional keyword arguments):
            Keyword arguments to be passed to the [`BuilderConfig`]
            and used in the [`DatasetBuilder`].

    Returns:
        [`DatasetBuilder`]

    Example:

    ```py
    >>> from datasets import load_dataset_builder
    >>> ds_builder = load_dataset_builder('rotten_tomatoes')
    >>> ds_builder.info.features
    {'label': ClassLabel(num_classes=2, names=['neg', 'pos'], id=None),
     'text': Value(dtype='string', id=None)}
    ```
    """
    download_mode = DownloadMode(download_mode or DownloadMode.REUSE_DATASET_IF_EXISTS)
    if token is not None:
        download_config = download_config.copy() if download_config else DownloadConfig()
        download_config.token = token
    if storage_options is not None:
        download_config = download_config.copy() if download_config else DownloadConfig()
        download_config.storage_options.update(storage_options)
    dataset_module = dataset_module_factory(
        path,
        revision=revision,
        download_config=download_config,
        download_mode=download_mode,
        data_dir=data_dir,
        data_files=data_files,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        _require_default_config_name=_require_default_config_name,
        _require_custom_configs=bool(config_kwargs),
    )
    # Get dataset builder class from the processing script
    builder_kwargs = dataset_module.builder_kwargs
    data_dir = builder_kwargs.pop("data_dir", data_dir)
    data_files = builder_kwargs.pop("data_files", data_files)
    config_name = builder_kwargs.pop(
        "config_name", name or dataset_module.builder_configs_parameters.default_config_name
    )
    dataset_name = builder_kwargs.pop("dataset_name", None)
    info = dataset_module.dataset_infos.get(config_name) if dataset_module.dataset_infos else None

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
    builder_instance._use_legacy_cache_dir_if_possible(dataset_module)

    return builder_instance

def load_fs_dataset(
    path: str,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
    split: Optional[Union[str, Split]] = None,
    cache_dir: Optional[str] = None,
    features: Optional[Features] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[Union[DownloadMode, str]] = None,
    verification_mode: Optional[Union[VerificationMode, str]] = None,
    keep_in_memory: Optional[bool] = None,
    save_infos: bool = False,
    revision: Optional[Union[str, Version]] = None,
    token: Optional[Union[bool, str]] = None,
    streaming: bool = False,
    num_proc: Optional[int] = None,
    storage_options: Optional[Dict] = None,
    trust_remote_code: bool = None,
    **config_kwargs,
) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
    """Load a dataset from the Hugging Face Hub, or a local dataset.

    You can find the list of datasets on the [Hub](https://huggingface.co/datasets) or with [`huggingface_hub.list_datasets`].

    A dataset is a directory that contains:

    - some data files in generic formats (JSON, CSV, Parquet, text, etc.).
    - and optionally a dataset script, if it requires some code to read the data files. This is used to load any kind of formats or structures.

    Note that dataset scripts can also download and read data files from anywhere - in case your data files already exist online.

    This function does the following under the hood:

        1. Download and import in the library the dataset script from `path` if it's not already cached inside the library.

            If the dataset has no dataset script, then a generic dataset script is imported instead (JSON, CSV, Parquet, text, etc.)

            Dataset scripts are small python scripts that define dataset builders. They define the citation, info and format of the dataset,
            contain the path or URL to the original data files and the code to load examples from the original data files.

            You can find the complete list of datasets in the Datasets [Hub](https://huggingface.co/datasets).

        2. Run the dataset script which will:

            * Download the dataset file from the original URL (see the script) if it's not already available locally or cached.
            * Process and cache the dataset in typed Arrow tables for caching.

                Arrow table are arbitrarily long, typed tables which can store nested objects and be mapped to numpy/pandas/python generic types.
                They can be directly accessed from disk, loaded in RAM or even streamed over the web.

        3. Return a dataset built from the requested splits in `split` (default: all).

    It also allows to load a dataset from a local directory or a dataset repository on the Hugging Face Hub without dataset script.
    In this case, it automatically loads all the data files from the directory or the dataset repository.

    Args:

        path (`str`):
            Path or name of the dataset.
            Depending on `path`, the dataset builder that is used comes from a generic dataset script (JSON, CSV, Parquet, text etc.) or from the dataset script (a python file) inside the dataset directory.

            For local datasets:

            - if `path` is a local directory (containing data files only)
              -> load a generic dataset builder (csv, json, text etc.) based on the content of the directory
              e.g. `'./path/to/directory/with/my/csv/data'`.
            - if `path` is a local dataset script or a directory containing a local dataset script (if the script has the same name as the directory)
              -> load the dataset builder from the dataset script
              e.g. `'./dataset/squad'` or `'./dataset/squad/squad.py'`.

            For datasets on the Hugging Face Hub (list all available datasets with [`huggingface_hub.list_datasets`])

            - if `path` is a dataset repository on the HF hub (containing data files only)
              -> load a generic dataset builder (csv, text etc.) based on the content of the repository
              e.g. `'username/dataset_name'`, a dataset repository on the HF hub containing your data files.
            - if `path` is a dataset repository on the HF hub with a dataset script (if the script has the same name as the directory)
              -> load the dataset builder from the dataset script in the dataset repository
              e.g. `glue`, `squad`, `'username/dataset_name'`, a dataset repository on the HF hub containing a dataset script `'dataset_name.py'`.

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
        save_infos (`bool`, defaults to `False`):
            Save the dataset information (checksums/size/splits/...).
        revision ([`Version`] or `str`, *optional*):
            Version of the dataset script to load.
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
        trust_remote_code (`bool`, defaults to `False`):
            Whether or not to allow for datasets defined on the Hub using a dataset script. This option
            should only be set to `True` for repositories you trust and in which you have read the code, as it will
            execute code present on the Hub on your local machine.

            <Added version="2.16.0"/>

            <Changed version="2.20.0">

            `trust_remote_code` defaults to `False` if not specified.

            </Changed>

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
    >>> ds = load_dataset('rotten_tomatoes', split='train')

    # Map data files to splits
    >>> data_files = {'train': 'train.csv', 'test': 'test.csv'}
    >>> ds = load_dataset('namespace/your_dataset_name', data_files=data_files)
    ```

    Load a local dataset:

    ```py
    # Load a CSV file
    >>> from datasets import load_dataset
    >>> ds = load_dataset('csv', data_files='path/to/local/my_dataset.csv')

    # Load a JSON file
    >>> from datasets import load_dataset
    >>> ds = load_dataset('json', data_files='path/to/local/my_dataset.json')

    # Load from a local loading script
    >>> from datasets import load_dataset
    >>> ds = load_dataset('path/to/local/loading_script/loading_script.py', split='train')
    ```

    Load an [`~datasets.IterableDataset`]:

    ```py
    >>> from datasets import load_dataset
    >>> ds = load_dataset('rotten_tomatoes', split='train', streaming=True)
    ```

    Load an image dataset with the `ImageFolder` dataset builder:

    ```py
    >>> from datasets import load_dataset
    >>> ds = load_dataset('imagefolder', data_dir='/path/to/images', split='train')
    ```
    """
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
        trust_remote_code=trust_remote_code,
        _require_default_config_name=name is None,
        **config_kwargs,
    )

    # Return iterable dataset in case of streaming
    if streaming:
        return builder_instance.as_streaming_dataset(split=split)

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
    if save_infos:
        builder_instance._save_infos()

    return ds