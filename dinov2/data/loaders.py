# Author: Tony Xu
#
# This code is adapted from the original DINOv2 repository: https://github.com/facebookresearch/dinov2
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

import logging
from enum import Enum
from typing import Any, Callable, List, Optional, TypeVar
import os
from copy import deepcopy
import random

import csv
import numpy as np
import torch
from torch.utils.data import Sampler
from monai.data import CacheNTransDataset, PersistentDataset
import json

from .samplers import EpochSampler, InfiniteSampler, ShardedInfiniteSampler


logger = logging.getLogger("dinov2")


class SamplerType(Enum):
    DISTRIBUTED = 0
    EPOCH = 1
    INFINITE = 2
    SHARDED_INFINITE = 3
    SHARDED_INFINITE_NEW = 4


def _make_bool_str(b: bool) -> str:
    return "yes" if b else "no"


def _make_sample_transform(image_transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
    def transform(sample):
        image, target = sample
        if image_transform is not None:
            image = image_transform(image)
        if target_transform is not None:
            target = target_transform(target)
        return image, target

    return transform


def make_dataset_3d(
    *,
    dataset_path: str,
    cache_path: str,
    data_min_axis_size: int,
    input_format: str = "nifti",
    csv_paths: Optional[List[str]] = None,
    csv_path_column: Optional[str] = None,
    csv_file_column: str = "files",
    csv_cohort_column: str = "cohort",
    cache_n_trans: int = 5,
    transform: Optional[Callable] = None,
):
    """
    Creates a 3d input dataset with the specified parameters.

    Args:
        dataset_path: A path to a list of sample paths for MONAI datasets, a directory of numpy files,
            or the root directory for CSV-backed numpy files.
        cache_path: A path to a directory to cache the dataset.
        data_min_axis_size: The minimum size of the smallest axis of the data.
        input_format: The input volume format. Supports "nifti" and "numpy".
        csv_paths: Optional CSV path or paths. Each CSV must contain file and cohort columns.
        csv_path_column: Optional CSV column containing the full numpy path. If set, this is used
            instead of joining dataset_path/cohort/files.
        csv_file_column: CSV column containing the numpy filename/path.
        csv_cohort_column: CSV column containing the cohort subdirectory.
        cache_n_trans: Number of leading transforms to cache in MONAI CacheNTransDataset.
        transform: A transform to apply to images.
    Returns:
        The created dataset.
    """
    logger.info(f'creating 3d dataset from datalist: {dataset_path}')
    input_format = input_format.lower()

    def _as_list(paths):
        if paths is None:
            return []
        if isinstance(paths, str):
            return [path.strip() for path in paths.split(",") if path.strip()]
        return list(paths)

    def _as_sample(item):
        if isinstance(item, str):
            sample = {"image": item}
        else:
            sample = item
        return sample

    def _spatial_shape(shape):
        if len(shape) == 4 and shape[0] <= 4:
            return shape[1:]
        if len(shape) == 4 and shape[-1] <= 4:
            return shape[:3]
        return shape[:3]

    def _load_csv_datalist(paths):
        datalist = []
        for csv_path in paths:
            with open(csv_path, newline="") as csv_f:
                reader = csv.DictReader(csv_f)
                required_columns = {csv_path_column} if csv_path_column else {csv_file_column, csv_cohort_column}
                missing_columns = required_columns - set(reader.fieldnames or [])
                if missing_columns:
                    raise ValueError(f"{csv_path} is missing required columns: {sorted(missing_columns)}")

                for row in reader:
                    if csv_path_column:
                        image_path = row[csv_path_column]
                        if not os.path.isabs(image_path):
                            image_path = os.path.join(dataset_path, image_path)
                        sample = {"image": image_path, "source_csv": csv_path}
                    else:
                        image_path = os.path.join(dataset_path, row[csv_cohort_column], row[csv_file_column])
                        sample = {
                            "image": image_path,
                            "cohort": row[csv_cohort_column],
                            "source_csv": csv_path,
                        }
                    datalist.append(sample)
        return datalist

    # load datalist
    csv_paths = _as_list(csv_paths)
    if csv_paths:
        datalist = _load_csv_datalist(csv_paths)
    elif input_format in ("numpy", "npy", "npz") and os.path.isdir(dataset_path):
        datalist = [
            {"image": os.path.join(dataset_path, filename)}
            for filename in sorted(os.listdir(dataset_path))
            if filename.endswith((".npy", ".npz"))
        ]
    else:
        with open(dataset_path, 'r') as json_f:
            datalist = json.load(json_f)

    logger.info(f"loaded 3d datalist with {len(datalist):,d} entries")
    datalist = [_as_sample(x) for x in datalist]

    should_filter_by_shape = data_min_axis_size is not None and data_min_axis_size > 0
    if input_format in ("numpy", "npy", "npz"):
        logger.info("skipping upfront shape filtering for numpy inputs")
    elif should_filter_by_shape and all("shape" in x for x in datalist):
        before_filter = len(datalist)
        datalist = [x for x in datalist if min(_spatial_shape(x['shape'])) > data_min_axis_size]
        logger.info(f"kept {len(datalist):,d}/{before_filter:,d} entries after shape filtering")
    elif should_filter_by_shape:
        logger.info("skipping upfront shape filtering because datalist entries do not include shape metadata")
    logger.info("creating MONAI CacheNTransDataset")
    dataset = CacheNTransDataset(datalist, transform=transform, cache_n_trans=cache_n_trans, cache_dir=cache_path)
    logger.info("finished creating MONAI CacheNTransDataset")

    # Aggregated datasets do not expose (yet) these attributes, so add them.
    if not hasattr(dataset, "transform"):
        setattr(dataset, "transform", transform)

    return dataset


def make_segmentation_dataset_3d(
    dataset_name: str,
    dataset_percent: int,
    base_directory: str,
    train_transforms: Callable,
    val_transforms: Callable,
    cache_path: str,
    batch_size: int,
):
    """
    Creates a 3d segmentation dataset with the specified parameters.

    Args:
        dataset_name: Name of the segmentation dataset (BTCV, BraTS, LA-SEG, TDSC-ABUS).
        dataset_percent: Percentage of the dataset to use for training.
        base_directory: Base directory where dataset json files are stored.
        train_transforms: Training transforms to apply to images.
        val_transforms: Validation transforms to apply to images.
        cache_path: A path to a directory to cache the dataset, used in PersistentDataset.
        batch_size: Batch size for the dataset.
    Returns:
        Created train, val, and test datasets, number of input channels, and number of classes for the dataset.
    """

    if dataset_name == 'BTCV':
        datalist_path = os.path.join(base_directory, 'BTCV_100_datalist.json')
        class_num = 14
        input_channels = 1
    elif dataset_name == 'BraTS':
        datalist_path = os.path.join(base_directory, 'BraTS_100_datalist.json')
        class_num = 3
        input_channels = 4
    elif dataset_name == 'LA-SEG':
        datalist_path = os.path.join(base_directory, 'LA-SEG_100_datalist.json')
        class_num = 2
        input_channels = 1
    elif dataset_name == 'TDSC-ABUS':
        datalist_path = os.path.join(base_directory, 'TDSC-ABUS_100_datalist.json')
        class_num = 2
        input_channels = 1
    else:
        raise ValueError(f'Unsupported dataset "{dataset_name}"')

    with open(datalist_path, 'r') as json_f:
        datalist = json.load(json_f)

    train_data_ind = int(round(len(datalist['training']) * (dataset_percent / 100)))

    train_datalist = datalist['training'][:train_data_ind]
    val_datalist = datalist['validation']
    test_datalist = datalist['test']
    logger.info(f"# of train samples: {len(train_datalist):,d}")
    logger.info(f"# of val samples: {len(val_datalist):,d}")
    logger.info(f"# of test samples: {len(test_datalist):,d}")

    if len(train_datalist) < batch_size:
        logger.info(f"copying train samples to match batch size: {batch_size:,d}")
        copied_datalist = []
        for i in range(batch_size // len(train_datalist)):
            copied_datalist.extend(deepcopy(train_datalist))
        assert len(copied_datalist) == batch_size
        train_datalist = copied_datalist

    train_dataset = PersistentDataset(train_datalist, transform=train_transforms, cache_dir=cache_path)
    val_dataset = PersistentDataset(val_datalist, transform=val_transforms, cache_dir=cache_path)
    test_dataset = PersistentDataset(test_datalist, transform=val_transforms, cache_dir=cache_path)

    return train_dataset, val_dataset, test_dataset, input_channels, class_num


def make_classification_dataset_3d(
    dataset_name: str,
    dataset_percent: int,
    base_directory: str,
    train_transforms: Callable,
    val_transforms: Callable,
    cache_path: str,
    dataset_seed: int,
):
    """
    Creates a 3d classification dataset with the specified parameters.

    Args:
        dataset_name: Name of the classification dataset (ICBM, COVID-CT-MD).
        dataset_percent: Percentage of the dataset to use for training.
        base_directory: Base directory where dataset json files are stored.
        train_transforms: Training transforms to apply to images.
        val_transforms: Validation transforms to apply to images.
        cache_path: A path to a directory to cache the dataset, used in PersistentDataset.
        dataset_seed: Seed for random shuffling of the dataset.
    Returns:
        Created train, val, and test datasets, and number of classes for the dataset.
    """

    if dataset_name == 'ICBM':
        datalist_path = os.path.join(base_directory, 'ICBM_cls_datalist.json')
        class_num = 4
    elif dataset_name == 'COVID-CT-MD':
        datalist_path = os.path.join(base_directory, 'COVID-CT-MD_cls_datalist.json')
        class_num = 3
    else:
        raise ValueError(f'Unsupported dataset "{dataset_name}"')

    with open(datalist_path, 'r') as json_f:
        datalist = json.load(json_f)

    # filter ages for icbm
    if dataset_name == 'ICBM':

        for k in datalist:
            for item in datalist[k]:
                item['image'] = item['image'].replace('.nii.gz', '_mask.nii.gz')

        datalist['training'] = [x for x in datalist['training'] if 20 <= x['label'] <= 60]
        datalist['validation'] = [x for x in datalist['validation'] if 20 <= x['label'] <= 60]
        datalist['test'] = [x for x in datalist['test'] if 20 <= x['label'] <= 60]

    # ensure reproducible shuffling
    random.Random(dataset_seed).shuffle(datalist['training'])
    print(f'Shuffled with seed: {dataset_seed}')

    train_data_ind = int(round(len(datalist['training']) * (dataset_percent / 100)))
    train_datalist = datalist['training'][:train_data_ind]
    val_datalist = datalist['validation']
    test_datalist = datalist['test']

    logger.info(f"# of train samples: {len(train_datalist):,d}")
    logger.info(f"# of val samples: {len(val_datalist):,d}")
    logger.info(f"# of test samples: {len(test_datalist):,d}")

    train_dataset = PersistentDataset(train_datalist, transform=train_transforms, cache_dir=cache_path)
    val_dataset = PersistentDataset(val_datalist, transform=val_transforms, cache_dir=cache_path)
    test_dataset = PersistentDataset(test_datalist, transform=val_transforms, cache_dir=cache_path)

    return train_dataset, val_dataset, test_dataset, class_num


def _make_sampler(
    *,
    dataset,
    type: Optional[SamplerType] = None,
    shuffle: bool = False,
    seed: int = 0,
    size: int = -1,
    advance: int = 0,
) -> Optional[Sampler]:
    sample_count = len(dataset)

    if type == SamplerType.INFINITE:
        logger.info("sampler: infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        return InfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
        )
    elif type in (SamplerType.SHARDED_INFINITE, SamplerType.SHARDED_INFINITE_NEW):
        logger.info("sampler: sharded infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        # TODO: Remove support for old shuffling
        use_new_shuffle_tensor_slice = type == SamplerType.SHARDED_INFINITE_NEW
        return ShardedInfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
            use_new_shuffle_tensor_slice=use_new_shuffle_tensor_slice,
        )
    elif type == SamplerType.EPOCH:
        logger.info("sampler: epoch")
        if advance > 0:
            raise NotImplementedError("sampler advance > 0 is not supported")
        size = size if size > 0 else sample_count
        logger.info(f"# of samples / epoch: {size:,d}")
        return EpochSampler(
            size=size,
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
        )
    elif type == SamplerType.DISTRIBUTED:
        logger.info("sampler: distributed")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        if advance > 0:
            raise ValueError("sampler advance > 0 is invalid")
        return torch.utils.data.DistributedSampler(
            dataset=dataset,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )

    logger.info("sampler: none")
    return None


T = TypeVar("T")


def make_data_loader(
    *,
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    seed: int = 0,
    sampler_type: Optional[SamplerType] = SamplerType.INFINITE,
    sampler_size: int = -1,
    sampler_advance: int = 0,
    drop_last: bool = True,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable[[List[T]], Any]] = None,
):
    """
    Creates a data loader with the specified parameters.

    Args:
        dataset: A dataset (third party, LaViDa or WebDataset).
        batch_size: The size of batches to generate.
        num_workers: The number of workers to use.
        shuffle: Whether to shuffle samples.
        seed: The random seed to use.
        sampler_type: Which sampler to use: EPOCH, INFINITE, SHARDED_INFINITE, SHARDED_INFINITE_NEW, DISTRIBUTED or None.
        sampler_size: The number of images per epoch (when applicable) or -1 for the entire dataset.
        sampler_advance: How many samples to skip (when applicable).
        drop_last: Whether the last non-full batch of data should be dropped.
        persistent_workers: maintain the workers Dataset instances alive after a dataset has been consumed once.
        collate_fn: Function that performs batch collation
    """

    sampler = _make_sampler(
        dataset=dataset,
        type=sampler_type,
        shuffle=shuffle,
        seed=seed,
        size=sampler_size,
        advance=sampler_advance,
    )

    logger.info("using PyTorch data loader")
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )

    try:
        logger.info(f"# of batches: {len(data_loader):,d}")
    except TypeError:  # data loader has no length
        logger.info("infinite data loader")
    return data_loader
