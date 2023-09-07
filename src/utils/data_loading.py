from abc import ABC
from typing import Callable

import numpy as np
import torch
import torchvision
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler


class InfiniteDataIterator(ABC):
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.data_iterator = iter(data_loader)

    def __iter__(self):
        raise NotImplementedError("To be implemented by subclass.")


class AutoResettingDataIterator(InfiniteDataIterator):
    """
    Infinite iterator over a DataLoader. Creates a new iterator when the old
    one is exhausted during a call to __iter__.
    """

    def __next__(self):
        try:
            batch = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.data_loader)
            batch = next(self.data_iterator)

        return batch


class ReplenishingDataIterator(InfiniteDataIterator):
    """
    Infinite iterator over a DataLoader. For full set evaluation.
    In contrast to InfiniteDataIterator, this iterator raises a StopIteration
    when the DataLoader is exhausted. This is used to signal the end of the
    data set. Still resets the internal iterator when exhausted, so can
    be reused.
    """

    def __next__(self):
        try:
            batch = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.data_loader)
            raise StopIteration

        return batch



def build_data_loaders(replay_memory: Dataset, collate_func: Callable,
                       config: dict, train_workers: int = 1,
                       train_prefetch: int = 1, val_workers: int = 0,
                       val_prefetch: int = 1, drop_last: bool = False,
                       shuffle: bool = True
                       ) -> tuple[DataLoader, DataLoader | None]:

    train_split = config["train_split"]

    if train_split < 1.0:
        logger.info("Splitting dataset.")

        dataset_size = len(replay_memory)
        dataset_indices = list(range(dataset_size))

        np.random.shuffle(dataset_indices)

        split_index = int(train_split * dataset_size)

        train_idx, val_idx = \
            dataset_indices[:split_index], dataset_indices[split_index:]

        if not shuffle:
            logger.warning("Passed shuffle=False, but specified train_split"
                           " for which a SubsetRandomSampler is used.")

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(
            replay_memory, batch_size=config["batch_size"], shuffle=False,
            num_workers=train_workers,
            collate_fn=collate_func,
            pin_memory=False,
            # NOTE: 2 is the default value for prefetch. When not using workers
            # the default value needs to be passed as no prefetch is done.
            prefetch_factor=train_prefetch if train_workers > 0 else 2,
            sampler=train_sampler,
            drop_last=drop_last
        )

        val_loader = DataLoader(
            replay_memory, batch_size=config["eval_batchsize"], shuffle=False,
            num_workers=val_workers,
            collate_fn=collate_func,
            pin_memory=False,
            prefetch_factor=val_prefetch if val_workers > 0 else 2,
            sampler=val_sampler,
            drop_last=drop_last
        )

        logger.info("Training data contains {} trajectories, eval data {}",
                    len(train_idx), len(val_idx))
    else:
        logger.info("No datasplit specified.")
        train_loader = DataLoader(
            replay_memory, batch_size=config["batch_size"], shuffle=shuffle,
            num_workers=2,
            collate_fn=collate_func,
            pin_memory=False,
            prefetch_factor=1)
        val_loader = None

    return train_loader, val_loader


def build_infinte_data_iterators(
        replay_memory: Dataset, collate_func: Callable, config: dict,
        train_workers: int = 1, train_prefetch: int = 1,
        val_workers: int = 0, val_prefetch: int = 1,
        drop_last: bool = False, shuffle: bool = True,
        full_set_training: bool = False, full_set_eval: bool = False
        ) -> tuple[InfiniteDataIterator, InfiniteDataIterator]:

    train_loader, val_loader = build_data_loaders(
        replay_memory, collate_func, config, train_workers, train_prefetch,
        val_workers, val_prefetch, drop_last, shuffle)

    if full_set_training:
        train_iterator = ReplenishingDataIterator(train_loader)
    else:
        train_iterator = AutoResettingDataIterator(train_loader)

    if val_loader is None:
        val_iterator = None
    elif full_set_eval:
        val_iterator = ReplenishingDataIterator(val_loader)
    else:
        val_iterator = AutoResettingDataIterator(val_loader) if val_loader \
            else None

    return train_iterator, val_iterator


img_to_tensor = torchvision.transforms.ToTensor()


def load_image(path, crop=None):
    # crop: left, right, top, bottom
    image = Image.open(path)
    tens = img_to_tensor(image)
    if crop is not None:
        l, r, t, b = crop
        tens = tens[:, t:b][:, :, l:r].contiguous()
    return tens


def load_tensor(path, crop=None):
    # crop: left, right, top, bottom
    tens = torch.load(path)
    if crop is not None:
        l, r, t, b = crop
        tens = tens[t:b][:, l:r].contiguous()
    return tens


def save_image(tens, path):
    return torchvision.utils.save_image(tens, path)


def save_tensor(tens, path):
    tens = tens.to('cpu') if tens is not None else tens
    return torch.save(tens, path)
