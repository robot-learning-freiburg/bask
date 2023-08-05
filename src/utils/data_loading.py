
import numpy as np
from loguru import logger
from torch.utils.data import DataLoader, SubsetRandomSampler


def build_data_loaders(replay_memory, collate_func, config,
                       train_workers=1, train_prefetch=1,
                       val_workers=0, val_prefetch=1,
                       drop_last=False, shuffle=True):
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
