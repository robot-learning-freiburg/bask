import argparse
import random

import numpy as np
import torch
from loguru import logger


def configure_seeds(args: argparse.Namespace) -> int:
    """
    Extract seed from args or sample if not provided.
    """
    seed = int(args.seed) if args.seed else random.randint(0, 2000)
    logger.info("Seed: {}", seed)
    set_seeds(seed)
    return seed

def set_seeds(seed: int = 0) -> None:
    """Sets all seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
