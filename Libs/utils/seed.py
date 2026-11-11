import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Sets random seeds across libraries for reproducibility.

    Args:
        seed: The random seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


