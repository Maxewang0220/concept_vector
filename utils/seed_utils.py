import random
import torch
import transformers
import numpy as np
import os

def set_seed(seed: int) -> None:
    """
    Sets the seed to make everything deterministic, for reproducibility of experiments

    Parameters:
    seed: the number to set the seed to

    Return: None
    """

    # Random seed
    random.seed(seed)

    # Numpy seed
    np.random.seed(seed)

    # Torch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CUDA determinism
    torch.backends.cudnn.deterministic = True
    # TODO: this may cause time consuming increase
    torch.backends.cudnn.benchmark = False

    # OS seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Transformers seed
    transformers.set_seed(seed)