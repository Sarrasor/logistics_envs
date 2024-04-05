import random

import numpy as np


def set_global_seeds(i):
    try:
        import MPI

        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    myseed = i + 1000 * rank if i is not None else None
    if myseed:
        try:
            import torch

            torch.manual_seed(myseed)
        except ImportError:
            pass

    np.random.seed(myseed)
    random.seed(myseed)


def generate_colors(
    n_colors: int,
) -> list:
    if n_colors < 1:
        raise ValueError(f"n_colors={n_colors} must be greater than 0")

    colors = [
        "#" + "".join([random.choice("0123456789ABCDEF") for _ in range(6)])
        for _ in range(n_colors)
    ]

    return colors
