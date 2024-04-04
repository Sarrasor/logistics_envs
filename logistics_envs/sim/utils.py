import colorsys
import random

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


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
    color_type="bright",
    first_color_black=False,
    last_color_black=False,
    verbose=False,
) -> list:
    """
    Creates a list of n_colors colors
    :param n_colors: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """

    if color_type not in ("bright", "soft"):
        raise RuntimeError(
            f'Unknown color type: {color_type}. Please choose "bright" or "soft" for type'
        )

    if verbose:
        print("Number of labels: " + str(n_colors))

    if color_type == "bright":
        randHSVcolors = [
            (
                np.random.uniform(low=0.0, high=1),
                np.random.uniform(low=0.6, high=1),
                np.random.uniform(low=0.7, high=1),
            )
            for _ in range(n_colors)
        ]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list("new_map", randRGBcolors, N=n_colors)

    if color_type == "soft":
        low = 0.6
        high = 0.95
        randRGBcolors = [
            (
                np.random.uniform(low=low, high=high),
                np.random.uniform(low=low, high=high),
                np.random.uniform(low=low, high=high),
            )
            for _ in range(n_colors)
        ]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list("new_map", randRGBcolors, N=n_colors)

    if verbose:
        from matplotlib import colorbar, colors

        _, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, n_colors, n_colors + 1)
        norm = colors.BoundaryNorm(bounds, n_colors)

        colorbar.ColorbarBase(
            ax,
            cmap=random_colormap,
            norm=norm,
            spacing="proportional",
            ticks=None,
            boundaries=bounds,
            format="%1i",
            orientation="horizontal",
        )
        plt.show()

    color_list = []
    for color in random_colormap(np.linspace(0, 1, n_colors)):
        color_list.append(matplotlib.colors.to_hex(color))

    return color_list
