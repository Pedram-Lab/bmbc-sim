from typing import Literal

import matplotlib.pyplot as plt


def plot_style(style: Literal["default", "pedramlab"]) -> tuple[float, float]:
    """Set the plot style according to the specified style.

    :param style: The style to apply. "default" for no changes, "pedramlab" for custom theme.
    :returns: A tuple (width, height) for the figure size in inches.
    :raises ValueError: If style is not "default" or "pedramlab"
    """
    match style:
        case "default":
            return 6.4, 4.8
        case "pedramlab":
            plt.rcParams.update({
                "font.size": 9,
                "axes.titlesize": 9,
                "axes.labelsize": 9,
                "legend.fontsize": 9,
                "legend.edgecolor": "black",
                "legend.frameon": False,
                "lines.linewidth": 0.5,
                "font.family": ["Arial", "sans-serif"],
                "axes.spines.top": True,
                "axes.spines.right": True,
                "axes.spines.left": True,
                "axes.spines.bottom": True,
                "axes.linewidth": 0.5,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
            })
            return 5.36, 3.27
        case _:
            raise ValueError(
                f"Unknown style '{style}'. Must be 'default' or 'pedramlab'."
            )
