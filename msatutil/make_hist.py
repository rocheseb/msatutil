from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Annotated


def make_hist(
    ax: plt.Axes,
    x: Sequence[float],
    label: str = "",
    color: Optional[str] = None,
    rng: Optional[Annotated[Sequence[float], 2]] = None,
    nbins: Optional[int] = None,
    exp_fmt: bool = True,
):
    """
    Make a historgram of the data in x
    ax: matplotlib axes object
    x: array of data
    label: label for the legend
    color: color of the bars
    rng: range of the horizontal axis
    nbins: number of bins for the histogram
    """
    if rng is not None:
        rng_slice = (x > rng[0]) & (x <= rng[1])
    else:
        rng_slice = slice(None)
        rng = [np.min(x), np.max(x)]

    x_rng = x[rng_slice]
    x_mean = np.nanmean(x_rng)
    x_std = np.nanstd(x_rng, ddof=1)
    x_med = np.nanmedian(x_rng)
    if exp_fmt:
        label = label = (
            rf"{label}\n$\mu\pm\sigma$: {x_mean:.3e}$\pm${x_std:.3e}\nmedian: {x_med:.3e} "
        )
    else:
        label = label = (
            rf"{label}\n$\mu\pm\sigma$: {x_mean:.2f}$\pm${x_std:.2f}\nmedian: {x_med:.2f} "
        )
    bin_vals, bin_edges, patches = ax.hist(
        x,
        edgecolor=color,
        facecolor="None",
        label=label,
        range=rng,
        bins=nbins,
        histtype="step",
    )

    ax.axvline(x=x_med, color=color, linestyle="--")
    ax.legend(frameon=False)
    return np.max(bin_vals)
