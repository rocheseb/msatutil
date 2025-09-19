from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def make_hist(
    ax: plt.Axes,
    x: Sequence[float],
    exp_fmt: bool = True,
    **kwargs,
):
    """
    Make a historgram of the data in x
    ax: matplotlib axes object
    x: array of data
    kwargs: passed to matplotlib.pyplot.hist
    """
    x = x[np.isfinite(x)]
    
    if "range" in kwargs and kwargs["range"] is not None:
        rng_slice = (x > kwargs["range"][0]) & (x <= kwargs["range"][1])
    else:
        rng_slice = slice(None)

    if "label" not in kwargs:
        kwargs["label"] = ""
    if "facecolor" not in kwargs:
        kwargs["facecolor"] = "None"
    if "histtype" not in kwargs:
        kwargs["histtype"] = "step"

    x_rng = x[rng_slice]
    x_mean = np.nanmean(x_rng)
    x_std = np.nanstd(x_rng, ddof=1)
    x_med = np.nanmedian(x_rng)
    if exp_fmt:
        kwargs["label"] = (
            f"{kwargs['label']}\n"
            + rf"$\mu\pm\sigma$: {x_mean:.3e}$\pm${x_std:.3e}"
            + f"\nmedian: {x_med:.3e} "
        )
    else:
        kwargs["label"] = (
            f"{kwargs['label']}\n"
            + rf"$\mu\pm\sigma$: {x_mean:.2f}$\pm${x_std:.2f}"
            + f"\nmedian: {x_med:.2f} "
        )
    bin_vals, bin_edges, patches = ax.hist(
        x,
        **kwargs,
    )

    ax.axvline(x=x_med, color=patches[0].get_edgecolor(), linestyle="--")
    ax.legend(frameon=False)
    return np.max(bin_vals)
