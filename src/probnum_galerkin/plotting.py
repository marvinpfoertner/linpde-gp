from typing import Optional, Tuple

import matplotlib.axes
import matplotlib.collections
import matplotlib.lines
import numpy as np
import probnum as pn
import scipy.stats


def plot_gaussian_process(
    gp: pn.randprocs.GaussianProcess,
    ax: matplotlib.axes.Axes,
    xs: np.ndarray,
    cred_int: float = 0.95,
    fill_alpha: float = 0.1,
    label: Optional[str] = None,
    **kwargs,
) -> Tuple[matplotlib.lines.Line2D, matplotlib.collections.PolyCollection]:
    mean = gp.mean(xs)
    std = gp.std(xs[:, None])

    std_factor = -scipy.stats.norm.ppf((1.0 - cred_int) / 2.0)

    fill_delta = std_factor * std

    (mean_line2d,) = ax.plot(xs, mean, label=label, **kwargs)
    std_poly = ax.fill_between(
        xs,
        mean - fill_delta,
        mean + fill_delta,
        alpha=fill_alpha,
        label=label,
        **kwargs,
    )

    return mean_line2d, std_poly


pn.randprocs.GaussianProcess.plot = plot_gaussian_process
