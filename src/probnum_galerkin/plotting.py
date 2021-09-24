from typing import List, Optional, Tuple

import matplotlib.axes
import matplotlib.collections
import matplotlib.lines
import numpy as np
import probnum as pn
import scipy.stats


def plot_gaussian_process(
    gp: pn.randprocs.GaussianProcess,
    /,
    ax: matplotlib.axes.Axes,
    xs: np.ndarray,
    *,
    cred_int: float = 0.95,
    num_samples: int = 0,
    rng: np.random.Generator = None,
    color: str = None,
    alpha: float = 1.0,
    rel_fill_alpha: float = 0.1,
    rel_sample_alpha: float = 0.1,
    label: Optional[str] = None,
    **kwargs,
) -> Tuple[
    matplotlib.lines.Line2D,
    matplotlib.collections.PolyCollection,
    List[matplotlib.lines.Line2D],
]:
    mean = gp.mean(xs[:, None])
    std = gp.std(xs[:, None])

    std_factor = -scipy.stats.norm.ppf((1.0 - cred_int) / 2.0)

    fill_delta = std_factor * std

    (mean_line2d,) = ax.plot(xs, mean, color=color, alpha=alpha, label=label, **kwargs)
    std_poly = ax.fill_between(
        xs,
        mean - fill_delta,
        mean + fill_delta,
        color=mean_line2d.get_color(),
        alpha=alpha * rel_fill_alpha,
        **kwargs,
    )

    samples_line2d = []

    if num_samples > 0:
        if rng is None:
            raise ValueError()

        samples_line2d = plot_gaussian_process_samples(
            gp,
            ax,
            xs,
            rng,
            num_samples=num_samples,
            color=mean_line2d.get_color(),
            alpha=alpha * rel_sample_alpha,
            **kwargs,
        )

    return mean_line2d, std_poly, samples_line2d


pn.randprocs.GaussianProcess.plot = plot_gaussian_process


def plot_gaussian_process_samples(
    gp: pn.randprocs.GaussianProcess,
    ax: matplotlib.axes.Axes,
    xs: np.ndarray,
    rng: np.random.Generator,
    num_samples: int = 1,
    **kwargs,
) -> List[matplotlib.lines.Line2D]:
    samples = gp.sample(rng, xs[:, None], size=(num_samples,))

    samples_line2d = ax.plot(
        np.broadcast_to(xs[:, None], (xs.shape[0], num_samples)),
        samples.T,
        **kwargs,
    )

    return samples_line2d


pn.randprocs.GaussianProcess.plot_samples = plot_gaussian_process_samples
