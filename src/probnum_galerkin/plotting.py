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


def plot_local_curvature(
    ax: matplotlib.axes.Axes,
    xs,
    f_xs,
    ddf_xs,
    df_xs=None,
    *,
    radius: float = 0.1,
    gridsize: int = 21,
    cred_int: float = 0.95,
    markersize: Optional[float] = None,
    color=None,
    alpha: float = 1.0,
    rel_fill_alpha: float = 0.1,
    label=None,
):
    xs = np.atleast_1d(xs)
    f_xs = np.atleast_1d(f_xs)

    ddf_xs = pn.asrandvar(ddf_xs)

    if ddf_xs.ndim == 0:
        ddf_xs = ddf_xs.reshape((1,))

    if df_xs is None:
        df_xs = np.zeros_like(f_xs)
    else:
        df_xs = np.atleast_1d(df_xs)

    # Create plotting grid
    gridsize = 2 * (gridsize // 2) + 1

    centered_grid = np.linspace(-radius, radius, gridsize)

    grids = xs[:, None] + centered_grid

    # Compute means and standard devations on plotting grid
    means = []
    stds = []

    for f_x, df_x, ddf_x in zip(f_xs, df_xs, ddf_xs):
        taylor_poly_grid = f_x
        taylor_poly_grid += df_x * centered_grid
        taylor_poly_grid = (
            taylor_poly_grid + ddf_x[None] / 2.0 @ centered_grid[None, :] ** 2
        )

        means.append(taylor_poly_grid.mean)
        stds.append(taylor_poly_grid.std)

    means = np.vstack(means)
    stds = np.vstack(stds)

    # Plot means with exactlt one marker at the expansion point
    means_line2d = ax.plot(
        grids.T,
        means.T,
        color=color,
        alpha=alpha,
        marker="|",
        markersize=markersize,
        markevery=(gridsize // 2, gridsize),
    )

    if label is not None:
        means_line2d[0].set_label(label)

    colors = [line2d.get_color() for line2d in means_line2d]

    # Plot credible interval
    if cred_int > 0.0:
        std_factor = -scipy.stats.norm.ppf((1.0 - cred_int) / 2.0)
        fill_deltas = std_factor * stds

        fills = []

        for grid, mean, fill_delta, color in zip(grids, means, fill_deltas, colors):
            fill = ax.fill_between(
                grid,
                mean - fill_delta,
                mean + fill_delta,
                color=color,
                alpha=alpha * rel_fill_alpha,
            )

            fills.append(fill)

        return means_line2d, fills

    return means_line2d, []
