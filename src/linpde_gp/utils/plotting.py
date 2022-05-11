from typing import List, Optional, Tuple

import matplotlib
import matplotlib.animation
import matplotlib.axes
import matplotlib.collections
import matplotlib.lines
import numpy as np
import probnum as pn
import scipy.stats

from linpde_gp import randprocs


def plot_function(
    f: pn.Function,
    /,
    ax: matplotlib.axes.Axes,
    xs: np.ndarray,
    **kwargs,
) -> matplotlib.lines.Line2D:
    return ax.plot(xs, f(xs), **kwargs)


pn.Function.plot = plot_function


def plot_random_process(
    randproc: pn.randprocs.RandomProcess,
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
    Optional[matplotlib.collections.PolyCollection],
    List[matplotlib.lines.Line2D],
]:
    # Plot mean function
    mean = randproc.mean(xs)

    (mean_line2d,) = ax.plot(xs, mean, color=color, alpha=alpha, label=label, **kwargs)

    # Plot marginal credible interval as shaded region
    try:
        std = randproc.std(xs)
    except NotImplementedError:
        std = None

    if std is not None:
        std_factor = -scipy.stats.norm.ppf((1.0 - cred_int) / 2.0)

        fill_delta = std_factor * std

        std_poly = ax.fill_between(
            xs,
            mean - fill_delta,
            mean + fill_delta,
            color=mean_line2d.get_color(),
            alpha=alpha * rel_fill_alpha,
            **kwargs,
        )
    else:
        std_poly = None

    # Plot samples
    samples_line2d = []

    if num_samples > 0:
        if rng is None:
            raise ValueError()

        samples_line2d = plot_random_process_samples(
            randproc,
            ax,
            xs,
            rng,
            num_samples=num_samples,
            color=mean_line2d.get_color(),
            alpha=alpha * rel_sample_alpha,
            **kwargs,
        )

    return mean_line2d, std_poly, samples_line2d


pn.randprocs.RandomProcess.plot = plot_random_process
randprocs.DeterministicProcess.plot = plot_random_process


def plot_random_process_samples(
    randproc: pn.randprocs.RandomProcess,
    ax: matplotlib.axes.Axes,
    xs: np.ndarray,
    rng: np.random.Generator,
    num_samples: int = 1,
    **kwargs,
) -> List[matplotlib.lines.Line2D]:
    samples = randproc.sample(rng, xs, size=(num_samples,))

    samples_line2d = ax.plot(
        np.broadcast_to(xs[:, None], (xs.shape[0], num_samples)),
        samples.T,
        **kwargs,
    )

    return samples_line2d


pn.randprocs.RandomProcess.plot_samples = plot_random_process_samples


def plot_gaussian_pdf(
    rv: pn.randvars.Normal,
    ax: matplotlib.axes.Axes,
    **kwargs,
) -> matplotlib.lines.Line2D:
    assert rv.ndim == 0 or rv.size == 1

    mean = np.squeeze(rv.mean)
    std = np.squeeze(rv.std)

    plt_grid = np.linspace(mean - 3.0 * std, mean + 3.0 * std, 2 * (60 // 2) + 1)

    (line_2d,) = ax.plot(
        plt_grid,
        rv.pdf(np.reshape(plt_grid, plt_grid.shape + rv.shape)),
        **kwargs,
    )

    return line_2d


pn.randvars.Normal.plot = plot_gaussian_pdf


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

    N = xs.shape[0]

    assert xs.shape == (N,)
    assert f_xs.shape == (N,)
    assert df_xs.shape == (N,)
    assert ddf_xs.shape == (N,)

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
        )  # workaround for += ddf_x / 2.0 * centered_grid ** 2

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


@matplotlib.animation.writers.register("pdf")
class PDFWriter(matplotlib.animation.AbstractMovieWriter):
    @classmethod
    def isAvailable(cls):
        return True

    def setup(self, fig: "Figure", outfile: str, dpi: Optional[float] = None) -> None:
        super().setup(fig, outfile, dpi=dpi)

        self._frame_idx = 0

    def grab_frame(self, **savefig_kwargs):
        self.fig.savefig(
            str(self.outfile).format(self._frame_idx),
            dpi=self.dpi,
            **savefig_kwargs,
        )

        self._frame_idx += 1

    def finish(self) -> None:
        super().finish()
