from typing import Any, List, Optional, Tuple

import matplotlib
import matplotlib.animation
import matplotlib.axes
import matplotlib.collections
import matplotlib.lines
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import probnum as pn
import scipy.stats

import linpde_gp
from linpde_gp import randprocs
from linpde_gp.typing import ArrayLike


def plot_function(
    f: pn.functions.Function,
    /,
    ax: matplotlib.axes.Axes,
    xs: np.ndarray,
    **kwargs,
) -> matplotlib.lines.Line2D:
    return ax.plot(xs, f(xs), **kwargs)


pn.functions.Function.plot = plot_function


def plot_random_process(randproc: pn.randprocs.RandomProcess, *args, **kwargs):
    if randproc.input_shape == ():
        return _plot_1d_random_process(randproc, *args, **kwargs)
    elif randproc.input_shape == (2,):
        return _plot_2d_random_process(randproc, *args, **kwargs)
    else:
        raise TypeError()


pn.randprocs.RandomProcess.plot = plot_random_process
randprocs.DeterministicProcess.plot = plot_random_process


def _plot_1d_random_process(
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
    mean_line2d_kwargs: dict[str, Any] = {},
    samples_kwargs: dict[str, Any] = {},
    vertical: bool = False,
    **kwargs,
) -> Tuple[
    matplotlib.lines.Line2D,
    Optional[matplotlib.collections.PolyCollection],
    List[matplotlib.lines.Line2D],
]:
    # Plot mean function
    mean = randproc.mean(xs)

    (mean_line2d,) = ax.plot(
        xs if not vertical else mean,
        mean if not vertical else xs,
        color=color,
        alpha=alpha,
        label=label,
        **(kwargs | mean_line2d_kwargs),
    )

    # Plot marginal credible interval as shaded region
    try:
        std = randproc.std(xs)
    except NotImplementedError:
        std = None

    if std is not None:
        std_factor = -scipy.stats.norm.ppf((1.0 - cred_int) / 2.0)

        fill_delta = std_factor * std

        std_poly = (ax.fill_between if not vertical else ax.fill_betweenx)(
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
            vertical=vertical,
            **(kwargs | samples_kwargs),
        )

    return mean_line2d, std_poly, samples_line2d


def plot_random_process_samples(
    randproc: pn.randprocs.RandomProcess,
    ax: matplotlib.axes.Axes,
    xs: np.ndarray,
    rng: np.random.Generator,
    num_samples: int = 1,
    vertical: bool = False,
    **kwargs,
) -> List[matplotlib.lines.Line2D]:
    samples = randproc.sample(rng, xs, size=(num_samples,))

    xs_plt = np.broadcast_to(xs[:, None], (xs.shape[0], num_samples))

    samples_line2d = ax.plot(
        xs_plt if not vertical else samples.T,
        samples.T if not vertical else xs_plt,
        **kwargs,
    )

    return samples_line2d


pn.randprocs.RandomProcess.plot_samples = plot_random_process_samples


def _plot_2d_random_process(
    f: pn.randprocs.RandomProcess,
    ax: mplot3d.Axes3D,
    xy: np.ndarray,
    /,
    *,
    mean_zorder: int = 2,
    slice_xs: ArrayLike | None = None,
    slice_ys: ArrayLike | None = None,
    slice_axis: str = "x",
    slice_lower_zorder: int = 1,
    slice_upper_zorder: int = 3,
    slice_color: str | None = None,
    slice_alpha: float = 0.8,
    slice_padding: float = 0.0,
    slice_cred_int: float = 0.95,
    slice_cred_int_color: str = None,
    slice_cred_int_alpha: str = 0.8,
    slice_num_samples: int = 0,
    slice_samples_rng: np.random.Generator | None = None,
    slice_samples_color: str | None = None,
    slice_samples_alpha: float = 0.9,
    slice_samples_linewidth: float | None = None,
    **kwargs,
):
    """
    Notes
    -----
    The axes of the subplot must be set up with

    .. code::
        fig, ax = plt.subplots(
            subplot_kw={
                "projection": "3d",
                "computed_zorder": False,
            }
        )

    in order for this to work.
    """
    # Plot mean function
    mean_xy = f.mean(xy)

    ax.plot_surface(
        xy[..., 0],
        xy[..., 1],
        mean_xy,
        zorder=mean_zorder,
        **kwargs,
    )

    z_min = np.min(mean_xy)
    z_max = np.max(mean_xy)

    # Plot slice of marginal credible interval
    if slice_xs is not None and slice_ys is not None:
        slice_xys = np.stack(
            np.broadcast_arrays(slice_xs, slice_ys),
            axis=-1,
        )

        slice_mean = f.mean(slice_xys)
        slice_std = f.std(slice_xys)

        slice_std_factor = -scipy.stats.norm.ppf((1.0 - slice_cred_int) / 2.0)

        slice_cred_int_color = (
            slice_cred_int_color
            if slice_cred_int_color is not None
            else ax._get_lines.get_next_color()
        )

        if slice_std_factor > 0.0:
            slice_cred_int_lower = slice_mean - slice_std_factor * slice_std
            slice_cred_int_upper = slice_mean + slice_std_factor * slice_std

            lower_cred_int_poly = linpde_gp.utils.plotting.fill_between_3d(
                ax,
                slice_xs,
                slice_ys,
                slice_cred_int_lower,
                slice_mean,
                axis=slice_axis,
                zorder=slice_lower_zorder,
                color=slice_cred_int_color,
                alpha=slice_cred_int_alpha,
            )

            linpde_gp.utils.plotting.fill_between_3d(
                ax,
                slice_xs,
                slice_ys,
                slice_mean,
                slice_cred_int_upper,
                axis=slice_axis,
                zorder=slice_upper_zorder,
                color=lower_cred_int_poly._facecolor3d,
                alpha=slice_cred_int_alpha,
            )

            z_min = np.minimum(z_min, np.min(slice_cred_int_lower))
            z_max = np.maximum(z_max, np.max(slice_cred_int_upper))
        else:
            slice_cred_int_lower = slice_mean
            slice_cred_int_upper = slice_mean

        if slice_num_samples > 0:
            samples = f.sample(slice_samples_rng, slice_xys, size=slice_num_samples)
            samples = np.atleast_2d(samples)

            slice_samples_color = (
                slice_samples_color
                if slice_samples_color is not None
                else slice_cred_int_color
            )

            slice_samples_linewidth = (
                slice_samples_linewidth
                if slice_samples_linewidth is not None
                else plt.rcParams["lines.linewidth"]
            )

            for idx in range(slice_num_samples):
                _plot_line_zbuffered_wrt_surface(
                    ax,
                    xs=slice_xs if slice_axis == "x" else slice_ys,
                    ys_line=samples[idx],
                    ys_surf=slice_mean,
                    z=slice_ys if slice_axis == "x" else slice_xs,
                    zdir="y" if slice_axis == "x" else "x",
                    zorder_above_surf=slice_upper_zorder,
                    zorder_below_surf=slice_lower_zorder,
                    linewidths=slice_samples_linewidth,
                    color=slice_samples_color,
                    alpha=slice_samples_alpha,
                )

            z_min = np.minimum(z_min, np.min(samples))
            z_max = np.maximum(z_max, np.max(samples))

        if slice_color is not None:
            z_min = z_min - slice_padding
            z_max = z_max + slice_padding

            linpde_gp.utils.plotting.fill_between_3d(
                ax,
                slice_xs,
                slice_ys,
                z_min,
                slice_cred_int_lower,
                zorder=slice_lower_zorder,
                color=slice_color,
                alpha=slice_alpha,
            )

            linpde_gp.utils.plotting.fill_between_3d(
                ax,
                slice_xs,
                slice_ys,
                slice_cred_int_upper,
                z_max,
                zorder=slice_upper_zorder,
                color=slice_color,
                alpha=slice_alpha,
            )

        ax.set_zlim(z_min, z_max)


def _plot_line_zbuffered_wrt_surface(
    ax: mplot3d.Axes3D,
    xs: np.ndarray,
    ys_line: np.ndarray,
    ys_surf: np.ndarray,
    z: float | np.floating,
    zdir: str,
    zorder_above_surf: int,
    zorder_below_surf: int,
    **kwargs,
):
    def line_chunks(xs, ys_line, ys_surf, above, **kwargs):
        chunks = [[]]

        for x, y_line, y_surf in zip(xs, ys_line, ys_surf):
            if (above and y_line >= y_surf) or (not above and y_line < y_surf):
                chunks[-1].append((x, y_line))
            else:
                chunks.append([])

        chunks = [line for line in chunks if len(line) > 0]

        return matplotlib.collections.LineCollection(chunks, **kwargs)

    # Line chunks below the surface
    ax.add_collection3d(
        line_chunks(
            xs,
            ys_line,
            ys_surf,
            above=False,
            zorder=zorder_below_surf,
            **kwargs,
        ),
        zs=z,
        zdir=zdir,
    )

    # Line chunks above the surface
    ax.add_collection3d(
        line_chunks(
            xs,
            ys_line,
            ys_surf,
            above=True,
            zorder=zorder_above_surf,
            **kwargs,
        ),
        zs=z,
        zdir=zdir,
    )


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


def fill_between_3d(
    ax: mplot3d.Axes3D,
    xs: ArrayLike,
    ys: ArrayLike,
    z1: ArrayLike,
    z2: ArrayLike,
    /,
    *,
    axis: str = "x",
    color: str | None = None,
    **kwargs,
) -> matplotlib.collections.PolyCollection:
    if axis not in ("x", "y"):
        raise ValueError()

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    xs_2d = xs if axis == "x" else ys

    ys1_2d = np.asarray(z1)
    ys2_2d = np.asarray(z2)

    fill_poly = matplotlib.collections.PolyCollection(
        (
            np.concatenate(
                (
                    np.stack(np.broadcast_arrays(xs_2d, ys2_2d), axis=-1),
                    np.flip(
                        np.stack(np.broadcast_arrays(xs_2d, ys1_2d), axis=-1),
                        axis=-2,
                    ),
                ),
                axis=-2,
            ),
        ),
        facecolors=(color if color is not None else ax._get_lines.get_next_color(),),
        **kwargs,
    )

    ax.add_collection3d(
        fill_poly,
        zs=(ys if axis == "x" else xs,),
        zdir="y" if axis == "x" else "x",
    )

    return fill_poly


def plot_local_taylor_processes(
    ax: matplotlib.axes.Axes,
    xs: ArrayLike,
    coeffs_xs: list,
    *,
    radius: float = 0.1,
    gridsize: int = 21,
    markersize: Optional[float] = None,
    label: str = None,
    color: str = None,
    **kwargs,
):
    xs = np.atleast_1d(xs)

    eval_grid = np.linspace(-radius, radius, gridsize)

    offsets = []

    for idx, (x0, coeffs_x) in enumerate(zip(xs, coeffs_xs)):
        coeffs_x = pn.asrandvar(coeffs_x)

        offsets.append(coeffs_x.mean[0])

        taylor_process_x = linpde_gp.randprocs.ParametricGaussianProcess(
            weights=coeffs_x,
            feature_fn=pn.functions.LambdaFunction(
                lambda x: (x - x0)[:, None] ** np.arange(coeffs_x.size),
                input_shape=(),
                output_shape=(coeffs_x.size,),
            ),
        )

        mean_line2d, _, _ = taylor_process_x.plot(
            ax,
            x0 + eval_grid,
            color=color,
            label=label if idx == 0 else None,
            mean_line2d_kwargs={
                "marker": "|",
                "markersize": markersize,
                "markevery": (gridsize // 2, gridsize),
            },
            **kwargs,
        )

        if color is None:
            color = mean_line2d.get_color()


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
