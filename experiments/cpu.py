"""
References
----
[1] https://en.wikichip.org/wiki/intel/microarchitectures/coffee_lake#Quad-Core
[2] https://ark.intel.com/content/www/us/en/ark/products/134896/intel-core-i59600k-processor-9m-cache-up-to-4-60-ghz.html
[3] https://ark.intel.com/content/www/us/en/ark/products/134599/intel-core-i912900k-processor-30m-cache-up-to-5-20-ghz.html
[4] https://journals.aps.org/pr/pdf/10.1103/PhysRev.134.A1058
"""

import matplotlib.axes
import matplotlib.axis
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import probnum as pn

import linpde_gp

########################################################################################
# Geometry
########################################################################################

# Domain
width = 16.28  # 16.0  # ≈ 16.28 mm, [1]
height = 9.19  # mm, [1]
depth = 0.37  # mm, datasheed linked under [3] ("Supplemental Information / Datasheet")

domain = linpde_gp.domains.Box([[0.0, width], [0.0, height], [0.0, depth]])

domain_2D = domain[0:2]

# Areas
A_top_bottom = width * height  # mm^2
A_side_NS = width * depth  # mm^2
A_side_EW = height * depth  # mm^2

A_sink = A_top_bottom + 2 * A_side_NS + 2 * A_side_EW  # mm^2, heat sink interface area
A_sink_1D = A_top_bottom + 2 * A_side_EW  # mm^2, in 1D we assume no vertical flow

# Volume
V = width * height * depth  # mm^3

# CPU Cores
N_cores_x = 3
N_cores_y = 2
N_cores = N_cores_x * N_cores_y

core_width = 2.5  # mm, estimated from the schematic in [1]
core_height = 0.45 * height  # mm, estimated from the schematic in [1]

core_offset_x = 1.95  # mm, estimated from the schematic in [1]
core_distance_x = 0.35  # mm, estimated from the schematic in [1]

core_centers_xs = (
    core_offset_x
    + (core_width + core_distance_x) * np.arange(N_cores_x, dtype=np.double)
    + core_width / 2.0
)
core_centers_ys = np.array([core_height / 2.0, height - core_height / 2.0])

core_centers = np.stack(np.meshgrid(core_centers_xs, core_centers_ys), axis=-1)

########################################################################################
# Material Properties
########################################################################################

# Thermal conductivity
kappa = 1.56  # W/cm K, [4], TODO: Improve this estimate
kappa *= 10  # W/mm K

########################################################################################
# Heat Sources
########################################################################################

TDP = 95.0  # W, [2]


def _core_heat_dist_x():
    # Shape of the heat souce
    xs = [domain[0][0]]
    ys = [0.0]

    eps = core_distance_x / 3

    for core_center_x in core_centers_xs:
        xs += [
            core_center_x - core_width / 2 - eps,
            core_center_x - core_width / 2,
            core_center_x + core_width / 2,
            core_center_x + core_width / 2 + eps,
        ]
        ys += [0.0, 1.0, 1.0, 0.0]

    xs += [domain[0][1]]
    ys += [0.0]

    heat_dist_unnorm = linpde_gp.functions.PiecewiseLinear.from_points(xs, ys)

    # Normalize
    normalization_constant = linpde_gp.linfunctls.LebesgueIntegral(domain[0])(
        heat_dist_unnorm
    )

    return (1 / normalization_constant) * heat_dist_unnorm


core_heat_dist_x = _core_heat_dist_x()


def _core_heat_dist_y():
    # Shape of the heat souce
    eps = (core_centers_ys[1] - core_centers_ys[0] - core_height) / 3

    xs = [
        core_centers_ys[0] - core_height / 2,
        core_centers_ys[0] + core_height / 2,
        core_centers_ys[0] + core_height / 2 + eps,
        core_centers_ys[1] - core_height / 2 - eps,
        core_centers_ys[1] - core_height / 2,
        core_centers_ys[1] + core_height / 2,
    ]
    ys = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]

    heat_dist_unnorm = linpde_gp.functions.PiecewiseLinear.from_points(xs, ys)

    # Normalize
    normalization_constant = linpde_gp.linfunctls.LebesgueIntegral(domain[1])(
        heat_dist_unnorm
    )

    return (1 / normalization_constant) * heat_dist_unnorm


core_heat_dist_y = _core_heat_dist_y()

q_dot_V_src_2D = pn.functions.LambdaFunction(
    fn=lambda xy: (
        TDP / depth * core_heat_dist_x(xy[..., 0]) * core_heat_dist_y(xy[..., 1])
    ),
    input_shape=(2,),
    output_shape=(),
)

q_dot_V_src_1D = (TDP / depth / height) * core_heat_dist_x

########################################################################################
# Heat Sinks
########################################################################################

q_dot_V_sink_2D = linpde_gp.functions.Constant(
    input_shape=(2,),
    value=-TDP / A_sink / depth,
)

q_dot_V_sink_1D = linpde_gp.functions.Constant(
    input_shape=(),
    value=-TDP / A_sink_1D / depth,
)

q_dot_V_sink_1D_dbc = linpde_gp.functions.Constant(
    input_shape=(),
    value=-TDP / V,
)

# Boundary function for all boundary parts in a row (order NESW)
q_dot_A_2D = linpde_gp.functions.Constant(
    input_shape=(),
    value=-TDP / A_sink,
)

q_dot_A_1D = np.full(2, -TDP / A_sink_1D)

########################################################################################
# Stationary PDE
########################################################################################

pde_2D = linpde_gp.problems.pde.PoissonEquation(
    domain=domain[:2],
    rhs=q_dot_V_src_2D + q_dot_V_sink_2D,
    alpha=kappa,
)

pde_1D = linpde_gp.problems.pde.PoissonEquation(
    domain=domain[0],
    rhs=q_dot_V_src_1D + q_dot_V_sink_1D,
    alpha=kappa,
)

pde_1D_dbc = linpde_gp.problems.pde.PoissonEquation(
    domain=domain[0],
    rhs=q_dot_V_src_1D + q_dot_V_sink_1D_dbc,
    alpha=kappa,
)

########################################################################################
# Boundary Heat Flux Measurements
########################################################################################

noise_nbc_1D = pn.randvars.Normal(np.zeros(2), np.diag(np.abs(q_dot_A_1D) ** 2))
y_nbc_1D = q_dot_A_1D + noise_nbc_1D.sample(rng=np.random.default_rng(3987))

########################################################################################
# Digital Thermal Sensor (DTS) Measurements
########################################################################################

X_dts_2D = core_centers
y_dts_2D = np.array(
    # fmt: off
    [[59.6, 60.1, 58.5],
     [61.03, 60.36, 59.22]],
    # fmt: on
)
noise_dts_2D = pn.randvars.Normal(
    mean=np.zeros_like(y_dts_2D),
    cov=1.0**2 * np.eye(y_dts_2D.size),
)

X_dts_1D = X_dts_2D[1, :, 0]
y_dts_1D = y_dts_2D[1, :]
noise_dts_1D = 0.5 * noise_dts_2D[1, :]

########################################################################################
# Boundary Value Problems
########################################################################################

bvp_2D = linpde_gp.problems.pde.BoundaryValueProblem(
    pde=pde_2D,
    boundary_conditions=[],
    solution=None,
)

_solution_1D = (
    linpde_gp.problems.pde.Solution_PoissonEquation_IVP_1D_RHSPiecewisePolynomial(
        domain=domain[0],
        rhs=pde_1D.rhs,
        initial_values=[60.0, -q_dot_A_1D[0] / kappa],
        alpha=kappa,
    )
)

bvp_1D = linpde_gp.problems.pde.BoundaryValueProblem(
    pde=pde_1D,
    boundary_conditions=[
        linpde_gp.problems.pde.BoundaryCondition(
            boundary=domain[0].boundary[0],
            operator=-kappa * linpde_gp.linfuncops.diffops.DirectionalDerivative(1.0),
            values=q_dot_A_1D[0],
        ),
        linpde_gp.problems.pde.BoundaryCondition(
            boundary=domain[0].boundary[1],
            operator=-kappa * linpde_gp.linfuncops.diffops.DirectionalDerivative(1.0),
            values=q_dot_A_1D[1],
        ),
    ],
    solution=_solution_1D,
)

_solution_1D_dbc = (
    linpde_gp.problems.pde.Solution_PoissonEquation_IVP_1D_RHSPiecewisePolynomial(
        domain=domain[0],
        rhs=pde_1D.rhs,
        initial_values=[60.0, 0.0],
        alpha=kappa,
    )
)

bvp_1D_dbc = linpde_gp.problems.pde.BoundaryValueProblem(
    pde=pde_1D,
    boundary_conditions=[
        linpde_gp.problems.pde.DirichletBoundaryCondition(
            boundary=point, values=_solution_1D_dbc(point)
        )
        for point in domain[0].boundary
    ],
    solution=_solution_1D_dbc,
)

########################################################################################
# Plotting
########################################################################################


def adjust_xaxis(ax: matplotlib.axes.Axes):
    ax.set_xlim(-0.5, width + 0.5)

    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.0, width]))
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FixedFormatter(
            [
                r"\qty{0.0}{\mm}",
                r"$w_{\mathrm{CPU}}$",
            ]
        )
    )

    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.xaxis.set_minor_formatter(
        matplotlib.ticker.FormatStrFormatter(r"\qty{%.1f}{\mm}")
    )


def adjust_yaxis(ax: matplotlib.axes.Axes):
    ax.set_ylim(-0.5, height + 0.5)

    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.0, height]))
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FixedFormatter([r"\qty{0.0}{\mm}", r"$h_{\mathrm{CPU}}$"])
    )

    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_formatter(
        matplotlib.ticker.FormatStrFormatter(r"\qty{%.1f}{\mm}")
    )


def adjust_tempaxis(axis: matplotlib.axis.Axis):
    axis.set_major_formatter(
        matplotlib.ticker.FormatStrFormatter(r"\qty{%.1f}{\degreeCelsius}")
    )


def adjust_q_dot_V_axis(axis: matplotlib.axis.Axis):
    axis.set_major_formatter(
        matplotlib.ticker.FormatStrFormatter(
            r"\qty[per-mode=fraction]{%.1f}{\watt\per\cubic\mm}"
        )
    )


def plot_schematic(ax: matplotlib.axes.Axes, show_dts: bool = False):
    ax.add_patch(
        plt.Rectangle(
            (0, 0),
            width,
            height,
            edgecolor="black",
            facecolor="None",
            linewidth=0.75,
        )
    )

    for core_center_x in core_centers_xs:
        for core_center_y in core_centers_ys:
            ax.add_patch(
                plt.Rectangle(
                    (
                        core_center_x - core_width / 2.0,
                        core_center_y - core_height / 2.0,
                    ),
                    core_width,
                    core_height,
                    edgecolor="black",
                    facecolor="None",
                    linewidth=0.75,
                )
            )

            ax.text(
                core_center_x,
                core_center_y,
                "CPU\nCore",
                rotation="vertical",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize="x-small",
            )

    adjust_xaxis(ax)
    adjust_yaxis(ax)

    if show_dts:
        ax.scatter(
            X_dts_2D[..., 0],
            X_dts_2D[..., 1],
            marker="+",
            color="C4",
            label=r"$X_\text{DTS}$",
        )
