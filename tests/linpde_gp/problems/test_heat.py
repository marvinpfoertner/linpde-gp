import numpy as np
import probnum as pn

import pytest

import linpde_gp


@pytest.fixture
def ibvp() -> linpde_gp.problems.pde.BoundaryValueProblem:
    spatial_domain = linpde_gp.domains.asdomain([-1.0, 1.0])

    return linpde_gp.problems.pde.HeatEquationDirichletProblem(
        t0=0.0,
        T=5.0,
        spatial_domain=spatial_domain,
        alpha=0.1,
        initial_values=linpde_gp.functions.TruncatedSineSeries(
            spatial_domain,
            coefficients=[1.0, 2.0],
        ),
    )


def assert_observations_match(obs, gp: pn.randprocs.GaussianProcess, tol=3e-2):
    X_obs, Y_obs = obs
    vals_gp = gp.mean(X_obs)
    assert np.allclose(vals_gp, Y_obs, rtol=0.0, atol=tol)


def assert_initial_condition(initial_condition_obs, gp: pn.randprocs.GaussianProcess):
    assert_observations_match(initial_condition_obs, gp)


def assert_boundary_conditions(
    boundary_conditions_obs, gp: pn.randprocs.GaussianProcess
):
    assert_observations_match(boundary_conditions_obs, gp)


def assert_within_uncertainty_region(obs, gp: pn.randprocs.GaussianProcess):
    X_obs, Y_obs = obs
    vals_gp = gp.mean(X_obs)
    std_gp = np.nan_to_num(gp.std(X_obs))
    assert np.min(vals_gp + 2 * std_gp - Y_obs) > -3e-2
    assert np.min(Y_obs - (vals_gp - 2 * std_gp)) > -3e-2


def get_noise(X):
    num_entries = np.prod(X.shape[:-1])
    return pn.randvars.Normal(
        np.zeros(num_entries), np.diag(1e-5 * np.ones(num_entries))
    )


def test_compare_solutions(ibvp):
    lengthscale_t = 2.5
    lengthscale_x = 2.0
    output_scale = 1.0

    u_prior = pn.randprocs.GaussianProcess(
        mean=linpde_gp.functions.Zero(input_shape=(2,)),
        cov=output_scale**2
        * linpde_gp.randprocs.covfuncs.TensorProduct(
            linpde_gp.randprocs.covfuncs.Matern((), nu=1.5, lengthscales=lengthscale_t),
            linpde_gp.randprocs.covfuncs.Matern((), nu=2.5, lengthscales=lengthscale_x),
        ),
    )

    N_ic = 5
    N_bc = 50

    X_ic = ibvp.initial_domain.uniform_grid(N_ic, inset=1e-6)
    Y_ic = ibvp.initial_condition.values(X_ic[..., 1])

    u_ic = u_prior.condition_on_observations(Y_ic, X_ic)  # pylint: disable=no-member
    assert_initial_condition((X_ic, Y_ic), u_ic)

    u_ic_bc = u_ic
    for bc in ibvp.boundary_conditions:
        X_bc = bc.boundary.uniform_grid(N_bc)
        Y_bc = bc.values(X_bc)

        u_ic_bc = u_ic_bc.condition_on_observations(Y_bc, X=X_bc, b=get_noise(X_bc))
        assert_boundary_conditions((X_bc, Y_bc), u_ic_bc)

    N_pde = (100, 20)
    X_pde = ibvp.domain.uniform_grid(N_pde)
    Y_pde = ibvp.pde.rhs(X_pde)

    u_ic_bc_pde = u_ic_bc.condition_on_observations(
        Y_pde,
        X=X_pde,
        L=ibvp.pde.diffop,
    )

    X_test = ibvp.domain.uniform_grid((50, 50))
    Y_test = ibvp.solution(X_test)
    assert_within_uncertainty_region((X_test, Y_test), u_ic_bc_pde)
