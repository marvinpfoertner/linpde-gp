import pytest
import numpy as np
import probnum as pn
import linpde_gp


@pytest.fixture
def domain() -> linpde_gp.domains.Domain:
    return linpde_gp.domains.Box([[0., 1.], [0., 1.]])

@pytest.fixture
def heat_operator(domain) -> linpde_gp.linfuncops.diffops.LinearDifferentialOperator:
    return linpde_gp.linfuncops.diffops.HeatOperator(domain_shape=(domain.shape), alpha = 1.0)

@pytest.fixture
def closed_form_solution() -> callable:
    def sol(t, x):
        return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)
    return sol

@pytest.fixture
def initial_condition() -> callable:
    def cond(x):
        return np.sin(np.pi * x)
    return cond

@pytest.fixture
def initial_condition_obs(domain: linpde_gp.domains.Domain, initial_condition: callable):
    N_obs = 30
    X_obs = domain.uniform_grid((1, N_obs), inset=(0, 0.001)).reshape(-1, 2)
    Y_obs = initial_condition(X_obs[:, 1])
    return X_obs, Y_obs

@pytest.fixture
def boundary_conditions():
    def left_boundary(t):
        return np.zeros_like(t)
    def right_boundary(t):
        return np.zeros_like(t)
    return left_boundary, right_boundary

@pytest.fixture
def boundary_conditions_obs(domain: linpde_gp.domains.Domain, boundary_conditions: callable):
    left_boundary, right_boundary = boundary_conditions
    N_obs = 25
    X_obs_left = domain.uniform_grid((N_obs, 1), inset=(0.001, 0)).reshape(-1, 2)
    X_obs_right = domain.uniform_grid((N_obs, 1), inset=(0.999, 0)).reshape(-1, 2)
    Y_obs_left = left_boundary(X_obs_left[:, 0])
    Y_obs_right = right_boundary(X_obs_right[:, 0])

    X_obs = np.concatenate((X_obs_left, X_obs_right), axis=0)
    Y_obs = np.concatenate((Y_obs_left, Y_obs_right), axis=0)
    return X_obs, Y_obs

def assert_observations_match(obs, gp: pn.randprocs.GaussianProcess, tol=3e-2):
    X_obs, Y_obs = obs
    vals_gp = gp.mean(X_obs)
    assert np.allclose(vals_gp, Y_obs, rtol=0., atol=tol)

def assert_initial_condition(initial_condition_obs, gp: pn.randprocs.GaussianProcess):
    assert_observations_match(initial_condition_obs, gp)

def assert_boundary_conditions(boundary_conditions_obs, gp: pn.randprocs.GaussianProcess):
    assert_observations_match(boundary_conditions_obs, gp)

def assert_within_uncertainty_region(obs, gp: pn.randprocs.GaussianProcess):
    # TODO: Strictly speaking this need not always be true, right? It's just highly likely...
    X_obs, Y_obs = obs
    vals_gp = gp.mean(X_obs)
    std_gp = gp.std(X_obs)
    assert all(Y_obs <= vals_gp + 2 * std_gp)
    assert all(Y_obs >= vals_gp - 2 * std_gp)

def get_noise(X):
    num_entries = np.prod(X.shape[:-1])
    return pn.randvars.Normal(np.zeros(num_entries), np.diag(1e-5 * np.ones(num_entries)))

def test_compare_solutions(domain, heat_operator, initial_condition_obs, boundary_conditions_obs, closed_form_solution):
    PRIOR_TEMPERATURE = 0.
    PRIOR_STD = 1.
    prior_mean = linpde_gp.functions.Constant(input_shape=(2,), value=PRIOR_TEMPERATURE)
    prior_cov = PRIOR_STD**2 * linpde_gp.randprocs.kernels.ProductMatern(input_shape=domain.shape, p=3, lengthscales = [0.3, 0.3])
    prior = pn.randprocs.GaussianProcess(prior_mean, prior_cov)

    X_obs_initial, Y_obs_initial = initial_condition_obs
    X_obs_boundary, Y_obs_boundary = boundary_conditions_obs
    cond_obs_initial = prior.condition_on_observations(X=X_obs_initial, Y=Y_obs_initial, b=get_noise(X_obs_initial))
    assert_initial_condition(initial_condition_obs, cond_obs_initial)
    cond_obs = cond_obs_initial.condition_on_observations(X=X_obs_boundary, Y=Y_obs_boundary, b=get_noise(X_obs_boundary))
    assert_initial_condition(initial_condition_obs, cond_obs)
    assert_boundary_conditions(boundary_conditions_obs, cond_obs)
    N_pde = 20
    X_pde = domain.uniform_grid((N_pde, N_pde), inset=(0.001, 0.001)) 
    Y_pde = np.zeros_like(X_pde, shape=X_pde.shape[:-1])
    cond_all = cond_obs.condition_on_observations(X=X_pde, Y=Y_pde, L=heat_operator, b=get_noise(X_pde))
    assert_initial_condition(initial_condition_obs, cond_all)
    assert_boundary_conditions(boundary_conditions_obs, cond_all)
    N_test = 10
    X_test = domain.uniform_grid((N_test, N_test), inset=(0.001, 0.001)).reshape(-1, 2)
    Y_test = closed_form_solution(X_test[:, 0], X_test[:, 1])
    assert_within_uncertainty_region((X_test, Y_test), cond_all)