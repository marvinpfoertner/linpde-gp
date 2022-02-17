import jax
import numpy as np

import probnum as pn

from ...linfuncops import JaxLambdaFunction, JaxLinearOperator
from ..kernels import JaxKernel, JaxLambdaKernel


def apply_jax_linop_to_gp(
    gp: pn.randprocs.GaussianProcess,
    linop: JaxLinearOperator,
    **linop_kwargs,
) -> tuple[pn.randprocs.GaussianProcess, JaxLambdaKernel]:
    mean = linop(gp.mean.jax, argnum=0, **linop_kwargs)
    crosscov = linop(gp.cov.jax, argnum=1, **linop_kwargs)
    cov = linop(crosscov, argnum=0, **linop_kwargs)

    gp_linop = pn.randprocs.GaussianProcess(
        mean=JaxLambdaFunction(mean, input_shape=gp.input_shape, vectorize=True),
        cov=JaxLambdaKernel(cov, input_shape=gp.input_shape, vectorize=True),
    )
    crosskernel = JaxLambdaKernel(crosscov, input_shape=gp.input_shape, vectorize=True)

    return gp_linop, crosskernel


pn.randprocs.GaussianProcess.apply_jax_linop = apply_jax_linop_to_gp


def condition_gp_on_observations_jax(
    gp: pn.randprocs.GaussianProcess, X: np.ndarray, fX: pn.randvars.Normal
) -> pn.randprocs.GaussianProcess:
    mX = gp.mean.jax(X)
    kXX = gp.cov.jax(X[:, None], X[None, :]) + fX.cov
    L_kXX = jax.scipy.linalg.cho_factor(kXX)

    @jax.jit
    def cond_mean(x):
        mx = gp.mean.jax(x)
        kxX = gp.cov.jax(x, X)
        return mx + kxX @ jax.scipy.linalg.cho_solve(L_kXX, (fX.mean - mX))

    @jax.jit
    def cond_cov(x0, x1):
        kxx = gp.cov.jax(x0, x1)
        kxX = gp.cov.jax(x0, X)
        kXx = gp.cov.jax(X, x1)
        return kxx - kxX @ jax.scipy.linalg.cho_solve(L_kXX, kXx)

    cond_gp = pn.randprocs.GaussianProcess(
        mean=JaxLambdaFunction(cond_mean, input_shape=gp.input_shape, vectorize=True),
        cov=JaxLambdaKernel(cond_cov, input_shape=gp.input_shape, vectorize=True),
    )

    return cond_gp


pn.randprocs.GaussianProcess.condition_on_observations_jax = (
    condition_gp_on_observations_jax
)


def condition_gp_on_predictive_gp_observations(
    gp: pn.randprocs.GaussianProcess,
    gp_pred: pn.randprocs.GaussianProcess,
    crosscov: JaxKernel,
    X: np.ndarray,
    observations: pn.randvars.Normal,
):
    gp_pred_mean_at_X = gp_pred.mean.jax(X)
    gramXX = gp_pred.cov.jax(X[:, None], X[None, :]) + observations.cov
    gramXX_cho = jax.scipy.linalg.cho_factor(gramXX)

    @jax.jit
    def pred_cond_mean(x):
        gp_mean_at_x = gp.mean.jax(x)
        crosscov_at_xX = crosscov.jax(x, X)
        return gp_mean_at_x + crosscov_at_xX @ jax.scipy.linalg.cho_solve(
            gramXX_cho, (observations.mean - gp_pred_mean_at_X)
        )

    @jax.jit
    def pred_cond_cov(x0, x1):
        cov_at_xx = gp.cov.jax(x0, x1)
        crosscov_at_xX = crosscov.jax(x0, X)
        crosscov_at_Xx = crosscov.jax(x1, X).T
        return cov_at_xx - crosscov_at_xX @ jax.scipy.linalg.cho_solve(
            gramXX_cho, crosscov_at_Xx
        )

    cond_gp = pn.randprocs.GaussianProcess(
        mean=JaxLambdaFunction(
            pred_cond_mean, input_shape=gp.input_shape, vectorize=True
        ),
        cov=JaxLambdaKernel(pred_cond_cov, input_shape=gp.input_shape, vectorize=True),
    )

    return cond_gp


pn.randprocs.GaussianProcess.condition_on_predictive_gp_observations_jax = (
    condition_gp_on_predictive_gp_observations
)
