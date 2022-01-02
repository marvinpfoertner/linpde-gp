from typing import Callable, Protocol, Tuple

import jax
import numpy as np
import probnum as pn

from ..typing import JaxLinearOperator
from .kernels import JaxKernel
from .mean_fns import JaxMean


def apply_jax_linop_to_gp(
    gp: pn.randprocs.GaussianProcess,
    linop: JaxLinearOperator,
    **linop_kwargs,
) -> Tuple[pn.randprocs.GaussianProcess, JaxKernel]:
    mean = linop(gp._meanfun, argnum=0, **linop_kwargs)
    crosscov = linop(gp._covfun.jax, argnum=1, **linop_kwargs)
    cov = linop(crosscov, argnum=0, **linop_kwargs)

    gp_linop = pn.randprocs.GaussianProcess(
        mean=JaxMean(mean, vectorize=True),
        cov=JaxKernel(cov, input_dim=gp.input_dim, vectorize=True),
    )
    crosskernel = JaxKernel(crosscov, input_dim=gp.input_dim, vectorize=True)

    return gp_linop, crosskernel


pn.randprocs.GaussianProcess.apply_jax_linop = apply_jax_linop_to_gp


def condition_gp_on_observations_jax(
    gp: pn.randprocs.GaussianProcess, X: np.ndarray, fX: pn.randvars.Normal
) -> pn.randprocs.GaussianProcess:
    mX = gp._meanfun(X)
    kXX = gp._covfun.jax(X[:, None, :], X[None, :, :]) + fX.cov
    L_kXX = jax.scipy.linalg.cho_factor(kXX)

    @jax.jit
    def cond_mean(x):
        mx = gp._meanfun(x)
        kxX = gp._covfun.jax(x, X)
        return mx + kxX @ jax.scipy.linalg.cho_solve(L_kXX, (fX.mean - mX))

    @jax.jit
    def cond_cov(x0, x1):
        kxx = gp._covfun.jax(x0, x1)
        kxX = gp._covfun.jax(x0, X)
        kXx = gp._covfun.jax(X, x1)
        return kxx - kxX @ jax.scipy.linalg.cho_solve(L_kXX, kXx)

    cond_gp = pn.randprocs.GaussianProcess(
        mean=JaxMean(cond_mean, vectorize=True),
        cov=JaxKernel(cond_cov, input_dim=gp.input_dim, vectorize=True),
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
    gp_pred_mean_at_X = gp_pred._meanfun(X)
    gramXX = gp_pred._covfun.jax(X[:, None, :], X[None, :, :]) + observations.cov
    gramXX_cho = jax.scipy.linalg.cho_factor(gramXX)

    @jax.jit
    def pred_cond_mean(x):
        gp_mean_at_x = gp._meanfun(x)
        crosscov_at_xX = crosscov.jax(x[None], X)
        return gp_mean_at_x + crosscov_at_xX @ jax.scipy.linalg.cho_solve(
            gramXX_cho, (observations.mean - gp_pred_mean_at_X)
        )

    @jax.jit
    def pred_cond_cov(x0, x1):
        cov_at_xx = gp._covfun.jax(x0, x1)
        crosscov_at_xX = crosscov.jax(x0, X)
        crosscov_at_Xx = crosscov.jax(x1, X).T
        return cov_at_xx - crosscov_at_xX @ jax.scipy.linalg.cho_solve(
            gramXX_cho, crosscov_at_Xx
        )

    cond_gp = pn.randprocs.GaussianProcess(
        mean=JaxMean(pred_cond_mean, vectorize=True),
        cov=JaxKernel(pred_cond_cov, input_dim=gp.input_dim, vectorize=True),
    )

    return cond_gp


pn.randprocs.GaussianProcess.condition_on_predictive_gp_observations_jax = (
    condition_gp_on_predictive_gp_observations
)
