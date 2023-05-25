import numpy as np
import probnum as pn

from pytest_cases import parametrize

import linpde_gp
from linpde_gp.randprocs.crosscov.linfunctls import integrals as covfunc_integrals

from ._test_case import CovarianceFunctionLinearFunctionalTestCase

nus = (0.5, 1.5, 2.5, 3.5, 4.5)
lengthscales = (0.8, 1.1, 2.1)
domains = (
    linpde_gp.domains.asdomain([-2.2, -1.8]),
    linpde_gp.domains.asdomain([-0.8, 0.5]),
)


@parametrize(
    nu=nus,
    lengthscale=lengthscales,
    domain=domains,
)
def case_matern_lebesgue_integral(
    nu: float,
    lengthscale: float,
    domain: linpde_gp.domains.Interval,
) -> CovarianceFunctionLinearFunctionalTestCase:
    k = pn.randprocs.covfuncs.Matern(input_shape=(), nu=nu, lengthscales=lengthscale)
    L = linpde_gp.linfunctls.LebesgueIntegral(domain)

    a, b = domain
    half_width = (b - a) / 2

    return CovarianceFunctionLinearFunctionalTestCase(
        k=k,
        L=L,
        Lk_fallback=_Lk_fallback(k, L),
        kL_fallback=_kL_fallback(k, L),
        X_test=np.linspace(a - half_width, b + half_width, 10),
        expected_type=covfunc_integrals.HalfIntegerMatern_Identity_LebesgueIntegral,
    )


def _Lk_fallback(
    k: pn.randprocs.covfuncs.CovarianceFunction,
    L: linpde_gp.linfunctls.LebesgueIntegral,
) -> covfunc_integrals.CovarianceFunction_Identity_LebesgueIntegral:
    return covfunc_integrals.CovarianceFunction_Identity_LebesgueIntegral(
        k,
        L,
        reverse=True,
    )


def _kL_fallback(
    k: pn.randprocs.covfuncs.CovarianceFunction,
    L: linpde_gp.linfunctls.LebesgueIntegral,
) -> covfunc_integrals.CovarianceFunction_Identity_LebesgueIntegral:
    return covfunc_integrals.CovarianceFunction_Identity_LebesgueIntegral(
        k,
        L,
        reverse=True,
    )
