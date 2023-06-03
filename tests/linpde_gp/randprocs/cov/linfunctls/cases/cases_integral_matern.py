import probnum as pn

from pytest_cases import parametrize

import linpde_gp
from linpde_gp.domains import Interval
from linpde_gp.randprocs.crosscov.linfunctls import integrals as covfunc_integrals

from ._test_case import CovarianceFunctionLinearFunctionalsTestCase

nus = (0.5, 1.5, 2.5, 3.5, 4.5)
lengthscales = (0.8, 1.1, 2.1)
domain_pairs = (
    (Interval(-2.2, -1.8), Interval(-0.8, 0.5)),
    (Interval(-1.3, 0.0), Interval(-0.2, 0.1)),
    (Interval(0.25, 0.75), Interval(0.3, 0.6)),
)


@parametrize(
    nu=nus,
    lengthscale=lengthscales,
    domain_pair=domain_pairs,
)
def case_matern_lebesgue_integral(
    nu: float,
    lengthscale: float,
    domain_pair: tuple[Interval, Interval],
) -> CovarianceFunctionLinearFunctionalsTestCase:
    covfunc = pn.randprocs.covfuncs.Matern(
        input_shape=(), nu=nu, lengthscales=lengthscale
    )

    domain0, domain1 = domain_pair
    integral0 = linpde_gp.linfunctls.LebesgueIntegral(domain0)
    integral1 = linpde_gp.linfunctls.LebesgueIntegral(domain1)

    return CovarianceFunctionLinearFunctionalsTestCase(
        covfunc=covfunc,
        L0=integral0,
        L1=integral1,
        L0kL1_fallback=integral0(
            covfunc_integrals.CovarianceFunction_Identity_LebesgueIntegral(
                covfunc,
                integral=integral1,
                reverse=False,
            )
        ),
    )
