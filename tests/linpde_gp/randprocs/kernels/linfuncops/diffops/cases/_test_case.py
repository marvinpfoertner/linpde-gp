import dataclasses
import functools
from typing import Type

import probnum as pn

import linpde_gp


@dataclasses.dataclass(frozen=True)
class CovarianceFunctionDiffOpTestCase:
    k: linpde_gp.randprocs.covfuncs.JaxCovarianceFunction
    L0: linpde_gp.linfuncops.LinearDifferentialOperator | None
    L1: linpde_gp.linfuncops.LinearDifferentialOperator | None
    expected_type: Type[pn.randprocs.covfuncs.CovarianceFunction] = None

    @functools.cached_property
    def k_jax(self) -> linpde_gp.randprocs.covfuncs.JaxCovarianceFunction:
        return linpde_gp.randprocs.covfuncs.JaxLambdaCovarianceFunction(
            k=self.k.jax,
            input_shape=self.k.input_shape,
            vectorize=False,
        )

    @functools.cached_property
    def L0kL1(self) -> linpde_gp.randprocs.covfuncs.JaxCovarianceFunction:
        k_L1_adj = self.L1(self.k, argnum=1) if self.L1 is not None else self.k

        return self.L0(k_L1_adj, argnum=0) if self.L0 is not None else k_L1_adj

    @functools.cached_property
    def L0kL1_jax(self) -> linpde_gp.randprocs.covfuncs.JaxCovarianceFunction:
        k_L1_adj = self.L1(self.k_jax, argnum=1) if self.L1 is not None else self.k_jax

        return self.L0(k_L1_adj, argnum=0) if self.L0 is not None else k_L1_adj
