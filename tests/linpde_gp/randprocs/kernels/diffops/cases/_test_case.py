import dataclasses
import functools

import linpde_gp


@dataclasses.dataclass(frozen=True)
class KernelLinFuncOpTestCase:
    k: linpde_gp.randprocs.kernels.JaxKernel
    L0: linpde_gp.linfuncops.LinearDifferentialOperator | None
    L1: linpde_gp.linfuncops.LinearDifferentialOperator | None

    @functools.cached_property
    def k_jax(self) -> linpde_gp.randprocs.kernels.JaxKernel:
        return linpde_gp.randprocs.kernels.JaxLambdaKernel(
            k=self.k.jax,
            input_shape=self.k.input_shape,
            vectorize=False,
        )

    @functools.cached_property
    def L0_k_L1_adj(self) -> linpde_gp.randprocs.kernels.JaxKernel:
        k_L1_adj = self.L1(self.k, argnum=1) if self.L1 is not None else self.k

        return self.L0(k_L1_adj, argnum=0) if self.L0 is not None else k_L1_adj

    @functools.cached_property
    def L0_k_L1_adj_jax(self) -> linpde_gp.randprocs.kernels.JaxKernel:
        k_L1_adj = self.L1(self.k_jax, argnum=1) if self.L1 is not None else self.k_jax

        return self.L0(k_L1_adj, argnum=0) if self.L0 is not None else k_L1_adj
