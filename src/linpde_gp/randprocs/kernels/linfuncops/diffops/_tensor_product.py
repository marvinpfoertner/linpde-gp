from collections.abc import Mapping
import functools
import operator

from jax import numpy as jnp
import numpy as np

from linpde_gp.linfuncops import diffops

from ..._jax import JaxKernel, JaxKernelMixin
from ..._tensor_product import (
    TensorProductKernel,
    evaluate_dimensionwise,
    evaluate_dimensionwise_jax,
)


class TensorProductKernel_Identity_DimSumDiffOp(JaxKernel):
    """Cross-covariance function obtained by applying a linear differential operator
    without any mixed partial derivatives to one argument of a
    :class:`TensorProductKernel`."""

    def __init__(
        self,
        k: TensorProductKernel,
        L: diffops.LinearDifferentialOperator,
        *,
        L_summands: Mapping[int, diffops.LinearDifferentialOperator],
        reverse: bool = False,
    ):
        super().__init__(k.input_shape, output_shape=())

        self._k = k
        self._L = L
        self._L_summands = dict(L_summands)
        self._reverse = bool(reverse)

    @property
    def k(self) -> TensorProductKernel:
        return self._k

    @property
    def L(self) -> diffops.LinearDifferentialOperator:
        return self._L

    @property
    def reverse(self) -> bool:
        return self._reverse

    @functools.cached_property
    def _kLs_or_Lks(self) -> Mapping[int, JaxKernelMixin]:
        return {
            dim_idx: L_summand(
                self._k.factors[dim_idx], argnum=(0 if self.reverse else 1)
            )
            for dim_idx, L_summand in self._L_summands.items()
        }

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        ks_x0_x1 = evaluate_dimensionwise(self._k.factors, x0, x1)
        kLs_or_Lks_x0_x1 = {
            dim_idx: kL_or_Lk(
                x0[..., dim_idx], x1[..., dim_idx] if x1 is not None else None
            )
            for dim_idx, kL_or_Lk in self._kLs_or_Lks.items()
        }

        res = 0.0

        for dim_idx, kL_or_Lk_x0_x1 in kLs_or_Lks_x0_x1.items():
            res += functools.reduce(
                operator.mul,
                ks_x0_x1[:dim_idx] + (kL_or_Lk_x0_x1,) + ks_x0_x1[dim_idx + 1 :],
            )

        return res

    def _evaluate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        ks_x0_x1 = evaluate_dimensionwise_jax(self._k.factors, x0, x1)
        kLs_or_Lks_x0_x1 = {
            dim_idx: kL_or_Lk.jax(
                x0[..., dim_idx], x1[..., dim_idx] if x1 is not None else None
            )
            for dim_idx, kL_or_Lk in self._kLs_or_Lks.items()
        }

        res = 0.0

        for dim_idx, kL_or_Lk_x0_x1 in kLs_or_Lks_x0_x1.items():
            res += functools.reduce(
                operator.mul,
                ks_x0_x1[:dim_idx] + (kL_or_Lk_x0_x1,) + ks_x0_x1[dim_idx + 1 :],
            )

        return res


class TensorProductKernel_DimSumDiffop_DimSumDiffop(JaxKernel):
    """Cross-covariance function obtained by applying a linear differential operators
    without any mixed partial derivatives to both arguments of a
    :class:`TensorProductKernel`."""

    def __init__(
        self,
        k: TensorProductKernel,
        *,
        L0: diffops.LinearDifferentialOperator,
        L1: diffops.LinearDifferentialOperator,
        L0_summands: Mapping[int, diffops.LinearDifferentialOperator],
        L1_summands: Mapping[int, diffops.LinearDifferentialOperator],
    ):
        super().__init__(k.input_shape, output_shape=())

        self._k = k

        self._L0 = L0
        self._L1 = L1

        self._L0_summands = dict(L0_summands)
        self._L1_summands = dict(L1_summands)

    @property
    def k(self) -> TensorProductKernel:
        return self._k

    @functools.cached_property
    def _L0ks(self) -> Mapping[int, JaxKernelMixin]:
        return {
            dim_idx: L0_summand(self._k.factors[dim_idx], argnum=0)
            for dim_idx, L0_summand in self._L0_summands.items()
        }

    @functools.cached_property
    def _kL1s(self) -> Mapping[int, JaxKernelMixin]:
        return {
            dim_idx: L1_summand(self._k.factors[dim_idx], argnum=1)
            for dim_idx, L1_summand in self._L1_summands.items()
        }

    @functools.cached_property
    def _L0kL1s(self) -> Mapping[int, JaxKernelMixin]:
        L0kL1s = {}

        for dim_idx, kL1 in self._kL1s.items():
            if dim_idx in self._L0_summands:
                L0kL1s[dim_idx] = self._L0_summands[dim_idx](kL1, argnum=0)

        return L0kL1s

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        ks_x0_x1 = evaluate_dimensionwise(self._k.factors, x0, x1)
        L0ks_x0_x1 = {
            dim_idx: L0k(x0[..., dim_idx], x1[..., dim_idx] if x1 is not None else None)
            for dim_idx, L0k in self._L0ks.items()
        }
        kL1s_x0_x1 = {
            dim_idx: kL1(x0[..., dim_idx], x1[..., dim_idx] if x1 is not None else None)
            for dim_idx, kL1 in self._kL1s.items()
        }
        L0kL1s_x0_x1 = {
            dim_idx: L0kL1(
                x0[..., dim_idx], x1[..., dim_idx] if x1 is not None else None
            )
            for dim_idx, L0kL1 in self._L0kL1s.items()
        }

        res = 0.0

        for i, L0k_x0_x1 in L0ks_x0_x1.items():
            for j, kL1_x0_x1 in kL1s_x0_x1.items():
                if i == j:
                    res += functools.reduce(
                        operator.mul,
                        ks_x0_x1[:i] + (L0kL1s_x0_x1[i],) + ks_x0_x1[i + 1 :],
                    )
                else:
                    m = min(i, j)
                    n = max(i, j)

                    res += functools.reduce(
                        operator.mul,
                        (
                            ks_x0_x1[:m]
                            + ks_x0_x1[m + 1 : n]
                            + ks_x0_x1[n + 1 :]
                            + (L0k_x0_x1, kL1_x0_x1)
                        ),
                    )

        return res

    def _evaluate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        ks_x0_x1 = evaluate_dimensionwise_jax(self._k.factors, x0, x1)
        L0ks_x0_x1 = {
            dim_idx: L0k.jax(
                x0[..., dim_idx], x1[..., dim_idx] if dim_idx is not None else None
            )
            for dim_idx, L0k in self._L0ks.items()
        }
        kL1s_x0_x1 = {
            dim_idx: kL1.jax(
                x0[..., dim_idx], x1[..., dim_idx] if dim_idx is not None else None
            )
            for dim_idx, kL1 in self._kL1s.items()
        }
        L0kL1s_x0_x1 = {
            dim_idx: L0kL1.jax(
                x0[..., dim_idx], x1[..., dim_idx] if dim_idx is not None else None
            )
            for dim_idx, L0kL1 in self._L0kL1s.items()
        }

        res = 0.0

        for i, L0k_x0_x1 in L0ks_x0_x1.items():
            for j, kL1_x0_x1 in kL1s_x0_x1.items():
                if i == j:
                    res += functools.reduce(
                        operator.mul,
                        ks_x0_x1[:i] + (L0kL1s_x0_x1[i],) + ks_x0_x1[i + 1 :],
                    )
                else:
                    m = min(i, j)
                    n = max(i, j)

                    res += functools.reduce(
                        operator.mul,
                        (
                            ks_x0_x1[:m]
                            + ks_x0_x1[m + 1 : n]
                            + ks_x0_x1[n + 1 :]
                            + (L0k_x0_x1, kL1_x0_x1)
                        ),
                    )

        return res


class TensorProductKernel_Identity_DirectionalDerivative(
    TensorProductKernel_Identity_DimSumDiffOp
):
    def __init__(
        self,
        k: TensorProductKernel,
        L: diffops.DirectionalDerivative,
        *,
        reverse: bool = False,
    ):
        super().__init__(
            k,
            L,
            L_summands={
                dim_idx: diffops.DirectionalDerivative(dim_dir)
                for dim_idx, dim_dir in enumerate(L.direction)
                if dim_dir != 0
            },
            reverse=reverse,
        )


class TensorProductKernel_DirectionalDerivative_DirectionalDerivative(
    TensorProductKernel_DimSumDiffop_DimSumDiffop
):
    def __init__(
        self,
        k: TensorProductKernel,
        *,
        L0: diffops.DirectionalDerivative,
        L1: diffops.DirectionalDerivative,
    ):
        super().__init__(
            k,
            L0=L0,
            L1=L1,
            L0_summands={
                dim_idx: diffops.DirectionalDerivative(dim_dir)
                for dim_idx, dim_dir in enumerate(L0.direction)
                if dim_dir != 0
            },
            L1_summands={
                dim_idx: diffops.DirectionalDerivative(dim_dir)
                for dim_idx, dim_dir in enumerate(L1.direction)
                if dim_dir != 0
            },
        )


class TensorProductKernel_Identity_WeightedLaplacian(
    TensorProductKernel_Identity_DimSumDiffOp
):
    def __init__(
        self,
        k: TensorProductKernel,
        L: diffops.WeightedLaplacian,
        *,
        reverse: bool = True,
    ):
        super().__init__(
            k,
            L,
            L_summands={
                dim_idx: diffops.WeightedLaplacian(weight)
                for dim_idx, weight in enumerate(L.weights)
                if weight != 0
            },
            reverse=reverse,
        )


class TensorProductKernel_WeightedLaplacian_WeightedLaplacian(
    TensorProductKernel_DimSumDiffop_DimSumDiffop
):
    def __init__(
        self,
        k: TensorProductKernel,
        *,
        L0: diffops.WeightedLaplacian,
        L1: diffops.WeightedLaplacian,
    ):
        super().__init__(
            k,
            L0=L0,
            L1=L1,
            L0_summands={
                dim_idx: diffops.WeightedLaplacian(weight)
                for dim_idx, weight in enumerate(L0.weights)
                if weight != 0
            },
            L1_summands={
                dim_idx: diffops.WeightedLaplacian(weight)
                for dim_idx, weight in enumerate(L1.weights)
                if weight != 0
            },
        )


class TensorProductKernel_DirectionalDerivative_WeightedLaplacian(
    TensorProductKernel_DimSumDiffop_DimSumDiffop
):
    def __init__(
        self,
        k: TensorProductKernel,
        *,
        dderiv: diffops.DirectionalDerivative,
        laplacian: diffops.WeightedLaplacian,
        reverse: bool = False,
    ):
        dderiv_summands = {
            dim_idx: diffops.DirectionalDerivative(dim_dir)
            for dim_idx, dim_dir in enumerate(dderiv.direction)
            if dim_dir != 0
        }

        laplacian_summands = {
            dim_idx: diffops.WeightedLaplacian(weight)
            for dim_idx, weight in enumerate(laplacian.weights)
            if weight != 0
        }

        if reverse:
            L0 = laplacian
            L1 = dderiv

            L0_summands = laplacian_summands
            L1_summands = dderiv_summands
        else:
            L0 = dderiv
            L1 = laplacian

            L0_summands = dderiv_summands
            L1_summands = laplacian_summands

        super().__init__(
            k,
            L0=L0,
            L1=L1,
            L0_summands=L0_summands,
            L1_summands=L1_summands,
        )
