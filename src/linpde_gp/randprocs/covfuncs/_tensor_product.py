from collections.abc import Iterable
import functools
import operator
from typing import Optional, Tuple

from jax import numpy as jnp
import numpy as np
import probnum as pn
from probnum.typing import ArrayLike
from pykeops.numpy import LazyTensor

from ._jax import JaxCovarianceFunctionMixin


class TensorProduct(
    JaxCovarianceFunctionMixin, pn.randprocs.covfuncs.CovarianceFunction
):
    def __init__(self, *factors: pn.randprocs.covfuncs.CovarianceFunction):
        if len(factors) < 0:
            raise ValueError("At least one factor is required.")

        if not all(k.input_shape == () for k in factors):
            raise ValueError("The input shape of all factors must be `()`.")

        if not all(
            k.output_shape_0 == factors[0].output_shape_0
            and k.output_shape_1 == factors[0].output_shape_1
            for k in factors
        ):
            raise ValueError("The output shape of all factors must be equal.")

        self._factors = tuple(factors)

        super().__init__(
            input_shape=(len(self._factors),),
            output_shape_0=self._factors[0].output_shape_0,
            output_shape_1=self._factors[0].output_shape_1,
        )

    @property
    def factors(self) -> tuple[pn.randprocs.covfuncs.CovarianceFunction]:
        return self._factors

    def _evaluate(self, x0: ArrayLike, x1: ArrayLike | None) -> np.ndarray:
        return functools.reduce(
            operator.mul,
            evaluate_dimensionwise(self._factors, x0, x1),
        )

    def _evaluate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        return functools.reduce(
            operator.mul,
            evaluate_dimensionwise_jax(self._factors, x0, x1),
        )

    def _keops_lazy_tensor(
        self, x0: np.ndarray, x1: Optional[np.ndarray]
    ) -> "LazyTensor":
        x0s, x1s = split_outputs_contiguous(x0, x1, len(self._factors))
        return functools.reduce(
            operator.mul, lazy_tensor_dimensionwise(self._factors, x0s, x1s)
        )


def evaluate_dimensionwise(
    ks: Iterable[pn.randprocs.covfuncs.CovarianceFunction],
    x0: np.ndarray,
    x1: np.ndarray | None = None,
) -> tuple[np.ndarray]:
    return tuple(
        k._evaluate(  # pylint: disable=protected-access
            x0[..., i], x1[..., i] if x1 is not None else None
        )
        for i, k in enumerate(ks)
    )


def evaluate_dimensionwise_jax(
    ks: Iterable[JaxCovarianceFunctionMixin],
    x0: jnp.ndarray,
    x1: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray]:
    return tuple(
        k._evaluate_jax(  # pylint: disable=protected-access
            x0[..., i], x1[..., i] if x1 is not None else None
        )
        for i, k in enumerate(ks)
    )


def lazy_tensor_dimensionwise(
    ks: Iterable[pn.randprocs.covfuncs.CovarianceFunction],
    x0s: Tuple[np.ndarray],
    x1s: Tuple[Optional[np.ndarray]],
) -> tuple[LazyTensor]:
    return tuple(
        k._keops_lazy_tensor(x0s[i], x1s[i])  # pylint: disable=protected-access
        for i, k in enumerate(ks)
    )


def split_outputs_contiguous(
    x0: np.ndarray, x1: Optional[np.ndarray], num_dims: int
) -> Tuple[Tuple[np.ndarray], Tuple[Optional[np.ndarray]]]:
    x0s = tuple(np.ascontiguousarray(x0[..., dim_idx]) for dim_idx in range(num_dims))
    x1s = tuple(
        np.ascontiguousarray(x1[..., dim_idx]) if x1 is not None else None
        for dim_idx in range(num_dims)
    )
    return x0s, x1s


class TensorProductGrid(np.ndarray):
    def __new__(cls, *factors: ArrayLike, indexing="ij"):
        factors = tuple(np.asarray(factor) for factor in factors)

        obj = np.stack(
            np.meshgrid(
                *factors,
                copy=True,
                sparse=False,
                indexing=indexing,
            ),
            axis=-1,
        ).view(cls)

        obj.factors = factors

        return obj
