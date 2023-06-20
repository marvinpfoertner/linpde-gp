from __future__ import annotations

from collections.abc import Callable
import functools
from typing import TYPE_CHECKING

import jax
from jax import numpy as jnp
import numpy as np
import probnum as pn
from probnum.typing import ArrayLike, ShapeLike

from linpde_gp import functions

from ._lindiffop import LinearDifferentialOperator, PartialDerivativeCoefficients

if TYPE_CHECKING:
    import linpde_gp


class WeightedLaplacian(LinearDifferentialOperator):
    r"""Generalization of the Laplacian operator, which multiplies each partial
    derivative with an individual weight.

    .. math::
        \Delta_w := \sum_{i = 1}^d w_i \frac{\partial^2}{\partial x_i^2}
    """

    def __init__(self, weights: ArrayLike) -> None:
        weights = np.asarray(weights)

        if weights.ndim > 1:
            raise ValueError(
                "The Laplacian operator only supports functions with input ndim of at "
                "most 1."
            )

        def get_one_hot(index: int) -> np.ndarray:
            one_hot = np.zeros(weights.size, dtype=int)
            one_hot[index] = 2
            return tuple(one_hot)

        coefficients = PartialDerivativeCoefficients(
            {
                (): {
                    get_one_hot(domain_index): coefficient
                    for domain_index, coefficient in enumerate(weights.reshape(-1))
                    if coefficient != 0.0
                }
            }
        )

        super().__init__(coefficients=coefficients, input_shapes=(weights.shape, ()))

        self._weights = weights

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)

    def _jax_fallback(  # pylint: disable=arguments-differ
        self, f: Callable, /, *, argnum: int = 0, **kwargs
    ) -> Callable:
        f_hessian = jax.hessian(f, argnums=argnum)

        @jax.jit
        def f_laplacian(*args, **kwargs):
            f_hessian_diag = jnp.diag(jnp.atleast_2d(f_hessian(*args, **kwargs)))

            return jnp.sum(self._weights * f_hessian_diag)

        return f_laplacian

    @functools.singledispatchmethod
    def weak_form(
        self, test_basis: pn.functions.Function, /
    ) -> "linpde_gp.linfunctls.LinearFunctional":
        raise NotImplementedError()


class Laplacian(WeightedLaplacian):
    def __init__(self, domain_shape: ShapeLike) -> None:
        super().__init__(np.ones(domain_shape, dtype=np.double))

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)

    @functools.singledispatchmethod
    def weak_form(
        self, test_basis: pn.functions.Function, /
    ) -> "linpde_gp.linfunctls.LinearFunctional":
        raise NotImplementedError()

    @weak_form.register(functions.bases.UnivariateLinearInterpolationBasis)
    def _weak_form_univariate_interpolation_basis(
        self, test_basis: functions.bases.UnivariateLinearInterpolationBasis
    ):
        from linpde_gp.linfunctls.weak_forms import (  # pylint: disable=import-outside-toplevel
            WeakForm_Laplacian_UnivariateInterpolationBasis,
        )

        return WeakForm_Laplacian_UnivariateInterpolationBasis(test_basis)


class SpatialLaplacian(WeightedLaplacian):
    def __init__(self, domain_shape: ShapeLike) -> None:
        domain_shape = pn.utils.as_shape(domain_shape)

        if len(domain_shape) != 1 or domain_shape[0] < 2:
            raise ValueError()

        self._laplacian = Laplacian(domain_shape=(domain_shape[0] - 1,))

        weights = np.ones(domain_shape, dtype=np.double)
        weights[0] = 0

        super().__init__(weights)

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)

    @functools.singledispatchmethod
    def weak_form(
        self, test_basis: pn.functions.Function, /
    ) -> "linpde_gp.linfunctls.LinearFunctional":
        raise NotImplementedError()
