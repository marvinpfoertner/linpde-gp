from __future__ import annotations

from collections.abc import Callable
import functools
from typing import TYPE_CHECKING

import jax
from jax import numpy as jnp
import probnum as pn
from probnum.typing import ShapeLike

from linpde_gp import functions

from ._lindiffop import LinearDifferentialOperator

if TYPE_CHECKING:
    import linpde_gp


class Laplacian(LinearDifferentialOperator):
    def __init__(self, domain_shape: ShapeLike) -> None:
        super().__init__(input_shapes=(domain_shape, ()))

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)

    def _jax_fallback(  # pylint: disable=arguments-differ
        self, f: Callable, /, *, argnum: int = 0, **kwargs
    ) -> Callable:
        f_hessian = jax.jit(jax.hessian(f, argnums=argnum))

        @jax.jit
        def f_laplacian(*args, **kwargs):
            return jnp.trace(jnp.atleast_2d(f_hessian(*args, **kwargs)))

        return f_laplacian

    @functools.singledispatchmethod
    def weak_form(
        self, test_basis: pn.functions.Function, /
    ) -> "linpde_gp.linfunctls.LinearFunctional":
        raise NotImplementedError()

    @weak_form.register(functions.bases.UnivariateLinearInterpolationBasis)
    def _(self, test_basis: functions.bases.UnivariateLinearInterpolationBasis):
        from linpde_gp.linfunctls.weak_forms import (
            WeakForm_Laplacian_UnivariateInterpolationBasis,
        )

        return WeakForm_Laplacian_UnivariateInterpolationBasis(test_basis)


class SpatialLaplacian(LinearDifferentialOperator):
    def __init__(self, domain_shape: ShapeLike) -> None:
        domain_shape = pn.utils.as_shape(domain_shape)

        if domain_shape in ((), (1,)) or len(domain_shape) > 1:
            raise ValueError()

        (self._D,) = domain_shape

        self._laplacian = Laplacian(domain_shape=(self._D - 1,))

        super().__init__(input_shapes=((self._D,), ()))

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)

    def _jax_fallback(  # pylint: disable=arguments-differ
        self, f, /, *, argnum: int = 0
    ):
        f_hessian = jax.jit(jax.hessian(f, argnums=argnum))

        @jax.jit
        def f_spatial_laplacian(*args, **kwargs) -> jnp.ndarray:
            return jnp.sum(
                jnp.diag(
                    f_hessian(*args, **kwargs),
                )[1:],
            )

        return f_spatial_laplacian
