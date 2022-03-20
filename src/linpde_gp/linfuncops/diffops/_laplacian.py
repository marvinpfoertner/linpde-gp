from __future__ import annotations

from collections.abc import Callable
import functools
from multiprocessing.sharedctypes import Value
from typing import TYPE_CHECKING

import jax
from jax import numpy as jnp
import numpy as np
import probnum as pn
from probnum.typing import ShapeLike

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
