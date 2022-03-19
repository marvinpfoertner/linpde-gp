from __future__ import annotations

from collections.abc import Callable
import functools
from typing import TYPE_CHECKING

import jax
from jax import numpy as jnp
import probnum as pn
from probnum.typing import ShapeLike

from .._jax import JaxLinearOperator

if TYPE_CHECKING:
    import linpde_gp


class ScaledLaplaceOperator(JaxLinearOperator):
    def __init__(self, domain_shape: ShapeLike, alpha: float = 1.0) -> None:
        self._alpha = alpha

        super().__init__(
            input_shapes=(domain_shape, ()),
            output_shapes=(domain_shape, ()),
        )

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)

    def _jax_fallback(self, f: Callable, /, *, argnum: int = 0, **kwargs) -> Callable:
        Hf = jax.jit(jax.hessian(f, argnum))

        @jax.jit
        def _scaled_hessian_trace(*args, **kwargs):
            return self._alpha * jnp.trace(jnp.atleast_2d(Hf(*args, **kwargs)))

        return _scaled_hessian_trace

    @functools.singledispatchmethod
    def project(self, basis: linpde_gp.bases.Basis) -> pn.linops.LinearOperator:
        raise NotImplementedError()


class ScaledSpatialLaplacian(JaxLinearOperator):
    def __init__(self, domain_shape: ShapeLike, alpha: float = 1.0) -> None:
        (D,) = pn.utils.as_shape(domain_shape)

        assert D > 1

        self._alpha = alpha
        self._laplacian = ScaledLaplaceOperator(
            domain_shape=(D - 1,),
            alpha=self._alpha,
        )

        super().__init__(
            input_shapes=((D,), ()),
            output_shapes=((D,), ()),
        )

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)

    def _jax_fallback(self, f, /, *, argnum: int = 0, **kwargs):
        # TODO: Implement using Hessian-vector products
        @jax.jit
        def _f(*args, **kwargs) -> jnp.ndarray:
            t, x = args[argnum : argnum + 2]
            tx = jnp.concatenate((t[None], x), axis=0)

            return f(
                *args[:argnum],
                tx,
                *args[argnum + 2 :],
                **kwargs,
            )

        _spatial_laplace_f_t_x = self._laplacian(_f, argnum=argnum + 1)

        @jax.jit
        def _spatial_laplace_f_tx(*args, **kwargs):
            tx = args[argnum]
            t, x = tx[0], tx[1:]
            args = args[:argnum] + (t, x) + args[argnum + 1 :]

            return _spatial_laplace_f_t_x(*args, **kwargs)

        return _spatial_laplace_f_tx
