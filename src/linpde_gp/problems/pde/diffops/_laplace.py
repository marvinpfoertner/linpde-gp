from __future__ import annotations

from collections.abc import Callable
import functools
from typing import TYPE_CHECKING

import jax
from jax import numpy as jnp
import probnum as pn
from probnum.typing import ShapeLike

from .... import linfuncops

if TYPE_CHECKING:
    import linpde_gp


def scaled_laplace_jax(
    f: linfuncops.JaxFunction, /, *, argnum: int = 0, alpha: float = 1.0
) -> linfuncops.JaxFunction:
    Hf = jax.jit(jax.hessian(f, argnum))

    @jax.jit
    def _scaled_hessian_trace(*args, **kwargs):
        return alpha * jnp.trace(jnp.atleast_2d(Hf(*args, **kwargs)))

    return _scaled_hessian_trace


class ScaledLaplaceOperator(linfuncops.JaxLinearOperator):
    def __init__(self, domain_shape: ShapeLike, alpha: float = 1.0) -> None:
        self._alpha = alpha

        super().__init__(
            input_shapes=(domain_shape, ()),
            output_shapes=(domain_shape, ()),
        )

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)

    def _jax_fallback(self, f: Callable, /, **kwargs) -> Callable:
        return scaled_laplace_jax(f, alpha=self._alpha, **kwargs)

    @functools.singledispatchmethod
    def project(self, basis: linpde_gp.bases.Basis) -> pn.linops.LinearOperator:
        raise NotImplementedError()


class ScaledSpatialLaplacian(linfuncops.JaxLinearOperator):
    def __init__(self, domain_shape: ShapeLike, alpha: float = 1.0) -> None:
        (D,) = pn.utils.as_shape(domain_shape)

        assert D > 1

        self._alpha = alpha

        super().__init__(
            input_shapes=(domain_shape, ()),
            output_shapes=(domain_shape, ()),
        )

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)

    def _jax_fallback(self, f, argnum: int = 0):
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

        _spatial_laplace_f_t_x = scaled_laplace_jax(
            _f, argnum=argnum + 1, alpha=self._alpha
        )

        @jax.jit
        def _spatial_laplace_f_tx(*args, **kwargs):
            tx = args[argnum]
            t, x = tx[0], tx[1:]
            args = args[:argnum] + (t, x) + args[argnum + 1 :]

            return _spatial_laplace_f_t_x(*args, **kwargs)

        return _spatial_laplace_f_tx
