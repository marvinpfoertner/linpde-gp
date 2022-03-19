from __future__ import annotations

from collections.abc import Callable
import functools
from typing import TYPE_CHECKING

import jax
from jax import numpy as jnp
import probnum as pn
from probnum.typing import ShapeLike

from .... import linfuncops
from ._directional_derivative import TimeDerivative
from ._laplace import ScaledSpatialLaplacian, scaled_laplace_jax

if TYPE_CHECKING:
    import linpde_gp


def heat_jax(f: Callable, /, *, argnum: int = 0) -> Callable:
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

    df_dt = jax.grad(_f, argnums=argnum)
    laplace_f = scaled_laplace_jax(_f, argnum=argnum + 1)

    @jax.jit
    def _f_heat(*args, **kwargs) -> jnp.ndarray:
        tx = args[argnum]
        t, x = tx[0], tx[1:]
        args = args[:argnum] + (t, x) + args[argnum + 1 :]

        return df_dt(*args, **kwargs) - laplace_f(*args, **kwargs)

    return _f_heat


class HeatOperator(linfuncops.SumLinearFunctionOperator):
    def __init__(self, domain_shape: ShapeLike, alpha: float = 1.0) -> None:
        self._alpha = float(alpha)

        super().__init__(
            TimeDerivative(domain_shape),
            ScaledSpatialLaplacian(domain_shape, alpha=-self._alpha),
        )

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)
