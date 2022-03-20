from typing import Optional

from jax import numpy as jnp
import numpy as np

from linpde_gp.linfuncops import diffops

from .._expquad import ExpQuad
from .._jax import JaxKernel
from ._expquad_directional_derivative import ExpQuad_Identity_DirectionalDerivative
from ._expquad_laplacian import (
    ExpQuad_DirectionalDerivative_Laplacian,
    ExpQuad_Identity_Laplacian,
    ExpQuad_Laplacian_Laplacian,
)


class ExpQuad_Identity_SpatialLaplacian(JaxKernel):
    def __init__(self, expquad: ExpQuad, reverse: bool):
        if expquad.input_ndim != 1 or expquad.input_size < 2:
            raise ValueError()

        self._expquad = expquad

        super().__init__(self._expquad.input_shape, output_shape=())

        self._reverse = reverse

        self._expquad_time = ExpQuad(
            input_shape=(),
            lengthscales=(
                self._expquad.lengthscales
                if self._expquad.lengthscales.ndim == 0
                else self._expquad.lengthscales[0]
            ),
        )

        self._expquad_space = ExpQuad(
            input_shape=(self.input_size - 1,),
            lengthscales=(
                self._expquad.lengthscales
                if self._expquad.lengthscales.ndim == 0
                else self._expquad.lengthscales[1:]
            ),
        )

        self._expquad_laplacian_space = ExpQuad_Identity_Laplacian(
            expquad=self._expquad_space,
            reverse=self._reverse,
        )

    @property
    def expquad(self) -> ExpQuad:
        return self._expquad

    @property
    def reverse(self) -> bool:
        return self._reverse

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        return self._expquad_time(
            x0[..., 0], None if x1 is None else x1[..., 0]
        ) * self._expquad_laplacian_space(
            x0[..., 1:], None if x1 is None else x1[..., 1:]
        )

    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        return self._expquad_time.jax(
            x0[..., 0], None if x1 is None else x1[..., 0]
        ) * self._expquad_laplacian_space.jax(
            x0[..., 1:], None if x1 is None else x1[..., 1:]
        )


@diffops.SpatialLaplacian.__call__.register  # pylint: disable=no-member
def _(self, k: ExpQuad, /, *, argnum: int = 0):  # pylint: disable=unused-argument
    return ExpQuad_Identity_SpatialLaplacian(expquad=k, reverse=(argnum == 0))


class ExpQuad_SpatialLaplacian_SpatialLaplacian(JaxKernel):
    def __init__(self, expquad: ExpQuad):
        if expquad.input_ndim != 1 or expquad.input_size < 2:
            raise ValueError()

        self._expquad = expquad

        super().__init__(self._expquad.input_shape, output_shape=())

        self._expquad_time = ExpQuad(
            input_shape=(),
            lengthscales=(
                self._expquad.lengthscales
                if self._expquad.lengthscales.ndim == 0
                else self._expquad.lengthscales[0]
            ),
        )

        self._expquad_laplacian_laplacian_space = ExpQuad_Laplacian_Laplacian(
            expquad=ExpQuad(
                input_shape=(self.input_size - 1,),
                lengthscales=(
                    self._expquad.lengthscales
                    if self._expquad.lengthscales.ndim == 0
                    else self._expquad.lengthscales[1:]
                ),
            )
        )

    @property
    def expquad(self) -> ExpQuad:
        return self._expquad

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        return self._expquad_time(
            x0[..., 0], None if x1 is None else x1[..., 0]
        ) * self._expquad_laplacian_laplacian_space(
            x0[..., 1:], None if x1 is None else x1[..., 1:]
        )

    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        return self._expquad_time.jax(
            x0[..., 0], None if x1 is None else x1[..., 0]
        ) * self._expquad_laplacian_laplacian_space.jax(
            x0[..., 1:], None if x1 is None else x1[..., 1:]
        )


@diffops.SpatialLaplacian.__call__.register  # pylint: disable=no-member
def _(self, k: ExpQuad_Identity_SpatialLaplacian, /, *, argnum: int = 0):
    if (argnum == 0 and not k.reverse) or (argnum == 1 and k.reverse):
        return ExpQuad_SpatialLaplacian_SpatialLaplacian(expquad=k.expquad)

    return super(diffops.Laplacian, self).__call__(k, argnum=argnum)


class ExpQuad_DirectionalDerivative_SpatialLaplacian(JaxKernel):
    def __init__(
        self,
        expquad: ExpQuad,
        direction: np.ndarray,
        reverse: bool = False,
        expquad_spatial_laplacian: Optional[ExpQuad_Identity_SpatialLaplacian] = None,
    ):
        if expquad.input_ndim != 1 or expquad.input_size < 2:
            raise ValueError()

        self._expquad = expquad

        super().__init__(self._expquad.input_shape, output_shape=())

        self._direction = direction

        self._reverse = bool(reverse)

        if expquad_spatial_laplacian is None:
            expquad_spatial_laplacian = ExpQuad_Identity_SpatialLaplacian(
                expquad=self._expquad,
                reverse=self._reverse,
            )

        assert expquad_spatial_laplacian._reverse == self._reverse

        self._expquad_spatial_laplacian = expquad_spatial_laplacian

        self._expquad_time = self._expquad_spatial_laplacian._expquad_time

        self._expquad_ddv_time = ExpQuad_Identity_DirectionalDerivative(
            expquad=self._expquad_time,
            direction=self._direction[0],
            reverse=not reverse,
        )

        self._expquad_laplacian_space = (
            self._expquad_spatial_laplacian._expquad_laplacian_space
        )

        self._expquad_ddv_laplacian_space = ExpQuad_DirectionalDerivative_Laplacian(
            expquad=self._expquad_spatial_laplacian._expquad_space,
            direction=self._direction[1:],
            reverse=self._reverse,
            expquad_laplacian=self._expquad_spatial_laplacian._expquad_laplacian_space,
        )

    @property
    def expquad(self) -> ExpQuad:
        return self._expquad

    @property
    def reverse(self) -> bool:
        return self._reverse

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        return (
            self._expquad_ddv_time(x0[..., 0], None if x1 is None else x1[..., 0])
            * self._expquad_laplacian_space(
                x0[..., 1:], None if x1 is None else x1[..., 1:]
            )
        ) + (
            self._expquad_time(x0[..., 0], None if x1 is None else x1[..., 0])
            * self._expquad_ddv_laplacian_space(
                x0[..., 1:], None if x1 is None else x1[..., 1:]
            )
        )

    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        return (
            self._expquad_ddv_time.jax(x0[..., 0], None if x1 is None else x1[..., 0])
            * self._expquad_laplacian_space.jax(
                x0[..., 1:], None if x1 is None else x1[..., 1:]
            )
        ) + (
            self._expquad_time.jax(x0[..., 0], None if x1 is None else x1[..., 0])
            * self._expquad_ddv_laplacian_space.jax(
                x0[..., 1:], None if x1 is None else x1[..., 1:]
            )
        )


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: ExpQuad_Identity_SpatialLaplacian, /, *, argnum: int = 0):
    if (argnum == 0 and not k.reverse) or (argnum == 1 and k.reverse):
        return ExpQuad_DirectionalDerivative_SpatialLaplacian(
            expquad=k.expquad,
            direction=self.direction,
            reverse=(argnum == 1),
            expquad_spatial_laplacian=k,
        )

    return super(diffops.DirectionalDerivative, self).__call__(k, argnum=argnum)


@diffops.SpatialLaplacian.__call__.register  # pylint: disable=no-member
def _(self, k: ExpQuad_Identity_DirectionalDerivative, /, *, argnum: int = 0):
    if (argnum == 0 and not k.reverse) or (argnum == 1 and k.reverse):
        return ExpQuad_DirectionalDerivative_SpatialLaplacian(
            expquad=k.expquad,
            direction=k.direction,
            reverse=(argnum == 0),
        )

    return super(diffops.Laplacian, self).__call__(k, argnum=argnum)
