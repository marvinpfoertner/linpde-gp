import functools
import operator
from typing import Optional

from jax import numpy as jnp
import numpy as np
from probnum.randprocs.kernels._arithmetic_fallbacks import ScaledKernel, SumKernel
from probnum.typing import ArrayLike, ScalarLike, ScalarType

from ... import linfuncops
from ._jax import JaxKernel, JaxKernelMixin


class JaxScaledKernel(JaxKernelMixin, ScaledKernel):
    def __init__(self, kernel: JaxKernel, scalar: ScalarLike) -> None:
        if not isinstance(kernel, JaxKernelMixin):
            raise TypeError()

        super().__init__(kernel, scalar)

    @property
    def scalar(self) -> ScalarType:
        return self._scalar

    @property
    def kernel(self) -> JaxKernel:
        return self._kernel

    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        return self._scalar * self.kernel.jax(x0, x1)

    def __rmul__(self, other: ArrayLike) -> JaxKernel:
        if np.ndim(other) == 0:
            return JaxScaledKernel(
                self.kernel,
                scalar=np.asarray(other) * self.scalar,
            )

        return super().__rmul__(other)


@linfuncops.LinearFunctionOperator.__call__.register  # pylint: disable=no-member
def _(self, k: JaxScaledKernel, /, *, argnum: int = 0) -> JaxScaledKernel:
    return k.scalar * self(k.kernel, argnum=argnum)


class JaxSumKernel(JaxKernelMixin, SumKernel):
    def __init__(self, *summands: JaxKernel):
        if not all(isinstance(summand, JaxKernelMixin) for summand in summands):
            raise TypeError()

        super().__init__(*summands)

    @property
    def summands(self) -> tuple[JaxKernel, ...]:
        return self._summands

    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        return functools.reduce(
            operator.add,
            (summand.jax(x0, x1) for summand in self.summands),
        )


@linfuncops.LinearFunctionOperator.__call__.register  # pylint: disable=no-member
def _(self, k: JaxSumKernel, /, *, argnum: int = 0) -> JaxSumKernel:
    return JaxSumKernel(*(self(summand, argnum=argnum) for summand in k.summands))
