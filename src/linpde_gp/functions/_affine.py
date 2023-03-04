from jax import numpy as jnp
import numpy as np
from probnum.typing import ArrayLike

from . import _jax


class Affine(_jax.JaxFunction):
    def __init__(self, A: ArrayLike, b: ArrayLike) -> None:
        self._A = np.asarray(A)
        self._b = np.asarray(b)

        if self._A.ndim == 0:
            input_shape = ()
            output_shape = ()
        elif self._A.ndim == 1:
            input_shape = ()
            output_shape = self._A.shape
        elif self._A.ndim == 2:
            input_shape = (self._A.shape[1],)
            output_shape = (self._A.shape[0],)
        else:
            raise ValueError("TODO")

        if self._b.shape != output_shape:
            raise ValueError("TODO")

        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
        )

    @property
    def A(self) -> np.ndarray:
        return self._A

    @property
    def b(self) -> np.ndarray:
        return self._b

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        if self._input_shape == ():
            return self._A * x + self._b

        return (self._A @ x[..., None])[..., 0] + self._b

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        if self._input_shape == ():
            return self._A * x + self._b

        return (self._A @ x[..., None])[..., 0] + self._b
