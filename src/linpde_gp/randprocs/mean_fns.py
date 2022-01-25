import numpy as np
from jax import numpy as jnp


class JaxMean:
    def __init__(self, m, vectorize: bool = True):
        if vectorize:
            m = jnp.vectorize(m, signature="(d)->()")

        self._m = m

    def __call__(self, x: np.ndarray):
        return np.array(self._m(x))

    @property
    def jax(self):
        return self._m
