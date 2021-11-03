import jax
import numpy as np
from linpde_gp._compute_backend import BackendDispatcher


class JaxMean:
    def __init__(self, m, vectorize: bool = True):
        if vectorize:
            m = jax.numpy.vectorize(m, signature="(d)->()")

        self._dispatcher = BackendDispatcher(
            numpy_impl=lambda x: np.asarray(m(x)),
            jax_impl=m,
        )

    def __call__(self, x):
        return self._dispatcher(x)
