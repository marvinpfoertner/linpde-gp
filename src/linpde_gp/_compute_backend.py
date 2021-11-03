import enum
from typing import Any, Callable, Optional


class Backend(enum.Enum):
    NUMPY = "numpy"
    JAX = "jax"


_BACKEND = Backend.JAX


def set_backend(backend: Backend) -> None:
    assert backend in Backend

    global _BACKEND
    _BACKEND = backend


class BackendDispatcher:
    def __init__(
        self,
        numpy_impl: Optional[Callable[..., Any]],
        jax_impl: Optional[Callable[..., Any]] = None,
    ):
        self._impl = {}

        if numpy_impl is not None:
            self._impl[Backend.NUMPY] = numpy_impl

        if jax_impl is not None:
            self._impl[Backend.JAX] = jax_impl

    def __call__(self, *args, **kwargs) -> Any:
        return self._impl[_BACKEND](*args, **kwargs)
