from typing import Any, Protocol, Union

from jax import numpy as jnp


class JaxFunction(Protocol):
    def __call__(self, *args: Union[jnp.ndarray, Any], **kwargs) -> jnp.ndarray:
        ...


class JaxLinearOperator(Protocol):
    def __call__(self, f: JaxFunction, argnum: int = 0, **kwargs) -> JaxFunction:
        ...
