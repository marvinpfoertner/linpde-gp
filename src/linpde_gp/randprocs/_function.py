from typing import Callable, Union

import numpy as np
import probnum as pn
from probnum.typing import ArrayLike, DTypeLike, IntLike, ShapeLike

_InputType = ArrayLike
_OutputType = Union[np.floating, np.ndarray]


class Function(pn.randprocs.RandomProcess[_InputType, _OutputType]):
    def __init__(
        self,
        fn: Callable[[np.ndarray], _OutputType],
        input_dim: IntLike,
        output_dim: IntLike,
        dtype: DTypeLike,
    ):
        self._fn = fn

        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            dtype=dtype,
        )

    def __call__(self, args: _InputType) -> pn.randvars.Constant[_OutputType]:
        y = self._fn(np.asarray(args))

        if np.ndim(args) == 0:
            y = pn.utils.as_numpy_scalar(y)

        return pn.randvars.Constant(support=y)

    def mean(self, args: _InputType) -> _OutputType:
        return self(args).mean

    def _sample_at_input(
        self,
        rng: np.random.Generator,
        args: np.ndarray,
        size: ShapeLike,
    ) -> np.ndarray:
        return pn.randvars.Constant(support=self._fn(args)).sample(rng, size=size)

    def cov(self, args0, args1=None):
        raise NotImplementedError()

    def push_forward(self, args, base_measure, sample):
        raise NotImplementedError()
