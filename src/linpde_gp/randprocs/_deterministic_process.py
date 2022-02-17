from typing import Callable

import numpy as np

import probnum as pn
from probnum.typing import ArrayLike, DTypeLike, ShapeLike


class DeterministicProcess(pn.randprocs.RandomProcess[ArrayLike, np.ndarray]):
    def __init__(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        input_shape: ShapeLike,
        output_shape: ShapeLike,
        dtype: DTypeLike,
    ):
        self._fn = fn

        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            dtype=dtype,
        )

    def __call__(self, args: ArrayLike) -> pn.randvars.Constant:
        return pn.randvars.Constant(support=self._fn(np.asarray(args)))

    @property
    def mean(self) -> pn.Function:
        return pn.LambdaFunction(
            fn=self._fn,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
        )

    def _sample_at_input(
        self,
        rng: np.random.Generator,
        args: np.ndarray,
        size: ShapeLike = (),
    ) -> np.ndarray:
        return self(args).sample(rng, size=size)
