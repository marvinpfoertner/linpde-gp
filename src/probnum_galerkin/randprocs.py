from typing import Callable, Optional, Union

import numpy as np
import probnum as pn
from numpy import typing as npt
from probnum.type import DTypeArgType, IntArgType, RandomStateArgType, ShapeArgType

_InputType = npt.ArrayLike
_OutputType = Union[np.floating, np.ndarray]


class Function(pn.randprocs.RandomProcess[_InputType, _OutputType]):
    def __init__(
        self,
        fn: Callable[[np.ndarray], _OutputType],
        input_dim: IntArgType,
        output_dim: IntArgType,
        dtype: DTypeArgType,
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
        self, args: np.ndarray, size: ShapeArgType, random_state: RandomStateArgType
    ) -> np.ndarray:
        return pn.randvars.Constant(
            support=self._fn(args), random_state=random_state
        ).sample(size=size)

    def cov(self, args0, args1=None):
        raise NotImplementedError()

    def push_forward(self, args, base_measure, sample):
        raise NotImplementedError()


class LinearTransformGaussianProcess(pn.randprocs.GaussianProcess):
    def __init__(
        self,
        input_dim: IntArgType,
        base_rv: pn.randvars.Normal,
        linop_fn: Callable[
            [np.ndarray], Union[np.ndarray, pn.linops.LinearOperatorLike]
        ],
        mean: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self._base_rv = base_rv
        self._linop_fn = linop_fn

        if mean is None:
            mean = lambda x: self._linop_fn(x) @ self._base_rv.mean

        super().__init__(
            mean=mean,
            cov=LinearTransformGaussianProcess.Kernel(
                input_dim=input_dim,
                base_rv=base_rv,
                linop_fn=linop_fn,
            ),
        )

    class Kernel(pn.kernels.Kernel):
        def __init__(
            self,
            input_dim: IntArgType,
            base_rv: pn.randvars.Normal,
            linop_fn: Callable[
                [np.ndarray], Union[np.ndarray, pn.linops.LinearOperatorLike]
            ],
        ):
            self._base_rv = base_rv
            self._linop_fn = linop_fn

            if input_dim != 1:
                raise NotImplementedError

            super().__init__(input_dim=input_dim, output_dim=1)

        def __call__(
            self, x0: np.ndarray, x1: Optional[np.ndarray] = None
        ) -> Union[np.ndarray, np.float_]:
            # Check and reshape inputs
            x0, x1, kernshape = self._check_and_reshape_inputs(x0, x1)

            # Compute kernel matrix
            linop_0 = self._linop_fn(np.squeeze(x0))

            if x1 is None:
                linop_1 = linop_0
            else:
                linop_1 = self._linop_fn(np.squeeze(x1))

            kernmat = linop_0 @ self._base_rv.cov @ linop_1.T

            return self._reshape_kernelmatrix(kernmat, newshape=kernshape)
