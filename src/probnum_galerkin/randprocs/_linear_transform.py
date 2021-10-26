from typing import Callable, Optional, Union

import numpy as np
import probnum as pn
from probnum.typing import ArrayLike, IntArgType


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
            mean = lambda x: self._linop_fn(x[..., 0]) @ self._base_rv.mean

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

            super().__init__(input_dim=input_dim)

        def _evaluate(
            self, x0: ArrayLike, x1: Optional[ArrayLike]
        ) -> Union[np.ndarray, np.float_]:
            linop_x0 = self._linop_fn(np.squeeze(x0, axis=-1))
            linop_x1 = (
                linop_x0 if x1 is None else self._linop_fn(np.squeeze(x1, axis=-1))
            )

            # TODO: This is inefficient, we need batches of linear operators for this
            if isinstance(linop_x0, pn.linops.LinearOperator):
                linop_x0 = linop_x0.todense()

            if isinstance(linop_x1, pn.linops.LinearOperator):
                linop_x1 = linop_x1.todense()

            return np.einsum(
                "...i,...i",
                linop_x0 @ self._base_rv.cov,
                linop_x1,
            )
