from ast import Param
from typing import Callable, Optional, Union

import numpy as np
import probnum as pn
from probnum.typing import ArrayLike, ShapeLike


class ParametricGaussianProcess(pn.randprocs.GaussianProcess):
    def __init__(
        self,
        input_shape: ShapeLike,
        weights: pn.randvars.Normal,
        feature_fn: Callable[[np.ndarray], Union[np.ndarray, pn.linops.LinearOperator]],
        mean: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self._weights = weights
        self._feature_fn = feature_fn

        if mean is None:
            mean = ParametricGaussianProcess.Mean(
                input_shape=input_shape,
                weights=self._weights,
                feature_fn=self._feature_fn,
            )

        super().__init__(
            mean=mean,
            cov=ParametricGaussianProcess.Kernel(
                input_shape=input_shape,
                weights=self._weights,
                feature_fn=self._feature_fn,
            ),
        )

    class Mean(pn.Function):
        def __init__(
            self,
            input_shape: ShapeLike,
            weights: pn.randvars.Normal,
            feature_fn: Callable[
                [np.ndarray], Union[np.ndarray, pn.linops.LinearOperator]
            ],
        ) -> None:
            super().__init__(input_shape, output_shape=())

            self._weights = weights
            self._feature_fn = feature_fn

        def _evaluate(self, x: np.ndarray) -> np.ndarray:
            return self._feature_fn(x) @ self._weights.mean

    class Kernel(pn.randprocs.kernels.Kernel):
        def __init__(
            self,
            input_shape: ShapeLike,
            weights: pn.randvars.Normal,
            feature_fn: Callable[
                [np.ndarray], Union[np.ndarray, pn.linops.LinearOperator]
            ],
        ):
            super().__init__(input_shape=input_shape, output_shape=())

            self._weights = weights
            self._feature_fn = feature_fn

        def _evaluate(self, x0: ArrayLike, x1: Optional[ArrayLike]) -> np.ndarray:
            phi_x0 = self._feature_fn(x0)
            phi_x1 = phi_x0 if x1 is None else self._feature_fn(x1)

            if isinstance(phi_x0, pn.linops.LinearOperator):
                phi_x0_Sigma = phi_x0 @ self._weights.cov

                # TODO: This is inefficient, we need batched linops here
                if isinstance(phi_x0_Sigma, pn.linops.LinearOperator):
                    phi_x0_Sigma = phi_x0_Sigma.todense()

                if isinstance(phi_x1, pn.linops.LinearOperator):
                    phi_x1 = phi_x1.todense()

                return (phi_x0_Sigma[..., None, :] @ phi_x1[..., :, None])[..., 0, 0]

            if isinstance(phi_x1, pn.linops.LinearOperator):
                phi_x1_Sigma = phi_x1 @ self._weights.cov

                # TODO: This is inefficient, we need batched linops here
                if isinstance(phi_x1_Sigma, pn.linops.LinearOperator):
                    phi_x1_Sigma = phi_x1_Sigma.todense()
            else:
                Sigma = pn.linops.aslinop(self._weights.cov)
                phi_x1_Sigma = Sigma(phi_x1, axis=-1)

            assert isinstance(phi_x0, np.ndarray)
            assert isinstance(phi_x1_Sigma, np.ndarray)

            return (phi_x0[..., None, :] @ phi_x1_Sigma[..., :, None])[..., 0, 0]
