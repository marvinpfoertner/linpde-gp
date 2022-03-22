from typing import Optional

import numpy as np
import probnum as pn
from probnum.typing import ArrayLike


class ParametricGaussianProcess(pn.randprocs.GaussianProcess):
    def __init__(
        self,
        weights: pn.randvars.Normal,
        feature_fn: pn.Function,
        mean: Optional[pn.Function] = None,
    ):
        self._weights = weights
        self._feature_fn = feature_fn

        if mean is None:
            mean = ParametricGaussianProcess.Mean(
                weights=self._weights,
                feature_fn=self._feature_fn,
            )

        super().__init__(
            mean=mean,
            cov=ParametricGaussianProcess.Kernel(
                weights=self._weights,
                feature_fn=self._feature_fn,
            ),
        )

    class Mean(pn.Function):
        def __init__(
            self, weights: pn.randvars.Normal, feature_fn: pn.Function
        ) -> None:
            self._weights = weights
            self._feature_fn = feature_fn

            super().__init__(input_shape=self._feature_fn.input_shape, output_shape=())

        def _evaluate(self, x: np.ndarray) -> np.ndarray:
            if self._feature_fn.output_shape == ():
                return self._feature_fn(x) * self._weights.mean

            return self._feature_fn(x) @ self._weights.mean

    class Kernel(pn.randprocs.kernels.Kernel):
        def __init__(
            self,
            weights: pn.randvars.Normal,
            feature_fn: pn.Function,
        ):
            self._weights = weights
            self._feature_fn = feature_fn

            super().__init__(input_shape=self._feature_fn.input_shape, output_shape=())

        def _evaluate(self, x0: ArrayLike, x1: Optional[ArrayLike]) -> np.ndarray:
            if self._feature_fn.output_shape == ():
                phi_x0 = self._feature_fn(x0)[..., None]
                phi_x1 = phi_x0 if x1 is None else self._feature_fn(x1)[..., None]
                Sigma = self._weights.cov[None, None]
            else:
                phi_x0 = self._feature_fn(x0)
                phi_x1 = phi_x0 if x1 is None else self._feature_fn(x1)
                Sigma = self._weights.cov

            if isinstance(phi_x0, pn.linops.LinearOperator):
                phi_x0_Sigma = phi_x0 @ Sigma

                # TODO: This is inefficient, we need batched linops here
                if isinstance(phi_x0_Sigma, pn.linops.LinearOperator):
                    phi_x0_Sigma = phi_x0_Sigma.todense()

                if isinstance(phi_x1, pn.linops.LinearOperator):
                    phi_x1 = phi_x1.todense()

                return (phi_x0_Sigma[..., None, :] @ phi_x1[..., :, None])[..., 0, 0]

            if isinstance(phi_x1, pn.linops.LinearOperator):
                phi_x1_Sigma = phi_x1 @ Sigma

                # TODO: This is inefficient, we need batched linops here
                if isinstance(phi_x1_Sigma, pn.linops.LinearOperator):
                    phi_x1_Sigma = phi_x1_Sigma.todense()
            else:
                Sigma = pn.linops.aslinop(Sigma)
                phi_x1_Sigma = Sigma(phi_x1, axis=-1)

            assert isinstance(phi_x0, np.ndarray)
            assert isinstance(phi_x1_Sigma, np.ndarray)

            return (phi_x0[..., None, :] @ phi_x1_Sigma[..., :, None])[..., 0, 0]
