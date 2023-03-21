import abc
from collections.abc import Sequence

import numpy as np
import probnum as pn
from linpde_gp.linops import BlockMatrix2x2
from linpde_gp.linfunctls import LinearFunctional
from dataclasses import dataclass


@dataclass
class GPInferenceParams:
    prior_mean: pn.functions.Function
    prior_gram: pn.linops.LinearOperator
    Ys: Sequence[np.ndarray]
    Ls: Sequence[LinearFunctional]
    bs: Sequence[pn.randvars.Normal | pn.randvars.Constant | None]
    prior_representer_weights: np.ndarray


class ConcreteGPSolver(abc.ABC):
    def __init__(self, gp_params: GPInferenceParams):
        self._gp_params = gp_params

    def compute_representer_weights(self):
        try:
            return self._representer_weights
        except AttributeError:
            self._representer_weights = self._compute_representer_weights()
            return self._representer_weights

    @abc.abstractmethod
    def _compute_representer_weights(self):
        raise NotImplementedError

    def _get_residual(self, Y, L, b):
        return np.reshape(
            (
                (Y - L(self._gp_params.prior_mean).reshape(-1, order="C"))
                if b is None
                else (
                    Y
                    - L(self._gp_params.prior_mean).reshape(-1, order="C")
                    - b.mean.reshape(-1, order="C")
                )
            ),
            (-1,),
            order="C",
        )

    def _get_full_residual(self):
        return np.concatenate(
            [
                self._get_residual(Y, L, b)
                for Y, L, b in zip(
                    self._gp_params.Ys, self._gp_params.Ls, self._gp_params.bs
                )
            ],
            axis=-1,
        )

    @abc.abstractmethod
    def compute_posterior_cov(
        self, k_xx: np.ndarray, k_xX: np.ndarray, k_Xx: np.ndarray
    ):
        raise NotImplementedError


class GPSolver(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_concrete_solver(self, gp_params: GPInferenceParams) -> ConcreteGPSolver:
        raise NotImplementedError
