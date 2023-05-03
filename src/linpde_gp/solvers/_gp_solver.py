import abc
from collections.abc import Sequence
from dataclasses import dataclass
import pickle
from typing import Optional

import numpy as np
import probnum as pn

from linpde_gp.linfunctls import LinearFunctional
from linpde_gp.randprocs.covfuncs import JaxCovarianceFunction
from linpde_gp.randprocs.crosscov import ProcessVectorCrossCovariance


@dataclass
class GPInferenceParams:
    """
    Parameters for affine Gaussian process inference.
    """

    prior: pn.randprocs.GaussianProcess
    prior_gram: pn.linops.LinearOperator
    Ys: Sequence[np.ndarray]
    Ls: Sequence[LinearFunctional]
    bs: Sequence[pn.randvars.Normal | pn.randvars.Constant | None]
    kLas: ProcessVectorCrossCovariance
    prev_representer_weights: Optional[np.ndarray]
    full_representer_weights: Optional[np.ndarray]


class ConcreteGPSolver(abc.ABC):
    """Abstract base class for concrete Gaussian process solvers.
    Concrete in the sense that we are dealing with one specific
    instance of affine GP regression with a concrete GP and concrete
    linear functionals."""

    def __init__(
        self,
        gp_params: GPInferenceParams,
        load_path: Optional[str] = None,
        save_path: Optional[str] = None,
    ):
        self._gp_params = gp_params
        self._load_path = load_path
        self._save_path = save_path

        # Typically None, but in some cases (e.g. applying a linear function
        # operator to a trained GP), the representer weights are already known
        self._representer_weights = self._gp_params.full_representer_weights

        if self._load_path is not None:
            self.load()

    @abc.abstractmethod
    def _compute_representer_weights(self):
        """
        Compute the representer weights.
        """
        raise NotImplementedError

    def compute_representer_weights(self):
        """
        Compute representer weights, or directly return cached
        result from previous computation.
        """
        if self._representer_weights is None:
            self._representer_weights = self._compute_representer_weights()
            self.save()
        return self._representer_weights

    @property
    @abc.abstractmethod
    def posterior_cov(self) -> JaxCovarianceFunction:
        raise NotImplementedError

    def _get_residual(self, Y, L, b):
        return np.reshape(
            (
                (Y - L(self._gp_params.prior.mean).reshape(-1, order="C"))
                if b is None
                else (
                    Y
                    - L(self._gp_params.prior.mean).reshape(-1, order="C")
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
    def _save_state(self) -> dict:
        """Save solver state to dict."""
        raise NotImplementedError

    @abc.abstractmethod
    def _load_state(self, dict):
        """Load solver state from dict."""
        raise NotImplementedError

    def save(self):
        """Save solver state to file."""
        if self._save_path is None:
            return
        solver_state = self._save_state()
        with open(self._save_path, "wb") as f:
            pickle.dump(solver_state, f)

    def load(self):
        """Load solver state from file."""
        if self._load_path is None:
            return
        with open(self._load_path, "rb") as f:
            loaded_state = pickle.load(f)
        self._load_state(loaded_state)


class GPSolver(abc.ABC):
    """
    User-facing interface for Gaussian process solvers used to pass
    hyperparameters.
    """

    def __init__(
        self, load_path: Optional[str] = None, save_path: Optional[str] = None
    ):
        self._load_path = load_path
        self._save_path = save_path

    @abc.abstractmethod
    def get_concrete_solver(self, gp_params: GPInferenceParams) -> ConcreteGPSolver:
        """
        Get concrete solver.
        Subclasses must implement this method.
        """
        raise NotImplementedError
