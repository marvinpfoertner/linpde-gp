from __future__ import annotations

from typing import Optional
from collections.abc import Iterator, Sequence
import functools

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
import probnum as pn
import scipy.linalg

from linpde_gp import linfunctls
from linpde_gp.functions import JaxFunction
from linpde_gp.linfuncops import LinearFunctionOperator
from linpde_gp.linfunctls import LinearFunctional
from linpde_gp.randprocs.covfuncs import JaxCovarianceFunction
from linpde_gp.linops import BlockMatrix2x2, BlockMatrix
from linpde_gp.randprocs.crosscov import ProcessVectorCrossCovariance
from linpde_gp.typing import RandomVariableLike


class ConditionalGaussianProcess(pn.randprocs.GaussianProcess):
    @classmethod
    def from_observations(
        cls,
        prior: pn.randprocs.GaussianProcess,
        Y: ArrayLike,
        X: ArrayLike | None = None,
        *,
        L: None | LinearFunctional | LinearFunctionOperator = None,
        b: None | RandomVariableLike = None,
    ):
        Y, L, b, kLa, Lm, gram = cls._preprocess_observations(
            prior=prior,
            Y=Y,
            X=X,
            L=L,
            b=b,
        )

        representer_weights = gram.solve(Y - Lm)

        return cls(
            prior=prior,
            Ys=(Y,),
            Ls=(L,),
            bs=(b,),
            kLas=ConditionalGaussianProcess._PriorPredictiveCrossCovariance((kLa,)),
            gram_matrix=gram,
            representer_weights=representer_weights,
        )

    def __init__(
        self,
        *,
        prior: pn.randprocs.GaussianProcess,
        Ys: Sequence[np.ndarray],
        Ls: Sequence[LinearFunctional],
        bs: Sequence[pn.randvars.Normal | pn.randvars.Constant | None],
        kLas: ConditionalGaussianProcess._PriorPredictiveCrossCovariance,
        gram_matrix: pn.linops.LinearOperator,
        representer_weights: np.ndarray | None = None,
    ):
        self._prior = prior

        self._Ys = tuple(Ys)
        self._Ls = tuple(Ls)
        self._bs = tuple(bs)

        self._kLas = kLas

        self._gram_matrix = gram_matrix

        self._representer_weights = representer_weights

        super().__init__(
            mean=ConditionalGaussianProcess.Mean(
                prior_mean=self._prior.mean,
                kLas=self._kLas,
                representer_weights=self.representer_weights,
            ),
            cov=ConditionalGaussianProcess.CovarianceFunction(
                prior_covfunc=self._prior.cov,
                kLas=self._kLas,
                gram_matrix=self.gram,
            ),
        )

    @functools.cached_property
    def gram(self) -> pn.linops.LinearOperator:
        return self._gram_matrix

    @property
    def representer_weights(self) -> np.ndarray:
        if self._representer_weights is None:
            y = np.concatenate(
                [
                    (Y - L(self._prior.mean))
                    if b is None
                    else (Y - L(self._prior.mean) - b.mean.reshape(-1, order="C"))
                    for Y, L, b in zip(self._Ys, self._Ls, self._bs)
                ],
                axis=-1,
            )
            self._representer_weights = self.gram.solve(y)

        return self._representer_weights

    class _PriorPredictiveCrossCovariance(ProcessVectorCrossCovariance):
        def __init__(
            self,
            kLas: Sequence[ProcessVectorCrossCovariance],
        ) -> None:
            self._kLas = tuple(kLas)

            assert all(
                kLa.randproc_input_shape == self._kLas[0].randproc_input_shape
                and kLa.randproc_output_shape == self._kLas[0].randproc_output_shape
                and not kLa.reverse
                for kLa in self._kLas
            )

            super().__init__(
                randproc_input_shape=self._kLas[0].randproc_input_shape,
                randproc_output_shape=self._kLas[0].randproc_output_shape,
                randvar_shape=(sum(kLa.randvar_size for kLa in self._kLas),),
                reverse=False,
            )

        def append(
            self, kLa: ProcessVectorCrossCovariance
        ) -> ConditionalGaussianProcess._PriorPredictiveCrossCovariance:
            return ConditionalGaussianProcess._PriorPredictiveCrossCovariance(
                self._kLas + (kLa,)
            )

        def _evaluate(self, x: np.ndarray) -> np.ndarray:
            batch_shape = x.shape[: x.ndim - self.randproc_input_ndim]

            return np.concatenate(
                [
                    np.reshape(
                        kLa(x),  # shape: batch_shape + u_output_shape + Lu_output_shape
                        batch_shape + self.randproc_output_shape + (-1,),
                        "C",
                    )
                    for kLa in self._kLas
                ],
                axis=-1,
            )

        def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
            batch_shape = x.shape[: x.ndim - self.randproc_input_ndim]

            return jnp.concatenate(
                [
                    jnp.reshape(
                        kLa(x),  # shape: batch_shape + u_output_shape + Lu_output_shape
                        batch_shape + self.randproc_output_shape + (-1,),
                        "C",
                    )
                    for kLa in self._kLas
                ],
                axis=-1,
            )

        def _evaluate_linop(self, x: np.ndarray) -> pn.linops.LinearOperator:
            return BlockMatrix([kLa.evaluate_linop(x) for kLa in self._kLas])

        def __iter__(self) -> Iterator[ProcessVectorCrossCovariance]:
            for kLa in self._kLas:
                yield kLa

    class Mean(JaxFunction):
        def __init__(
            self,
            prior_mean: JaxFunction,
            kLas: ConditionalGaussianProcess._PriorPredictiveCrossCovariance,
            representer_weights: np.ndarray,
        ):
            self._prior_mean = prior_mean
            self._kLas = kLas
            self._representer_weights = representer_weights

            super().__init__(
                input_shape=self._prior_mean.input_shape,
                output_shape=self._prior_mean.output_shape,
            )

        def _evaluate(self, x: np.ndarray) -> np.ndarray:
            m_x = self._prior_mean(x)
            kLas_x = self._kLas(x)

            return m_x + kLas_x @ self._representer_weights

        @functools.partial(jax.jit, static_argnums=0)
        def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
            m_x = self._prior_mean.jax(x)
            kLas_x = self._kLas.jax(x)

            return m_x + kLas_x @ self._representer_weights

    class CovarianceFunction(JaxCovarianceFunction):
        def __init__(
            self,
            prior_covfunc: JaxCovarianceFunction,
            kLas: ConditionalGaussianProcess._PriorPredictiveCrossCovariance,
            gram_matrix: pn.linops.LinearOperator,
        ):
            self._prior_covfunc = prior_covfunc
            self._kLas = kLas
            self._gram_matrix = gram_matrix

            super().__init__(
                input_shape=self._prior_covfunc.input_shape,
                output_shape_0=self._prior_covfunc.output_shape_0,
                output_shape_1=self._prior_covfunc.output_shape_1,
            )

        def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
            k_xx = self._prior_covfunc(x0, x1)
            kLas_x0 = self._kLas(x0)
            kLas_x1 = self._kLas(x1) if x1 is not None else kLas_x0
            cov_update = (kLas_x0[..., None, :] @ (self._gram_matrix.solve(kLas_x1[..., None])))[..., 0, 0]

            return k_xx - cov_update

        @functools.partial(jax.jit, static_argnums=0)
        def _evaluate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
            k_xx = self._prior_covfunc.jax(x0, x1)
            kLas_x0 = self._kLas.jax(x0)
            kLas_x1 = self._kLas.jax(x1) if x1 is not None else kLas_x0
            cov_update = kLas_x0[..., None, :] @ (self._gram_matrix.solve(kLas_x1[..., None]))[..., 0, 0]

            return k_xx - cov_update

        def _evaluate_linop(
            self, x0: np.ndarray, x1: Optional[np.ndarray]
        ) -> pn.linops.LinearOperator:
            k_xx = self._prior_covfunc.linop(x0, x1)
            kLas_x0 = self._kLas.evaluate_linop(x0)
            kLas_x1 = self._kLas.evaluate_linop(x1) if x1 is not None else kLas_x0
            return k_xx - kLas_x0 @ self._gram_matrix.solve(kLas_x1.T)

    def condition_on_observations(
        self,
        Y: ArrayLike,
        X: ArrayLike | None = None,
        *,
        L: LinearFunctional | LinearFunctionOperator | None = None,
        b: RandomVariableLike | None = None,
    ):
        Y, L, b, kLa, pred_mean, gram = self._preprocess_observations(
            prior=self._prior,
            Y=Y,
            X=X,
            L=L,
            b=b,
        )

        # Compute lower-left block in the new covariance matrix
        gram_L_La_prev_blocks = L(self._kLas)

        # Update the Cholesky decomposition of the previous covariance matrix and the
        # representer weights

        gram_matrix = BlockMatrix2x2(
            self.gram,
            gram_L_La_prev_blocks.T,
            None,
            gram,
            is_spd=True,
        )
        representer_weights = gram_matrix.schur_update(
            self.representer_weights, Y - pred_mean
        )

        return ConditionalGaussianProcess(
            prior=self._prior,
            Ys=self._Ys + (Y,),
            Ls=self._Ls + (L,),
            bs=self._bs + (b,),
            kLas=self._kLas.append(kLa),
            gram_matrix=gram_matrix,
            representer_weights=representer_weights,
        )

    @classmethod
    def _preprocess_observations(
        cls,
        *,
        prior: pn.randprocs.GaussianProcess,
        Y: ArrayLike,
        X: ArrayLike | None,
        L: LinearFunctional | LinearFunctionOperator | None,
        b: RandomVariableLike | None,
    ) -> tuple[
        np.ndarray,
        LinearFunctional,
        pn.randvars.Normal | pn.randvars.Constant | None,
        ProcessVectorCrossCovariance,
        np.ndarray,
        pn.linops.LinearOperator,
    ]:
        # TODO: Allow `RandomProcessLike` for `b` ("b = b(X)")

        # Build measurement functional `L`
        match L:
            case LinearFunctional():
                if X is not None:
                    raise TypeError(
                        "If `L` is a `LinearFunctional`, `X` must be `None`."
                    )
            case LinearFunctionOperator():
                if X is None:
                    raise ValueError(
                        "`X` must not be omitted if `L` is a `LinearFunctionOperator`."
                    )

                L = L.to_linfunctl(X)
            case None:
                if X is None:
                    raise ValueError("`X` and `L` can not be omitted at the same time.")

                L = linfunctls._EvaluationFunctional(
                    input_domain_shape=prior.input_shape,
                    input_codomain_shape=prior.output_shape,
                    X=X,
                )
            case _:
                raise TypeError("TODO")

        assert isinstance(L, LinearFunctional)

        # Check measurement noise model
        if b is not None:
            b = pn.randvars.asrandvar(b)

            if not isinstance(b, (pn.randvars.Constant, pn.randvars.Normal)):
                raise TypeError(
                    f"`b` must be a `Normal` or a `Constant` `RandomVariable`"
                    f"({type(b)=})"
                )

            if b.shape != L.output_shape:
                raise ValueError(f"{b.shape=} must be equal to {L.output_shape}")

        # Compute the joint measure (f, L[f])
        Lf = L(prior)
        kLa = L(prior.cov, argnum=1)

        # Compute predictive mean and covariance matrix
        pred_mean = Lf.mean
        gram = Lf.cov

        pred_mean = pred_mean.reshape(-1, order="C")
        # Check observations
        Y = np.asarray(Y)
        if isinstance(L, linfunctls._EvaluationFunctional) and prior.mean.output_ndim > 0:
            if Y.shape[-prior.mean.output_ndim:] != prior.mean.output_shape:
                raise ValueError(f"Expected Y to have shape (batch shape) + {prior.mean.output_shape}, got shape {Y.shape}")
            Y = np.moveaxis(
                Y,
                -prior.mean.output_ndim + np.arange(prior.mean.output_ndim),
                np.arange(prior.mean.output_ndim),
            )
        if Y.shape != L.output_shape:
            raise ValueError(f"Expected Y to have shape {L.output_shape}, got shape {Y.shape}.")
        Y = Y.reshape(-1, order="C")

        assert Y.size == Lf.cov.shape[1]

        if b is not None:
            pred_mean = pred_mean + np.asarray(b.mean).reshape(-1, order="C")
            gram = gram + pn.linops.aslinop(b.cov)

        gram.is_symmetric = True
        gram.is_positive_definite = True

        return Y, L, b, kLa, pred_mean, gram


pn.randprocs.GaussianProcess.condition_on_observations = (
    lambda *args, **kwargs: ConditionalGaussianProcess.from_observations(
        *args, **kwargs
    )
)


@LinearFunctionOperator.__call__.register(  # pylint: disable=no-member
    ConditionalGaussianProcess._PriorPredictiveCrossCovariance
)
def _(
    self, crosscov: ConditionalGaussianProcess._PriorPredictiveCrossCovariance, /
) -> ConditionalGaussianProcess._PriorPredictiveCrossCovariance:
    return ConditionalGaussianProcess._PriorPredictiveCrossCovariance(
        (self(kLa) for kLa in crosscov)
    )


@LinearFunctional.__call__.register(  # pylint: disable=no-member
    ConditionalGaussianProcess._PriorPredictiveCrossCovariance
)
def _(
    self, crosscov: ConditionalGaussianProcess._PriorPredictiveCrossCovariance, /
) -> pn.linops.LinearOperator:
    return BlockMatrix([self(kLa_prev) for kLa_prev in crosscov])


@LinearFunctionOperator.__call__.register(  # pylint: disable=no-member
    ConditionalGaussianProcess
)
def _(
    self, conditional_gp: ConditionalGaussianProcess, /
) -> ConditionalGaussianProcess:
    # pylint: disable=protected-access

    linop_prior = self(conditional_gp._prior)

    return ConditionalGaussianProcess(
        prior=linop_prior,
        Ys=conditional_gp._Ys,
        Ls=conditional_gp._Ls,
        bs=conditional_gp._bs,
        kLas=self(conditional_gp._kLas),
        gram_matrix=conditional_gp.gram,
        representer_weights=conditional_gp.representer_weights,
    )


@LinearFunctional.__call__.register(  # pylint: disable=no-member
    ConditionalGaussianProcess
)
def _(
    self, conditional_gp: ConditionalGaussianProcess, /
) -> ConditionalGaussianProcess:
    # pylint: disable=protected-access

    linfunctl_prior = self(conditional_gp._prior)
    crosscov = self(conditional_gp._kLas)

    mean = linfunctl_prior.mean + crosscov @ conditional_gp.representer_weights
    cov = linfunctl_prior.cov - crosscov @ conditional_gp.gram.solve(crosscov.T)

    return pn.randvars.Normal(mean, cov)


def cho_solve(L, b):
    """Fixes a bug in scipy.linalg.cho_solve"""
    (L, lower) = L

    if L.shape == (1, 1) and b.shape[0] == 1:
        return b / L[0, 0] ** 2

    return scipy.linalg.cho_solve((L, lower), b)
