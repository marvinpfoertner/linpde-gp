from __future__ import annotations

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

        # Compute representer weights
        gram_cho = scipy.linalg.cho_factor(gram)

        representer_weights = scipy.linalg.cho_solve(
            gram_cho,
            (Y - Lm).reshape((-1,), order="C"),
        )

        return cls(
            prior=prior,
            Ys=(Y,),
            Ls=(L,),
            bs=(b,),
            kLas=ConditionalGaussianProcess._PriorPredictiveCrossCovariance((kLa,)),
            gram_blocks=((gram,),),
            gram_cho=gram_cho,
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
        gram_blocks: Sequence[Sequence[np.ndarray]],
        gram_cho: tuple[np.ndarray, bool] | None = None,
        representer_weights: np.ndarray | None = None,
    ):
        self._prior = prior

        self._Ys = tuple(Ys)
        self._Ls = tuple(Ls)
        self._bs = tuple(bs)

        self._kLas = kLas

        self._gram_blocks = tuple(tuple(row) for row in gram_blocks)
        self._gram_cho = gram_cho

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
                gram_cho=self.gram_cho,
            ),
        )

    @functools.cached_property
    def gram(self) -> np.ndarray:
        return np.block(
            [
                [
                    self._gram_blocks[i][j] if i >= j else self._gram_blocks[j][i].T
                    for j in range(len(self._Ys))
                ]
                for i in range(len(self._Ys))
            ]
        )

    @property
    def gram_cho(self) -> tuple[np.ndarray, bool]:
        if self._gram_cho is None:
            self._gram_cho = scipy.linalg.cho_factor(self.gram)

        return self._gram_cho

    @property
    def representer_weights(self) -> np.ndarray:
        if self._representer_weights is None:
            self._representer_weights = scipy.linalg.cho_solve(
                self.gram_cho,
                np.concatenate(
                    [
                        np.reshape(
                            (
                                (Y - L(self._prior.mean))
                                if b is None
                                else (Y - L(self._prior.mean) - b.mean)
                            ),
                            (-1,),
                            order="C",
                        )
                        for Y, L, b in zip(self._Ys, self._Ls, self._bs)
                    ],
                    axis=-1,
                ),
            )

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
            gram_cho: np.ndarray,
        ):
            self._prior_covfunc = prior_covfunc
            self._kLas = kLas
            self._gram_cho = gram_cho

            super().__init__(
                input_shape=self._prior_covfunc.input_shape,
                output_shape_0=self._prior_covfunc.output_shape_0,
                output_shape_1=self._prior_covfunc.output_shape_1,
            )

        def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
            k_xx = self._prior_covfunc(x0, x1)
            kLas_x0 = self._kLas(x0)
            kLas_x1 = self._kLas(x1) if x1 is not None else kLas_x0

            return (
                k_xx
                - (
                    kLas_x0[..., None, :]
                    @ cho_solve(
                        self._gram_cho,
                        kLas_x1.transpose(),
                    ).transpose()[..., :, None]
                )[..., 0, 0]
            )

        @functools.partial(jax.jit, static_argnums=0)
        def _evaluate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
            k_xx = self._prior_covfunc.jax(x0, x1)
            kLas_x0 = self._kLas.jax(x0)
            kLas_x1 = self._kLas.jax(x1) if x1 is not None else kLas_x0

            return k_xx - kLas_x0 @ jax.scipy.linalg.cho_solve(self._gram_cho, kLas_x1)

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
        gram_L_La_prev_blocks = tuple(
            L(kLa_prev).reshape((L.output_size, kLa_prev.randvar_size))
            for kLa_prev in self._kLas
        )
        gram_L_row_blocks = gram_L_La_prev_blocks + (gram,)

        # Update the Cholesky decomposition of the previous covariance matrix and the
        # representer weights
        gram_cho, representer_weights = _block_cholesky(
            A_cho=self.gram_cho,
            B=np.concatenate(gram_L_La_prev_blocks, axis=-1).T,
            D=gram,
            sol_update=(
                self.representer_weights,
                (Y - pred_mean).reshape((-1,), order="C"),
            ),
        )

        return ConditionalGaussianProcess(
            prior=self._prior,
            Ys=self._Ys + (Y,),
            Ls=self._Ls + (L,),
            bs=self._bs + (b,),
            kLas=self._kLas.append(kLa),
            gram_blocks=self._gram_blocks + (gram_L_row_blocks,),
            gram_cho=gram_cho,
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
        np.ndarray,
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

                L = linfunctls.DiracFunctional(
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

        # Check observations
        Y = np.asarray(Y)

        if Y.shape != L.output_shape:
            raise ValueError(f"{Y.shape=} must be equal to {L.output_shape}.")

        # Compute the joint measure (f, L[f])
        Lf = L(prior)
        kLa = L(prior.cov, argnum=1)

        # Compute predictive mean and covariance matrix
        pred_mean = Lf.mean
        gram = np.atleast_2d(Lf.cov)

        if b is not None:
            pred_mean = pred_mean + b.mean
            gram = gram + b.cov

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
) -> ConditionalGaussianProcess._PriorPredictiveCrossCovariance:
    return np.concatenate(
        [np.atleast_1d(self(kLa)) for kLa in crosscov],
        axis=-1,
    )


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
        gram_blocks=conditional_gp._gram_blocks,
        gram_cho=conditional_gp.gram_cho,
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
    cov = linfunctl_prior.cov - crosscov @ scipy.linalg.cho_solve(
        conditional_gp.gram_cho, crosscov.T
    )

    return pn.randvars.Normal(mean, cov)


def cho_solve(L, b):
    """Fixes a bug in scipy.linalg.cho_solve"""
    (L, lower) = L

    if L.shape == (1, 1) and b.shape[0] == 1:
        return b / L[0, 0] ** 2

    return scipy.linalg.cho_solve((L, lower), b)


def _schur_update(A_cho, B, C, D, A_inv_u, v):
    """
    This function solves the linear system

    [[A, B], @ [[x], = [[u],
     [C, D]]    [y]]    [v]]

    given the Cholesky factor of A, the matrices B, C, and D, and the vectors A^{-1} u, and v.
    """
    A_inv_B = jax.scipy.linalg.cho_solve(A_cho, B)

    S = D - C @ A_inv_B

    y = jax.scipy.linalg.solve(S, v - C @ A_inv_u)
    x = A_inv_u - A_inv_B @ y

    return jnp.concatenate((x, y))


def _block_cholesky(
    A_cho: tuple[np.ndarray, bool],
    B: np.ndarray,
    D: np.ndarray,
    sol_update: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, bool]:
    A_sqrt, lower = A_cho

    tri = np.tril if lower else np.triu

    A_sqrt = tri(A_sqrt)

    L_A_inv_B = scipy.linalg.solve_triangular(
        A_sqrt,
        B,
        lower=lower,
        trans="N" if lower else "T",
    )

    # Compute the Schur complement of A in the block matrix
    S = D - L_A_inv_B.T @ L_A_inv_B
    S_sqrt = scipy.linalg.cholesky(S, lower=lower)
    S_cho = (S_sqrt, lower)

    # Assemble the block Cholesky factor
    block_sqrt = np.empty_like(
        A_sqrt,
        shape=(
            A_sqrt.shape[0] + D.shape[0],
            A_sqrt.shape[1] + D.shape[1],
        ),
    )

    A_i_end, A_j_end = A_sqrt.shape
    D_i_start, D_j_start = A_sqrt.shape

    # A-block
    block_sqrt[:A_i_end, :A_j_end] = A_sqrt

    # B-block
    block_sqrt[:A_i_end, D_j_start:] = 0 if lower else L_A_inv_B

    # C-block
    block_sqrt[D_i_start:, :A_j_end] = L_A_inv_B.T if lower else 0

    # D-block
    block_sqrt[D_i_start:, D_j_start:] = S_sqrt

    block_cho = (block_sqrt, lower)

    if sol_update is not None:
        C = B.T
        A_inv_u, v = sol_update

        y = scipy.linalg.cho_solve(S_cho, v - C @ A_inv_u)
        x = A_inv_u - scipy.linalg.solve_triangular(  # A^{-1} @ B @ y
            A_sqrt,
            L_A_inv_B @ y,
            lower=lower,
            trans="T" if lower else "N",
        )

        return block_cho, np.concatenate((x, y))

    return block_cho
