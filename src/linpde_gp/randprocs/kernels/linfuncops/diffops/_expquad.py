import functools

from jax import numpy as jnp
import numpy as np

from linpde_gp.linfuncops import diffops

from ... import _jax
from ..._expquad import ExpQuad


class ExpQuad_Identity_DirectionalDerivative(_jax.JaxKernel):
    def __init__(
        self,
        expquad: ExpQuad,
        direction: np.ndarray,
        reverse: bool = False,
    ):
        self._expquad = expquad

        super().__init__(self._expquad.input_shape, output_shape=())

        self._direction = direction

        self._reverse = reverse

    @property
    def expquad(self) -> ExpQuad:
        return self._expquad

    @property
    def direction(self) -> np.ndarray:
        return self._direction

    @property
    def reverse(self) -> bool:
        return self._reverse

    @functools.cached_property
    def _rescaled_direction(self) -> np.ndarray:
        rescaled_dir = self._direction / self._expquad.lengthscales**2

        return -rescaled_dir if self._reverse else rescaled_dir

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        if x1 is None:
            return np.zeros_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        diffs = x0 - x1

        proj_diffs = self._batched_sum(self._rescaled_direction * diffs)
        dists_sq = self._batched_euclidean_norm_sq(diffs / self._expquad.lengthscales)

        return proj_diffs * np.exp(-0.5 * dists_sq)

    def _evaluate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        if x1 is None:
            return jnp.zeros_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        diffs = x0 - x1

        proj_diffs = self._batched_sum_jax(self._rescaled_direction * diffs)
        dists_sq = self._batched_euclidean_norm_sq_jax(
            diffs / self._expquad.lengthscales
        )

        return proj_diffs * jnp.exp(-0.5 * dists_sq)


class ExpQuad_DirectionalDerivative_DirectionalDerivative(_jax.JaxKernel):
    def __init__(
        self,
        expquad: ExpQuad,
        direction0: np.ndarray,
        direction1: np.ndarray,
    ):
        self._expquad = expquad

        super().__init__(self._expquad.input_shape, output_shape=())

        self._direction0 = direction0
        self._direction1 = direction1

    @property
    def expquad(self) -> ExpQuad:
        return self._expquad

    @functools.cached_property
    def _rescaled_direction0(self) -> np.ndarray:
        return self._direction0 / self._expquad.lengthscales**2

    @functools.cached_property
    def _rescaled_direction1(self) -> np.ndarray:
        return self._direction1 / self._expquad.lengthscales**2

    @functools.cached_property
    def _directions_inprod(self) -> np.ndarray:
        return self._batched_sum(self._direction0 * self._rescaled_direction1)

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        if x1 is None:
            return np.full_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                self._directions_inprod,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        diffs = x0 - x1

        proj_diffs0 = self._batched_sum(self._rescaled_direction0 * diffs)
        proj_diffs1 = self._batched_sum(self._rescaled_direction1 * diffs)
        dists_sq = self._batched_euclidean_norm_sq(diffs / self._expquad.lengthscales)

        return (self._directions_inprod - proj_diffs0 * proj_diffs1) * np.exp(
            -0.5 * dists_sq
        )

    def _evaluate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        if x1 is None:
            return jnp.full_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                self._directions_inprod,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        diffs = x0 - x1

        proj_diffs0 = self._batched_sum_jax(self._rescaled_direction0 * diffs)
        proj_diffs1 = self._batched_sum_jax(self._rescaled_direction1 * diffs)
        dists_sq = self._batched_euclidean_norm_sq_jax(
            diffs / self._expquad.lengthscales
        )

        return (self._directions_inprod - proj_diffs0 * proj_diffs1) * jnp.exp(
            -0.5 * dists_sq
        )


class ExpQuad_Identity_WeightedLaplacian(_jax.JaxKernel):
    def __init__(
        self,
        expquad: ExpQuad,
        L: diffops.WeightedLaplacian,
        reverse: bool = True,
    ):
        self._expquad = expquad
        self._L = L

        super().__init__(self._expquad.input_shape, output_shape=())

        self._reverse = bool(reverse)

    @property
    def expquad(self) -> ExpQuad:
        return self._expquad

    @property
    def reverse(self) -> bool:
        return self._reverse

    @functools.cached_property
    def _weighted_inv_lengthscales_sq(self) -> np.ndarray:
        return 2.0 * self._L.weights * self._expquad._scale_factors**2

    @functools.cached_property
    def _scale_factors_sq(self) -> np.ndarray:
        return (
            2.0 * self._weighted_inv_lengthscales_sq * self._expquad._scale_factors**2
        )

    @functools.cached_property
    def _trace_term(self):
        return np.sum(2.0 * self._L.weights * self._expquad._scale_factors**2)

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        if x1 is None:
            return np.full_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                -self._trace_term,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        diffs = x0 - x1

        return (
            self._batched_sum(self._scale_factors_sq * diffs * diffs) - self._trace_term
        ) * np.exp(
            -self._batched_euclidean_norm_sq(self._expquad._scale_factors * diffs)
        )

    def _evaluate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        if x1 is None:
            return jnp.full_like(
                x0,
                -self._trace_term,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        diffs = x0 - x1

        return (
            self._batched_sum_jax(self._scale_factors_sq * diffs * diffs)
            - self._trace_term
        ) * jnp.exp(
            -self._batched_euclidean_norm_sq_jax(self._expquad._scale_factors * diffs)
        )


class ExpQuad_WeightedLaplacian_WeightedLaplacian(_jax.JaxKernel):
    def __init__(
        self,
        expquad: ExpQuad,
        L0: diffops.WeightedLaplacian,
        L1: diffops.WeightedLaplacian,
    ):
        super().__init__(expquad.input_shape, output_shape=())

        self._expquad = expquad
        self._L0 = L0
        self._L1 = L1

        self._expquad_laplacian_0 = ExpQuad_Identity_WeightedLaplacian(
            self._expquad,
            L=self._L0,
            reverse=True,
        )
        self._expquad_laplacian_1 = ExpQuad_Identity_WeightedLaplacian(
            self._expquad,
            L=self._L1,
            reverse=False,
        )

    @property
    def expquad(self) -> ExpQuad:
        return self._expquad

    @functools.cached_property
    def _scale_factors_sq(self) -> np.ndarray:
        return (
            self._expquad_laplacian_0._scale_factors_sq
            * self._expquad_laplacian_1._weighted_inv_lengthscales_sq
        )

    @functools.cached_property
    def _trace_term(self):
        return np.sum(
            self._expquad_laplacian_0._weighted_inv_lengthscales_sq
            * self._expquad_laplacian_1._weighted_inv_lengthscales_sq
        )

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        if x1 is None:
            return np.full_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                (
                    self._expquad_laplacian_0._trace_term
                    * self._expquad_laplacian_1._trace_term
                    + 2 * self._trace_term
                ),
                shape=x0.shape[: x0.ndim - self._input_ndim],
            )

        diffs = x0 - x1

        return (
            (
                self._batched_sum(
                    self._expquad_laplacian_0._scale_factors_sq * diffs**2
                )
                - self._expquad_laplacian_0._trace_term
            )
            * (
                self._batched_sum(
                    self._expquad_laplacian_1._scale_factors_sq * diffs**2
                )
                - self._expquad_laplacian_1._trace_term
            )
            - 4 * self._batched_sum(self._scale_factors_sq * diffs**2)
            + 2 * self._trace_term
        ) * np.exp(
            -self._batched_euclidean_norm_sq(self._expquad._scale_factors * diffs)
        )

    def _evaluate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        if x1 is None:
            return jnp.full_like(
                x0,
                (
                    self._expquad_laplacian_0._trace_term
                    * self._expquad_laplacian_1._trace_term
                    + 2 * self._trace_term
                ),
                shape=x0.shape[: x0.ndim - self._input_ndim],
            )

        diffs = x0 - x1

        return (
            (
                self._batched_sum_jax(
                    self._expquad_laplacian_0._scale_factors_sq * diffs**2
                )
                - self._expquad_laplacian_0._trace_term
            )
            * (
                self._batched_sum_jax(
                    self._expquad_laplacian_1._scale_factors_sq * diffs**2
                )
                - self._expquad_laplacian_1._trace_term
            )
            - 4 * self._batched_sum_jax(self._scale_factors_sq * diffs**2)
            + 2 * self._trace_term
        ) * jnp.exp(
            -self._batched_euclidean_norm_sq_jax(self._expquad._scale_factors * diffs)
        )


class ExpQuad_DirectionalDerivative_WeightedLaplacian(_jax.JaxKernel):
    def __init__(
        self,
        expquad: ExpQuad,
        direction: np.ndarray,
        L1: diffops.WeightedLaplacian,
        reverse: bool = False,
    ):
        self._expquad = expquad

        super().__init__(self._expquad.input_shape, output_shape=())

        self._direction = direction
        self._L1 = L1

        self._reverse = bool(reverse)

        self._expquad_laplacian = ExpQuad_Identity_WeightedLaplacian(
            self._expquad,
            L=self._L1,
            reverse=self._reverse,
        )

    @property
    def expquad(self) -> ExpQuad:
        return self._expquad

    @functools.cached_property
    def _rescaled_direction(self) -> np.ndarray:
        return self._direction * 2.0 * self._expquad._scale_factors**2

    @functools.cached_property
    def _rescaled_weighted_direction(self) -> np.ndarray:
        return (
            self._rescaled_direction
            * self._expquad_laplacian._weighted_inv_lengthscales_sq
        )

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        if x1 is None:
            return np.zeros_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        diffs = x0 - x1

        proj_diffs = self._batched_sum(self._rescaled_direction * diffs)
        proj_weighted_diffs = self._batched_sum(
            self._rescaled_weighted_direction * diffs
        )

        k_x0_x1 = 2 * proj_weighted_diffs * self._expquad(x0, x1)
        k_x0_x1 -= proj_diffs * self._expquad_laplacian(x0, x1)

        if self._reverse:
            return -k_x0_x1

        return k_x0_x1

    def _evaluate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        if x1 is None:
            return jnp.zeros_like(
                x0,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        diffs = x0 - x1

        proj_diffs = self._batched_sum_jax(self._rescaled_direction * diffs)
        proj_weighted_diffs = self._batched_sum_jax(
            self._rescaled_weighted_direction * diffs
        )

        k_x0_x1 = 2 * proj_weighted_diffs * self._expquad.jax(x0, x1)
        k_x0_x1 -= proj_diffs * self._expquad_laplacian.jax(x0, x1)

        if self._reverse:
            return -k_x0_x1

        return k_x0_x1
