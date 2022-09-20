import functools

import numpy as np
import probnum as pn
import scipy.sparse

from linpde_gp.functions import bases

from .._linfunctl import LinearFunctional


class WeakForm_Laplacian_UnivariateInterpolationBasis(LinearFunctional):
    def __init__(self, test_basis: bases.UnivariateLinearInterpolationBasis):
        assert test_basis.zero_boundary

        self._test_basis = test_basis

        super().__init__(
            input_shapes=((), ()),
            output_shape=test_basis.output_shape,
        )

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)

    @__call__.register(bases.UnivariateLinearInterpolationBasis)
    def _(
        self, trial_basis: bases.UnivariateLinearInterpolationBasis, /
    ) -> pn.linops.LinearOperator:
        if trial_basis.zero_boundary:
            raise NotImplementedError()

        if len(trial_basis) == len(self._test_basis) + 2 and np.all(
            trial_basis.grid[1:-1] == self._test_basis.grid
        ):
            grid = trial_basis.grid

            dist_normalizer = 1 / (grid[1:] - grid[:-1])

            diag = -dist_normalizer[:-1] - dist_normalizer[1:]

            return pn.linops.Matrix(
                scipy.sparse.diags(
                    (dist_normalizer[:-1], diag, dist_normalizer[1:]),
                    offsets=(0, 1, 2),
                    format="csr",
                    shape=(len(self._test_basis), len(trial_basis)),
                )
            )

        return super().__call__(trial_basis)
