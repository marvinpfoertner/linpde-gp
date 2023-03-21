import functools

import numpy as np
import probnum as pn
from probnum.typing import ScalarLike, ScalarType

from ._lindiffop import LinearDifferentialOperator


class ScaledLinearDifferentialOperator(LinearDifferentialOperator):
    def __init__(
        self, lindiffop: LinearDifferentialOperator, /, scalar: ScalarLike
    ) -> None:
        self._lindiffop = lindiffop

        super().__init__(
            input_shapes=self._lindiffop.input_shapes,
            output_codomain_shape=self._lindiffop.output_codomain_shape,
        )

        if not np.ndim(scalar) == 0:
            raise ValueError()

        self._scalar = np.asarray(scalar, dtype=np.double)

    @property
    def lindiffop(self) -> LinearDifferentialOperator:
        return self._lindiffop

    @property
    def scalar(self) -> ScalarType:
        return self._scalar

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return self._scalar * self._lindiffop(f, **kwargs)

    def _jax_fallback(self, f, /, **kwargs):
        raise NotImplementedError()

    # TODO: Only need until GPs can be scaled
    @__call__.register
    def _(
        self, gp: pn.randprocs.GaussianProcess, /, **kwargs
    ) -> pn.randprocs.GaussianProcess:
        return super().__call__(gp, **kwargs)

    def __rmul__(self, other) -> LinearDifferentialOperator:
        if np.ndim(other) == 0:
            return ScaledLinearDifferentialOperator(
                lindiffop=self._lindiffop,
                scalar=np.asarray(other) * self._scalar,
            )

        return super().__rmul__(other)

    @functools.singledispatchmethod
    def weak_form(self, test_basis, /):
        return self._scalar * self._lindiffop.weak_form(test_basis)
