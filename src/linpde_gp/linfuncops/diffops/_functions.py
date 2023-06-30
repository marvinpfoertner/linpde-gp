from linpde_gp import functions

from ._derivative import Derivative
from ._directional_derivative import DirectionalDerivative
from ._laplacian import Laplacian, SpatialLaplacian
from ._partial_derivative import PartialDerivative


@DirectionalDerivative.__call__.register  # pylint: disable=no-member
@Laplacian.__call__.register  # pylint: disable=no-member
@PartialDerivative.__call__.register  # pylint: disable=no-member
@SpatialLaplacian.__call__.register  # pylint: disable=no-member
def _(self, f: functions.Constant, /) -> functions.Zero:
    assert f.input_shape == self.input_domain_shape
    assert f.output_shape == self.input_codomain_shape

    return functions.Zero(
        input_shape=self.output_domain_shape,
        output_shape=self.output_codomain_shape,
    )
