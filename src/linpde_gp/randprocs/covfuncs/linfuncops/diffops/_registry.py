import numpy as np
from probnum.randprocs import covfuncs as pn_covfuncs

from linpde_gp.linfuncops import diffops
from linpde_gp.randprocs import covfuncs

from . import _expquad, _matern, _tensor_product
from ..._utils import validate_covfunc_transformation

########################################################################################
# General `LinearFunctionOperators` ####################################################
########################################################################################


@diffops.LinearDifferentialOperator.__call__.register  # pylint: disable=no-member
def _(self, k: covfuncs.JaxCovarianceFunctionMixin, /, *, argnum=0):
    validate_covfunc_transformation(self, k, argnum)

    try:
        return super(diffops.LinearDifferentialOperator, self).__call__(
            k, argnum=argnum
        )
    except NotImplementedError:
        return covfuncs.JaxLambdaCovarianceFunction(
            self._jax_fallback(  # pylint: disable=protected-access
                k.jax, argnum=argnum
            ),
            input_shape=self.output_domain_shape,
            vectorize=True,
        )


@diffops.PartialDerivative.__call__.register  # pylint: disable=no-member
@diffops.LinearDifferentialOperator.__call__.register  # pylint: disable=no-member
def _(
    self,
    k: _tensor_product.TensorProduct,
    /,
    *,
    argnum: int = 0,
):
    validate_covfunc_transformation(self, k, argnum)

    D0 = (
        self
        if argnum == 0
        else diffops.PartialDerivative(diffops.MultiIndex(np.zeros(k.input_shape_0)))
    )
    D1 = (
        self
        if argnum == 1
        else diffops.PartialDerivative(diffops.MultiIndex(np.zeros(k.input_shape_1)))
    )
    return _tensor_product.TensorProduct_LinDiffop_LinDiffop(k, L0=D0, L1=D1)


@diffops.PartialDerivative.__call__.register  # pylint: disable=no-member
@diffops.LinearDifferentialOperator.__call__.register  # pylint: disable=no-member
def _(
    self,
    k: _tensor_product.TensorProduct_LinDiffop_LinDiffop,
    /,
    *,
    argnum: int = 0,
):
    validate_covfunc_transformation(self, k, argnum)

    if argnum == 0 and k.L0.order == 0:
        D0 = self
        D1 = k.L1
    elif argnum == 1 and k.L1.order == 0:
        D0 = k.L0
        D1 = self
    else:
        return NotImplemented
    return _tensor_product.TensorProduct_LinDiffop_LinDiffop(k.k, L0=D0, L1=D1)


########################################################################################
# Jax Partial Derivative ###############################################################
########################################################################################


@diffops.JaxPartialDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: covfuncs.JaxCovarianceFunction, /, *, argnum=0):
    validate_covfunc_transformation(self, k, argnum)

    return covfuncs.JaxLambdaCovarianceFunction(
        self._derive(k.jax, argnum=argnum),  # pylint: disable=protected-access
        input_shape=self.output_domain_shape,
        vectorize=True,
    )


########################################################################################
# Partial Derivative ###################################################################
########################################################################################


def _partial_derivative_fallback(
    D: diffops.PartialDerivative, k: pn_covfuncs.CovarianceFunction, argnum: int = 0
):
    validate_covfunc_transformation(D, k, argnum)

    if D.order == 0:
        return k
    if int(np.prod(D.input_domain_shape)) == 1:
        if D.order == 1:
            return diffops.DirectionalDerivative(1.0)(k, argnum=argnum)
        if D.order == 2:
            return diffops.WeightedLaplacian(1.0)(k, argnum=argnum)
    return NotImplemented


@diffops.PartialDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: pn_covfuncs.Matern, /, *, argnum: int = 0):
    return _partial_derivative_fallback(self, k, argnum=argnum)


@diffops.PartialDerivative.__call__.register  # pylint: disable=no-member
def _(
    self,
    k: _matern.HalfIntegerMatern_Identity_DirectionalDerivative,
    /,
    *,
    argnum: int = 0,
):
    return _partial_derivative_fallback(self, k, argnum=argnum)


@diffops.PartialDerivative.__call__.register  # pylint: disable=no-member
def _(
    self,
    k: _matern.UnivariateHalfIntegerMatern_Identity_WeightedLaplacian,
    /,
    *,
    argnum: int = 0,
):
    return _partial_derivative_fallback(self, k, argnum=argnum)


@diffops.PartialDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: pn_covfuncs.ExpQuad, /, *, argnum: int = 0):
    return _partial_derivative_fallback(self, k, argnum=argnum)


@diffops.PartialDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: _expquad.ExpQuad_Identity_DirectionalDerivative, /, *, argnum: int = 0):
    return _partial_derivative_fallback(self, k, argnum=argnum)


@diffops.PartialDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: _expquad.ExpQuad_Identity_WeightedLaplacian, /, *, argnum: int = 0):
    return _partial_derivative_fallback(self, k, argnum=argnum)


########################################################################################
# Directional Derivative ###############################################################
########################################################################################


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: pn_covfuncs.Matern, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    if k.p is not None:
        return _matern.HalfIntegerMatern_Identity_DirectionalDerivative(
            k,
            direction=self.direction,
            reverse=(argnum == 0),
        )

    return super(diffops.DirectionalDerivative, self).__init__(k, argnum=argnum)


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(
    self,
    k: _matern.HalfIntegerMatern_Identity_DirectionalDerivative,
    /,
    *,
    argnum: int = 0,
):
    validate_covfunc_transformation(self, k, argnum)

    assert k.matern.p is not None

    if argnum == 0 and not k.reverse:
        return (
            _matern.UnivariateHalfIntegerMatern_DirectionalDerivative_DirectionalDerivative  # pylint: disable=line-too-long
            if k.matern.input_size == 1
            else _matern.HalfIntegerMatern_DirectionalDerivative_DirectionalDerivative
        )(
            k.matern,
            direction0=self.direction,
            direction1=k.direction,
        )

    if argnum == 1 and k.reverse:
        return (
            _matern.UnivariateHalfIntegerMatern_DirectionalDerivative_DirectionalDerivative  # pylint: disable=line-too-long
            if k.matern.input_size == 1
            else _matern.HalfIntegerMatern_DirectionalDerivative_DirectionalDerivative
        )(
            k.matern,
            direction0=k.direction,
            direction1=self.direction,
        )

    return super(diffops.DirectionalDerivative, self).__call__(k, argnum=argnum)


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(
    self,
    k: _matern.UnivariateHalfIntegerMatern_Identity_WeightedLaplacian,
    /,
    *,
    argnum: int = 0,
):
    validate_covfunc_transformation(self, k, argnum)

    assert k.matern.p is not None

    if (argnum == 0 and not k.reverse) or (argnum == 1 and k.reverse):
        return (
            _matern.UnivariateHalfIntegerMatern_DirectionalDerivative_WeightedLaplacian(
                k.matern,
                direction=self.direction,
                L1=k.L,
                reverse=(argnum == 1),
            )
        )

    return super(diffops.DirectionalDerivative, self).__call__(k, argnum=argnum)


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: pn_covfuncs.ExpQuad, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    return _expquad.ExpQuad_Identity_DirectionalDerivative(
        expquad=k,
        direction=self.direction,
        reverse=(argnum == 0),
    )


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: _expquad.ExpQuad_Identity_DirectionalDerivative, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    if argnum == 0 and not k.reverse:
        return _expquad.ExpQuad_DirectionalDerivative_DirectionalDerivative(
            expquad=k.expquad,
            direction0=self.direction,
            direction1=k.direction,
        )

    if argnum == 1 and k.reverse:
        return _expquad.ExpQuad_DirectionalDerivative_DirectionalDerivative(
            expquad=k.expquad,
            direction0=k.direction,
            direction1=self.direction,
        )

    return super(diffops.DirectionalDerivative, self).__call__(k, argnum=argnum)


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: _expquad.ExpQuad_Identity_WeightedLaplacian, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    if (argnum == 0 and not k.reverse) or (argnum == 1 and k.reverse):
        return _expquad.ExpQuad_DirectionalDerivative_WeightedLaplacian(
            k.expquad,
            direction=self.direction,
            L1=k.L,
            reverse=(argnum == 1),
        )

    return super(diffops.DirectionalDerivative, self).__call__(k, argnum=argnum)


########################################################################################
# (Weighted) Laplacian #################################################################
########################################################################################


@diffops.WeightedLaplacian.__call__.register  # pylint: disable=no-member
def _(self, k: pn_covfuncs.Matern, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    if k.input_size == 1:
        if k.p is not None:
            return _matern.UnivariateHalfIntegerMatern_Identity_WeightedLaplacian(
                k, L=self, reverse=(argnum == 0)
            )

    return super(diffops.WeightedLaplacian, self).__call__(k, argnum=argnum)


@diffops.WeightedLaplacian.__call__.register  # pylint: disable=no-member
def _(
    self,
    k: _matern.UnivariateHalfIntegerMatern_Identity_WeightedLaplacian,
    /,
    *,
    argnum: int = 0,
):
    validate_covfunc_transformation(self, k, argnum)

    assert k.matern.p is not None
    assert k.input_size == 1

    if argnum == 0 and not k.reverse:
        return _matern.UnivariateHalfIntegerMatern_WeightedLaplacian_WeightedLaplacian(
            k.matern, L0=self, L1=k.L
        )

    if argnum == 1 and k.reverse:
        return _matern.UnivariateHalfIntegerMatern_WeightedLaplacian_WeightedLaplacian(
            k.matern, L0=k.L, L1=self
        )

    return super(diffops.WeightedLaplacian, self).__call__(k, argnum=argnum)


@diffops.WeightedLaplacian.__call__.register  # pylint: disable=no-member
def _(
    self,
    k: _matern.HalfIntegerMatern_Identity_DirectionalDerivative,
    /,
    *,
    argnum: int = 0,
):
    validate_covfunc_transformation(self, k, argnum)

    assert k.matern.p is not None
    assert k.input_size == 1

    if k.input_size == 1:
        if (argnum == 0 and not k.reverse) or (argnum == 1 and k.reverse):
            return _matern.UnivariateHalfIntegerMatern_DirectionalDerivative_WeightedLaplacian(  # pylint: disable=line-too-long
                k.matern,
                direction=k.direction,
                L1=self,
                reverse=(argnum == 0),
            )

    return super(diffops.WeightedLaplacian, self).__call__(k, argnum=argnum)


@diffops.WeightedLaplacian.__call__.register  # pylint: disable=no-member
def _(self, k: pn_covfuncs.ExpQuad, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    return _expquad.ExpQuad_Identity_WeightedLaplacian(k, L=self, reverse=argnum == 0)


@diffops.WeightedLaplacian.__call__.register  # pylint: disable=no-member
def _(self, k: _expquad.ExpQuad_Identity_WeightedLaplacian, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    if argnum == 0 and not k.reverse:
        return _expquad.ExpQuad_WeightedLaplacian_WeightedLaplacian(
            k.expquad, L0=self, L1=k.L
        )

    if argnum == 1 and k.reverse:
        return _expquad.ExpQuad_WeightedLaplacian_WeightedLaplacian(
            k.matern, L0=k.L, L1=self
        )

    return super(diffops.WeightedLaplacian, self).__call__(k, argnum=argnum)


@diffops.WeightedLaplacian.__call__.register  # pylint: disable=no-member
def _(self, k: _expquad.ExpQuad_Identity_DirectionalDerivative, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    if (argnum == 0 and not k.reverse) or (argnum == 1 and k.reverse):
        return _expquad.ExpQuad_DirectionalDerivative_WeightedLaplacian(
            k.expquad,
            direction=k.direction,
            L1=self,
            reverse=(argnum == 0),
        )

    return super(diffops.WeightedLaplacian, self).__call__(k, argnum=argnum)
