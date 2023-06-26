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


########################################################################################
# Partial Derivative ###################################################################
########################################################################################


@diffops.Derivative.__call__.register  # pylint: disable=no-member
def _(self, k: pn_covfuncs.Matern, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    if self.order == 0:
        return k

    if k.p is not None:
        if argnum == 0:
            L0 = self
            L1 = diffops.Derivative(0)
        else:
            assert argnum == 1

            L0 = diffops.Derivative(0)
            L1 = self

        return _matern.UnivariateHalfIntegerMatern_Derivative_Derivative(k, L0, L1)

    return super(diffops.Derivative, self).__call__(k, argnum=argnum)


@diffops.Derivative.__call__.register  # pylint: disable=no-member
def _(
    self,
    L0kL1: _matern.UnivariateHalfIntegerMatern_Derivative_Derivative,
    /,
    *,
    argnum: int = 0,
):
    validate_covfunc_transformation(self, L0kL1, argnum)

    assert L0kL1.covfunc.p is not None

    if self.order == 0:
        return L0kL1

    L0 = L0kL1.L0
    L1 = L0kL1.L1

    if argnum == 0:
        L0 = diffops.Derivative(self.order + L0.order)
    else:
        assert argnum == 1

        L1 = diffops.Derivative(self.order + L1.order)

    return _matern.UnivariateHalfIntegerMatern_Derivative_Derivative(
        L0kL1.covfunc, L0, L1
    )


########################################################################################
# Directional Derivative ###############################################################
########################################################################################


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: covfuncs.TensorProduct, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    return _tensor_product.TensorProduct_Identity_DirectionalDerivative(
        k, self, reverse=argnum == 0
    )


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(
    self,
    k: _tensor_product.TensorProduct_Identity_DirectionalDerivative,
    /,
    *,
    argnum: int = 0,
):
    validate_covfunc_transformation(self, k, argnum)

    if argnum == 0 and not k.reverse:
        return (
            _tensor_product.TensorProduct_DirectionalDerivative_DirectionalDerivative(
                k.k, L0=self, L1=k.L
            )
        )

    if argnum == 1 and k.reverse:
        return (
            _tensor_product.TensorProduct_DirectionalDerivative_DirectionalDerivative(
                k.k, L0=k.L, L1=self
            )
        )

    return super(diffops.DirectionalDerivative, self).__call__(k, argnum=argnum)


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(
    self,
    k: _tensor_product.TensorProduct_Identity_WeightedLaplacian,
    /,
    *,
    argnum: int = 0,
):
    validate_covfunc_transformation(self, k, argnum)

    if (argnum == 0 and not k.reverse) or (argnum == 1 and k.reverse):
        return _tensor_product.TensorProduct_DirectionalDerivative_WeightedLaplacian(
            k.k,
            dderiv=self,
            laplacian=k.L,
            reverse=k.reverse,
        )

    return super(diffops.DirectionalDerivative, self).__call__(k, argnum=argnum)


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
def _(self, k: covfuncs.TensorProduct, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    return _tensor_product.TensorProduct_Identity_WeightedLaplacian(
        k, self, reverse=argnum == 0
    )


@diffops.WeightedLaplacian.__call__.register  # pylint: disable=no-member
def _(
    self,
    k: _tensor_product.TensorProduct_Identity_WeightedLaplacian,
    /,
    *,
    argnum: int = 0,
):
    validate_covfunc_transformation(self, k, argnum)

    if argnum == 0 and not k.reverse:
        return _tensor_product.TensorProduct_WeightedLaplacian_WeightedLaplacian(
            k.k,
            L0=self,
            L1=k.L,
        )

    if argnum == 1 and k.reverse:
        return _tensor_product.TensorProduct_WeightedLaplacian_WeightedLaplacian(
            k.k,
            L0=k.L,
            L1=self,
        )

    return super(diffops.WeightedLaplacian, self).__call__(k, argnum=argnum)


@diffops.WeightedLaplacian.__call__.register  # pylint: disable=no-member
def _(
    self,
    k: _tensor_product.TensorProduct_Identity_DirectionalDerivative,
    /,
    *,
    argnum: int = 0,
):
    validate_covfunc_transformation(self, k, argnum)

    if (argnum == 0 and not k.reverse) or (argnum == 1 and k.reverse):
        return _tensor_product.TensorProduct_DirectionalDerivative_WeightedLaplacian(
            k.k,
            dderiv=k.L,
            laplacian=self,
            reverse=not k.reverse,
        )

    return super(diffops.WeightedLaplacian, self).__call__(k, argnum=argnum)


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
