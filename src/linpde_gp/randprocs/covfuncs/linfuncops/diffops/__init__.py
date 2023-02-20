from probnum.randprocs import covfuncs as _pn_covfuncs

from linpde_gp.linfuncops import diffops

from ... import _tensor_product
from ._expquad import (
    ExpQuad_DirectionalDerivative_DirectionalDerivative,
    ExpQuad_DirectionalDerivative_WeightedLaplacian,
    ExpQuad_Identity_DirectionalDerivative,
    ExpQuad_Identity_WeightedLaplacian,
    ExpQuad_WeightedLaplacian_WeightedLaplacian,
)
from ._matern import (
    HalfIntegerMatern_DirectionalDerivative_DirectionalDerivative,
    HalfIntegerMatern_Identity_DirectionalDerivative,
    UnivariateHalfIntegerMatern_DirectionalDerivative_DirectionalDerivative,
    UnivariateHalfIntegerMatern_DirectionalDerivative_WeightedLaplacian,
    UnivariateHalfIntegerMatern_Identity_WeightedLaplacian,
    UnivariateHalfIntegerMatern_WeightedLaplacian_WeightedLaplacian,
)
from ._tensor_product import (
    TensorProduct_DirectionalDerivative_DirectionalDerivative,
    TensorProduct_DirectionalDerivative_WeightedLaplacian,
    TensorProduct_Identity_DirectionalDerivative,
    TensorProduct_Identity_WeightedLaplacian,
    TensorProduct_WeightedLaplacian_WeightedLaplacian,
)

########################################################################################
# Directional Derivative ###############################################################
########################################################################################


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: _tensor_product.TensorProduct, /, *, argnum: int = 0):
    return TensorProduct_Identity_DirectionalDerivative(k, self, reverse=(argnum == 0))


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: TensorProduct_Identity_DirectionalDerivative, /, *, argnum: int = 0):
    if argnum == 0 and not k.reverse:
        return TensorProduct_DirectionalDerivative_DirectionalDerivative(
            k.k, L0=self, L1=k.L
        )

    if argnum == 1 and k.reverse:
        return TensorProduct_DirectionalDerivative_DirectionalDerivative(
            k.k, L0=k.L, L1=self
        )

    return super(diffops.DirectionalDerivative, self).__call__(k, argnum=argnum)


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: TensorProduct_Identity_WeightedLaplacian, /, *, argnum: int = 0):
    if (argnum == 0 and not k.reverse) or (argnum == 1 and k.reverse):
        return TensorProduct_DirectionalDerivative_WeightedLaplacian(
            k.k,
            dderiv=self,
            laplacian=k.L,
            reverse=k.reverse,
        )

    return super(diffops.DirectionalDerivative, self).__call__(k, argnum=argnum)


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: _pn_covfuncs.Matern, /, *, argnum: int = 0):
    if k.p is not None:
        return HalfIntegerMatern_Identity_DirectionalDerivative(
            k,
            direction=self.direction,
            reverse=(argnum == 0),
        )

    return super(diffops.DirectionalDerivative, self).__init__(k, argnum=argnum)


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: HalfIntegerMatern_Identity_DirectionalDerivative, /, *, argnum: int = 0):
    assert k.matern.p is not None

    if argnum == 0 and not k.reverse:
        return (
            UnivariateHalfIntegerMatern_DirectionalDerivative_DirectionalDerivative
            if k.matern.input_size == 1
            else HalfIntegerMatern_DirectionalDerivative_DirectionalDerivative
        )(
            k.matern,
            direction0=self.direction,
            direction1=k.direction,
        )

    if argnum == 1 and k.reverse:
        return (
            UnivariateHalfIntegerMatern_DirectionalDerivative_DirectionalDerivative
            if k.matern.input_size == 1
            else HalfIntegerMatern_DirectionalDerivative_DirectionalDerivative
        )(
            k.matern,
            direction0=k.direction,
            direction1=self.direction,
        )

    return super(diffops.DirectionalDerivative, self).__call__(k, argnum=argnum)


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(
    self,
    k: UnivariateHalfIntegerMatern_Identity_WeightedLaplacian,
    /,
    *,
    argnum: int = 0,
):
    assert k.matern.p is not None

    if (argnum == 0 and not k.reverse) or (argnum == 1 and k.reverse):
        return UnivariateHalfIntegerMatern_DirectionalDerivative_WeightedLaplacian(
            k.matern,
            direction=self.direction,
            L1=k._L,
            reverse=(argnum == 1),
        )

    return super(diffops.DirectionalDerivative, self).__call__(k, argnum=argnum)


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: _pn_covfuncs.ExpQuad, /, *, argnum: int = 0):
    return ExpQuad_Identity_DirectionalDerivative(
        expquad=k,
        direction=self.direction,
        reverse=(argnum == 0),
    )


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: ExpQuad_Identity_DirectionalDerivative, /, *, argnum: int = 0):
    if argnum == 0 and not k.reverse:
        return ExpQuad_DirectionalDerivative_DirectionalDerivative(
            expquad=k.expquad,
            direction0=self.direction,
            direction1=k.direction,
        )

    if argnum == 1 and k.reverse:
        return ExpQuad_DirectionalDerivative_DirectionalDerivative(
            expquad=k.expquad,
            direction0=k.direction,
            direction1=self.direction,
        )

    return super(diffops.DirectionalDerivative, self).__call__(k, argnum=argnum)


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: ExpQuad_Identity_WeightedLaplacian, /, *, argnum: int = 0):
    if (argnum == 0 and not k.reverse) or (argnum == 1 and k.reverse):
        return ExpQuad_DirectionalDerivative_WeightedLaplacian(
            k.expquad,
            direction=self.direction,
            L1=k._L,
            reverse=(argnum == 1),
        )

    return super(diffops.DirectionalDerivative, self).__call__(k, argnum=argnum)


########################################################################################
# (Weighted) Laplacian #################################################################
########################################################################################


@diffops.WeightedLaplacian.__call__.register  # pylint: disable=no-member
def _(self, k: _tensor_product.TensorProduct, /, *, argnum: int = 0):
    return TensorProduct_Identity_WeightedLaplacian(k, self, reverse=(argnum == 0))


@diffops.WeightedLaplacian.__call__.register  # pylint: disable=no-member
def _(self, k: TensorProduct_Identity_WeightedLaplacian, /, *, argnum: int = 0):
    if argnum == 0 and not k.reverse:
        return TensorProduct_WeightedLaplacian_WeightedLaplacian(
            k.k,
            L0=self,
            L1=k.L,
        )

    if argnum == 1 and k.reverse:
        return TensorProduct_WeightedLaplacian_WeightedLaplacian(
            k.k,
            L0=k.L,
            L1=self,
        )

    return super(diffops.WeightedLaplacian, self).__call__(k, argnum=argnum)


@diffops.WeightedLaplacian.__call__.register  # pylint: disable=no-member
def _(self, k: TensorProduct_Identity_DirectionalDerivative, /, *, argnum: int = 0):
    if (argnum == 0 and not k.reverse) or (argnum == 1 and k.reverse):
        return TensorProduct_DirectionalDerivative_WeightedLaplacian(
            k.k,
            dderiv=k.L,
            laplacian=self,
            reverse=not k.reverse,
        )

    return super(diffops.WeightedLaplacian, self).__call__(k, argnum=argnum)


@diffops.WeightedLaplacian.__call__.register  # pylint: disable=no-member
def _(self, k: _pn_covfuncs.Matern, /, *, argnum: int = 0):
    if k.input_size == 1:

        if k.p is not None:
            return UnivariateHalfIntegerMatern_Identity_WeightedLaplacian(
                k, L=self, reverse=(argnum == 0)
            )

    return super(diffops.WeightedLaplacian, self).__call__(k, argnum=argnum)


@diffops.WeightedLaplacian.__call__.register  # pylint: disable=no-member
def _(
    self,
    k: UnivariateHalfIntegerMatern_Identity_WeightedLaplacian,
    /,
    *,
    argnum: int = 0,
):
    assert k.matern.p is not None
    assert k.input_size == 1

    if argnum == 0 and not k.reverse:
        return UnivariateHalfIntegerMatern_WeightedLaplacian_WeightedLaplacian(
            k.matern, L0=self, L1=k._L
        )

    if argnum == 1 and k.reverse:
        return UnivariateHalfIntegerMatern_WeightedLaplacian_WeightedLaplacian(
            k.matern, L0=k._L, L1=self
        )

    return super(diffops.WeightedLaplacian, self).__call__(k, argnum=argnum)


@diffops.WeightedLaplacian.__call__.register  # pylint: disable=no-member
def _(self, k: HalfIntegerMatern_Identity_DirectionalDerivative, /, *, argnum: int = 0):
    assert k.matern.p is not None
    assert k.input_size == 1

    if k.input_size == 1:
        if (argnum == 0 and not k.reverse) or (argnum == 1 and k.reverse):
            return UnivariateHalfIntegerMatern_DirectionalDerivative_WeightedLaplacian(
                k.matern,
                direction=k.direction,
                L1=self,
                reverse=(argnum == 0),
            )

    return super(diffops.WeightedLaplacian, self).__call__(k, argnum=argnum)


@diffops.WeightedLaplacian.__call__.register  # pylint: disable=no-member
def _(self, k: _pn_covfuncs.ExpQuad, /, *, argnum: int = 0):
    return ExpQuad_Identity_WeightedLaplacian(k, L=self, reverse=(argnum == 0))


@diffops.WeightedLaplacian.__call__.register  # pylint: disable=no-member
def _(self, k: ExpQuad_Identity_WeightedLaplacian, /, *, argnum: int = 0):
    if argnum == 0 and not k.reverse:
        return ExpQuad_WeightedLaplacian_WeightedLaplacian(k.expquad, L0=self, L1=k._L)

    if argnum == 1 and k.reverse:
        return ExpQuad_WeightedLaplacian_WeightedLaplacian(k.matern, L0=k._L, L1=self)

    return super(diffops.WeightedLaplacian, self).__call__(k, argnum=argnum)


@diffops.WeightedLaplacian.__call__.register  # pylint: disable=no-member
def _(self, k: ExpQuad_Identity_DirectionalDerivative, /, *, argnum: int = 0):
    if (argnum == 0 and not k.reverse) or (argnum == 1 and k.reverse):
        return ExpQuad_DirectionalDerivative_WeightedLaplacian(
            k.expquad,
            direction=k.direction,
            L1=self,
            reverse=(argnum == 0),
        )

    return super(diffops.WeightedLaplacian, self).__call__(k, argnum=argnum)
