from linpde_gp.linfuncops import diffops

from ... import _tensor_product
from ._tensor_product import (
    TensorProductKernel_DirectionalDerivative_DirectionalDerivative,
    TensorProductKernel_DirectionalDerivative_WeightedLaplacian,
    TensorProductKernel_Identity_DirectionalDerivative,
    TensorProductKernel_Identity_WeightedLaplacian,
    TensorProductKernel_WeightedLaplacian_WeightedLaplacian,
)

########################################################################################
# Directional Derivative ###############################################################
########################################################################################


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: _tensor_product.TensorProductKernel, /, *, argnum: int = 0):
    return TensorProductKernel_Identity_DirectionalDerivative(
        k, self, reverse=(argnum == 0)
    )


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(
    self, k: TensorProductKernel_Identity_DirectionalDerivative, /, *, argnum: int = 0
):
    if argnum == 0 and not k.reverse:
        return TensorProductKernel_DirectionalDerivative_DirectionalDerivative(
            k.k, L0=self, L1=k.L
        )

    if argnum == 1 and k.reverse:
        return TensorProductKernel_DirectionalDerivative_DirectionalDerivative(
            k.k, L0=k.L, L1=self
        )

    return super(diffops.DirectionalDerivative, self).__call__(k, argnum=argnum)


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: TensorProductKernel_Identity_WeightedLaplacian, /, *, argnum: int = 0):
    if (argnum == 0 and not k.reverse) or (argnum == 1 and k.reverse):
        return TensorProductKernel_DirectionalDerivative_WeightedLaplacian(
            k.k,
            dderiv=self,
            laplacian=k.L,
            reverse=k.reverse,
        )

    return super(diffops.DirectionalDerivative, self).__call__(k, argnum=argnum)


########################################################################################
# (Weighted) Laplacian #################################################################
########################################################################################


@diffops.WeightedLaplacian.__call__.register  # pylint: disable=no-member
def _(self, k: _tensor_product.TensorProductKernel, /, *, argnum: int = 0):
    return TensorProductKernel_Identity_WeightedLaplacian(
        k, self, reverse=(argnum == 0)
    )


@diffops.WeightedLaplacian.__call__.register  # pylint: disable=no-member
def _(self, k: TensorProductKernel_Identity_WeightedLaplacian, /, *, argnum: int = 0):
    if argnum == 0 and not k.reverse:
        return TensorProductKernel_WeightedLaplacian_WeightedLaplacian(
            k.k,
            L0=self,
            L1=k.L,
        )

    if argnum == 1 and k.reverse:
        return TensorProductKernel_WeightedLaplacian_WeightedLaplacian(
            k.k,
            L0=k.L,
            L1=self,
        )

    return super(diffops.Laplacian, self).__call__(k, argnum=argnum)


@diffops.WeightedLaplacian.__call__.register  # pylint: disable=no-member
def _(
    self, k: TensorProductKernel_Identity_DirectionalDerivative, /, *, argnum: int = 0
):
    if (argnum == 0 and not k.reverse) or (argnum == 1 and k.reverse):
        return TensorProductKernel_DirectionalDerivative_WeightedLaplacian(
            k.k,
            dderiv=k.L,
            laplacian=self,
            reverse=not k.reverse,
        )

    return super(diffops.Laplacian, self).__call__(k, argnum=argnum)
