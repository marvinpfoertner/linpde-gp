from linpde_gp.linfunctls import CompositeLinearFunctional
from linpde_gp.randvars import ArrayCovariance, Covariance

from .._pv_crosscov import ProcessVectorCrossCovariance


@CompositeLinearFunctional.__call__.register(  # pylint: disable=no-member
    ProcessVectorCrossCovariance
)
def _(self, pv_crosscov: ProcessVectorCrossCovariance, /) -> Covariance:
    res = pv_crosscov

    if self.linfuncop is not None:
        res = self.linfuncop(res)

    res = self.linfunctl(res)
    assert isinstance(res, Covariance)

    if self.linop is not None:
        # NOTE: Consider adding option to do this matrix-free in the future?
        axis = res.ndim0 + res.ndim1 - 1 if pv_crosscov.reverse else res.ndim0 - 1
        array_res = self.linop(res.array, axis=axis)
        shape0 = array_res.shape[: res.ndim0]
        shape1 = array_res.shape[res.ndim0 :]
        res = ArrayCovariance(array_res, shape0, shape1)

    return res
