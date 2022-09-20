from linpde_gp.linfunctls import CompositeLinearFunctional

from .._pv_crosscov import ProcessVectorCrossCovariance


@CompositeLinearFunctional.__call__.register(  # pylint: disable=no-member
    ProcessVectorCrossCovariance
)
def _(self, pv_crosscov: ProcessVectorCrossCovariance, /):
    res = pv_crosscov

    if self.linfuncop is not None:
        res = self.linfuncop(res)

    res = self.linfunctl(res)

    if self.linop is not None:
        res = self.linop(res, axis=-1 if pv_crosscov.reverse else 0)

    return res
