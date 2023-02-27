import numpy as np
import probnum as pn
from linpde_gp import linfunctls

from .._pv_crosscov import ProcessVectorCrossCovariance
from .._reshaped import ReshapedProcessVectorCrossCovariance


@linfunctls.ReshapedLinearFunctional.__call__.register(  # pylint: disable=no-member
    ProcessVectorCrossCovariance
)
def _(self, pv_crosscov: ProcessVectorCrossCovariance, /) -> np.ndarray:
    inner_result = self.linfctl(pv_crosscov)
    inner_output_ndim = self.linfctl.output_ndim

    inner_shape = inner_result.shape
    # Take the existing shape and flatten the linear functional component
    if not pv_crosscov.reverse:
        # k(x, .)
        target_shape = self.output_shape + inner_shape[inner_output_ndim:]
    else:
        # k(., x)
        target_shape = inner_shape[:-inner_output_ndim] + self.output_shape
    return inner_result.reshape(target_shape, order=self.order)


class CovarianceFunction_Identity_Reshaped(ReshapedProcessVectorCrossCovariance):
    def __init__(
        self,
        covfunc: pn.randprocs.covfuncs.CovarianceFunction,
        reshaped: linfunctls.ReshapedLinearFunctional,
    ):
        self._covfunc = covfunc
        self._reshaped = reshaped
        inner_pv_crosscov = self.reshaped.linfctl(self._covfunc, argnum=1)
        super().__init__(inner_pv_crosscov, order=reshaped.order)

    @property
    def covfunc(self) -> pn.randprocs.covfuncs.CovarianceFunction:
        return self._covfunc

    @property
    def reshaped(self) -> linfunctls.ReshapedLinearFunctional:
        return self._reshaped

class CovarianceFunction_Reshaped_Identity(ReshapedProcessVectorCrossCovariance):
    def __init__(
        self,
        covfunc: pn.randprocs.covfuncs.CovarianceFunction,
        reshaped: linfunctls.ReshapedLinearFunctional,
    ):
        self._covfunc = covfunc
        self._reshaped = reshaped
        inner_pv_crosscov = self.reshaped.linfctl(self._covfunc, argnum=0)
        super().__init__(inner_pv_crosscov, order=reshaped.order)

    @property
    def covfunc(self) -> pn.randprocs.covfuncs.CovarianceFunction:
        return self._covfunc

    @property
    def reshaped(self) -> linfunctls.ReshapedLinearFunctional:
        return self._reshaped