import numpy as np
import probnum as pn
from probnum.typing import ScalarLike, ScalarType

from ._covariance import Covariance


class ScaledCovariance(Covariance):
    def __init__(self, covariance: Covariance, scalar: ScalarLike) -> None:
        self._covariance = covariance
        self._scalar = scalar

        super().__init__(
            shape0=self._covariance.shape0,
            shape1=self._covariance.shape1,
        )

        if not np.ndim(scalar) == 0:
            raise ValueError()

        self._scalar = np.asarray(scalar, dtype=np.double)

    @property
    def covariance(self) -> Covariance:
        return self._covariance

    @property
    def scalar(self) -> ScalarType:
        return self._scalar

    @property
    def array(self) -> np.ndarray:
        return self._scalar * self._covariance.array

    @property
    def linop(self) -> pn.linops.LinearOperator:
        return self._scalar * self._covariance.linop

    @property
    def matrix(self) -> np.ndarray:
        return self._scalar * self._covariance.matrix


class SumCovariance(Covariance):
    def __init__(self, covariance1: Covariance, covariance2: Covariance) -> None:
        self._covariance1 = covariance1
        self._covariance2 = covariance2

        super().__init__(
            shape0=self._covariance1.shape0,
            shape1=self._covariance1.shape1,
        )

        if self._covariance1.shape0 != self._covariance2.shape0:
            raise ValueError()
        if self._covariance1.shape1 != self._covariance2.shape1:
            raise ValueError()

    @property
    def covariance1(self) -> Covariance:
        return self._covariance1

    @property
    def covariance2(self) -> Covariance:
        return self._covariance2

    @property
    def array(self) -> np.ndarray:
        return self._covariance1.array + self._covariance2.array

    @property
    def linop(self) -> pn.linops.LinearOperator:
        return self._covariance1.linop + self._covariance2.linop

    @property
    def matrix(self) -> np.ndarray:
        return self._covariance1.matrix + self._covariance2.matrix
