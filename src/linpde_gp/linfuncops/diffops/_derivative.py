from ._coefficients import MultiIndex
from ._partial_derivative import PartialDerivative


class Derivative(PartialDerivative):
    def __init__(
        self,
        order: int,
    ) -> None:
        if order < 0:
            raise ValueError(f"Order must be >= 0, but got {order}.")

        super().__init__(
            MultiIndex(order),
        )

        self._order = order
