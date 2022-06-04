import functools

from probnum.typing import FloatLike

from linpde_gp.functions import Affine, Constant

from . import _linfuncop


class UndefinedLebesgueIntegral(_linfuncop.LinearFunctionOperator):
    def __init__(self, lower_bound: FloatLike = 0.0) -> None:
        super().__init__(input_shapes=((), ()), output_shapes=((), ()))

        self._lower_bound = lower_bound

    @property
    def lower_bound(self) -> float:
        return self._lower_bound

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        raise NotImplementedError()


@UndefinedLebesgueIntegral.__call__.register  # pylint: disable=no-member
def _(self, f: Constant) -> Affine:
    return Affine(
        A=f.value,
        b=-f.value * self.lower_bound,
    )
