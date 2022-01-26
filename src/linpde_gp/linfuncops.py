import functools


class LinearFunctionOperator:
    @functools.singledispatchmethod
    def __call__(self, f, **kwargs):
        raise NotImplementedError()


class JaxLinearOperator(LinearFunctionOperator):
    def __init__(self, L) -> None:
        self._L = L

        super().__init__()

    @functools.singledispatchmethod
    def __call__(self, f, *, argnum=0, **kwargs):
        try:
            return super().__call__(f, argnum=argnum, **kwargs)
        except NotImplementedError:
            return self._L(f, argnum=argnum, **kwargs)
