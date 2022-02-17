import functools


class LinearFunctionOperator:
    @functools.singledispatchmethod
    def __call__(self, f, **kwargs):
        raise NotImplementedError()
