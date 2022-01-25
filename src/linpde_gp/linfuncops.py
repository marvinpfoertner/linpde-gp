import functools

import probnum as pn

from . import randprocs


class LinearFunctionOperator:
    @functools.singledispatchmethod
    def __call__(self, f, **kwargs):
        raise NotImplementedError()

    @__call__.register
    def _(self, f: pn.randprocs.GaussianProcess, **kwargs):
        mean = self(f._meanfun, argnum=0)
        crosscov = self(f._covfun, argnum=1)
        cov = self(crosscov, argnum=0)

        return pn.randprocs.GaussianProcess(mean, cov)


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

    @__call__.register
    def _(self, f: randprocs.mean_fns.JaxMean, *, argnum=0, **kwargs):
        assert argnum == 0

        return randprocs.mean_fns.JaxMean(
            self._L(f.jax, argnum=argnum, **kwargs),
            vectorize=True,
        )

    @__call__.register
    def _(self, f: randprocs.kernels.JaxKernel, *, argnum=0, **kwargs):
        return randprocs.kernels.JaxKernel(
            self._L(f.jax, argnum=argnum, **kwargs),
            input_dim=f.input_dim,
            vectorize=True,
        )
