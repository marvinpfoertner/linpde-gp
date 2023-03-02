from linpde_gp import linfuncops
from linpde_gp.randprocs import covfuncs

########################################################################################
# General `LinearFunctionOperators` ####################################################
########################################################################################


@linfuncops.LinearFunctionOperator.__call__.register  # pylint: disable=no-member
def _(self, k: covfuncs.Zero, /, *, argnum: int = 0):
    assert argnum in (0, 1)

    return covfuncs.Zero(
        input_shape_0=self.output_domain_shape if argnum == 0 else k.input_shape_0,
        input_shape_1=self.output_domain_shape if argnum == 1 else k.input_shape_1,
        output_shape_0=self.output_codomain_shape if argnum == 0 else k.output_shape_0,
        output_shape_1=self.output_codomain_shape if argnum == 1 else k.output_shape_1,
    )


@linfuncops.LinearFunctionOperator.__call__.register  # pylint: disable=no-member
def _(self, k: covfuncs.StackCovarianceFunction, /, *, argnum: int = 0):
    if (argnum == 0 and k.output_idx == 1) or (argnum == 1 and k.output_idx == 0):
        return covfuncs.StackCovarianceFunction(
            *(self(covfunc, argnum=argnum) for covfunc in k.covfuncs),
            output_idx=k.output_idx,
        )

    raise NotImplementedError()


@linfuncops.SelectOutput.__call__.register  # pylint: disable=no-member
def _(
    self,
    k: covfuncs.IndependentMultiOutputCovarianceFunction,
    /,
    *,
    argnum: int = 0,
):
    zero_cov = covfuncs.Zero(
        input_shape_0=k.input_shape_0,
        input_shape_1=k.input_shape_1,
        output_shape_0=(),
        output_shape_1=(),
    )

    assert isinstance(self.idx, int)

    return covfuncs.StackCovarianceFunction(
        *([zero_cov] * self.idx),
        k.covfuncs[self.idx],
        *([zero_cov] * (len(k.covfuncs) - self.idx - 1)),
        output_idx=1 - argnum,
    )


@linfuncops.SelectOutput.__call__.register  # pylint: disable=no-member
def _(self, k: covfuncs.StackCovarianceFunction, /, *, argnum: int = 0):
    if (argnum == 0 and k.output_idx == 0) or (argnum == 1 and k.output_idx == 1):
        return k.covfuncs[self.idx]

    return super(linfuncops.SelectOutput, self).__call__(k, argnum=argnum)
