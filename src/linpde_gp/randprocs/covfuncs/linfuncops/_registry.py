from probnum.randprocs import covfuncs as pn_covfuncs

from linpde_gp import linfuncops
from linpde_gp.randprocs import covfuncs

from .._utils import validate_covfunc_transformation

########################################################################################
# General `LinearFunctionOperators` ####################################################
########################################################################################


@linfuncops.LinearFunctionOperator.__call__.register  # pylint: disable=no-member
def _(
    self, k: covfuncs.JaxScaledCovarianceFunction, /, *, argnum: int = 0
) -> covfuncs.JaxScaledCovarianceFunction:
    validate_covfunc_transformation(self, k, argnum)

    return k.scalar * self(k.covfunc, argnum=argnum)


@linfuncops.LinearFunctionOperator.__call__.register  # pylint: disable=no-member
def _(
    self, k: covfuncs.JaxSumCovarianceFunction, /, *, argnum: int = 0
) -> covfuncs.JaxSumCovarianceFunction:
    validate_covfunc_transformation(self, k, argnum)

    return covfuncs.JaxSumCovarianceFunction(
        *(self(summand, argnum=argnum) for summand in k.summands)
    )


@linfuncops.LinearFunctionOperator.__call__.register  # pylint: disable=no-member
def _(self, k: covfuncs.StackCovarianceFunction, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    if (argnum == 0 and k.output_idx == 1) or (argnum == 1 and k.output_idx == 0):
        return covfuncs.StackCovarianceFunction(
            *(self(covfunc, argnum=argnum) for covfunc in k.covfuncs),
            output_idx=k.output_idx,
        )

    raise NotImplementedError()


@linfuncops.LinearFunctionOperator.__call__.register  # pylint: disable=no-member
def _(self, k: covfuncs.Zero, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    return covfuncs.Zero(
        input_shape_0=self.output_domain_shape if argnum == 0 else k.input_shape_0,
        input_shape_1=self.output_domain_shape if argnum == 1 else k.input_shape_1,
        output_shape_0=self.output_codomain_shape if argnum == 0 else k.output_shape_0,
        output_shape_1=self.output_codomain_shape if argnum == 1 else k.output_shape_1,
    )


########################################################################################
# `Identity` ###########################################################################
########################################################################################


@linfuncops.Identity.__call__.register  # pylint: disable=no-member
def _(
    self, covfunc: pn_covfuncs.CovarianceFunction, /, *, argnum: int = 0
) -> pn_covfuncs.CovarianceFunction:
    validate_covfunc_transformation(self, covfunc, argnum)

    return covfunc


########################################################################################
# `SelectOutput` #######################################################################
########################################################################################


@linfuncops.SelectOutput.__call__.register  # pylint: disable=no-member
def _(self, k: covfuncs.StackCovarianceFunction, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    if (argnum == 0 and k.output_idx == 0) or (argnum == 1 and k.output_idx == 1):
        return k.covfuncs[self.idx]

    return super(linfuncops.SelectOutput, self).__call__(k, argnum=argnum)


@linfuncops.SelectOutput.__call__.register  # pylint: disable=no-member
def _(
    self,
    k: covfuncs.IndependentMultiOutputCovarianceFunction,
    /,
    *,
    argnum: int = 0,
):
    validate_covfunc_transformation(self, k, argnum)

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
