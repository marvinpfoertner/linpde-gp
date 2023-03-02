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
