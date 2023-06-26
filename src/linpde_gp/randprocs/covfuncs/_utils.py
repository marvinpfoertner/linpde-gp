from probnum.randprocs import covfuncs as pn_covfuncs

from linpde_gp import linfuncops, linfunctls


def validate_covfunc_transformation(
    L: linfuncops.LinearFunctionOperator | linfunctls.LinearFunctional,
    covfunc: pn_covfuncs.CovarianceFunction,
    argnum: int,
):
    if argnum not in (0, 1):
        raise ValueError("`argnum` must either be 0 or 1.")

    # Check if the input shape of the covariance function matches the input domain shape
    # of the linear transformation
    covfunc_input_shape_argnum = (
        covfunc.input_shape_0 if argnum == 0 else covfunc.input_shape_1
    )

    if covfunc_input_shape_argnum != L.input_domain_shape:
        raise ValueError(
            f"`{L=!r}` can not be applied to `{covfunc=!r}`, since "
            f"`{L.input_domain_shape=}` is not equal to "
            f"`covfunc.input_shape_{argnum}={covfunc_input_shape_argnum}`."
        )

    # Check if the output shape of the covariance function matches the input codomain
    # shape of the linear transformation
    covfunc_output_shape_argnum = (
        covfunc.output_shape_0 if argnum == 0 else covfunc.output_shape_1
    )

    if covfunc_output_shape_argnum != L.input_codomain_shape:
        raise ValueError(
            f"`{L=!r}` can not be applied to `{covfunc=!r}`, since "
            f"`{L.input_codomain_shape=}` is not equal to "
            f"`covfunc.output_shape_{argnum}={covfunc_output_shape_argnum}`."
        )
