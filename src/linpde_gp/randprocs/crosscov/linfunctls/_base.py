from probnum.randprocs import covfuncs as pn_covfuncs

from linpde_gp import linfunctls

from .._pv_crosscov import ProcessVectorCrossCovariance


class LinearFunctionalProcessVectorCrossCovariance(ProcessVectorCrossCovariance):
    def __init__(
        self,
        covfunc: pn_covfuncs.CovarianceFunction,
        linfunctl: linfunctls.LinearFunctional,
        reverse: bool = True,
    ):
        self._covfunc = covfunc

        if linfunctl.input_domain_shape != (
            self._covfunc.input_shape_0 if reverse else self._covfunc.input_shape_1
        ):
            covfunc_input_shape_linfunctl = (
                self._covfunc.input_shape_0 if reverse else self._covfunc.input_shape_1
            )

            raise ValueError(
                f"The shape of the {'left' if reverse else 'right'} argument of the "
                f"covariance function must match the `input_domain_shape` of the linear"
                f"functional, but {covfunc_input_shape_linfunctl} != "
                f"{linfunctl.input_domain_shape}."
            )

        if linfunctl.input_codomain_shape != (
            self._covfunc.output_shape_0 if reverse else self._covfunc.output_shape_1
        ):
            covfunc_output_shape_linfunctl = (
                self._covfunc.output_shape_0
                if reverse
                else self._covfunc.output_shape_1
            )

            raise ValueError(
                f"The shape of the {'left' if reverse else 'right'} output of the "
                f"covariance function must match the `input_codomain_shape` of the "
                f"linear functional, but {covfunc_output_shape_linfunctl} != "
                f"{linfunctl.input_codomain_shape}."
            )

        self._linfunctl = linfunctl

        super().__init__(
            randproc_input_shape=(
                self._covfunc.input_shape_1 if reverse else self._covfunc.input_shape_0
            ),
            randproc_output_shape=(
                self._covfunc.output_shape_1
                if reverse
                else self._covfunc.output_shape_0
            ),
            randvar_shape=linfunctl.output_shape,
            reverse=reverse,
        )

    @property
    def covfunc(self) -> pn_covfuncs.CovarianceFunction:
        return self._covfunc

    @property
    def k(self) -> pn_covfuncs.CovarianceFunction:
        return self._covfunc

    @property
    def linfunctl(self) -> linfunctls.LinearFunctional:
        return self._linfunctl

    @property
    def L(self) -> linfunctls.LinearFunctional:
        return self._linfunctl
