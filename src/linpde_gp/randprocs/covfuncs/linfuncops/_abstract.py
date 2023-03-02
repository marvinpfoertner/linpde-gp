from probnum.randprocs import covfuncs

from linpde_gp import linfuncops


class CrossCovarianceFunction_Identity_LinearFunctionOperator(
    covfuncs.CovarianceFunction
):
    def __init__(
        self,
        k: covfuncs.CovarianceFunction,
        L: linfuncops.LinearFunctionOperator,
        reverse: bool = False,
    ):
        self._k = k
        self._L = L
        self._reverse = bool(reverse)

        if self._L.input_domain_shape != (
            self._k.input_shape_0 if self._reverse else self._k.input_shape_1
        ):
            raise ValueError()

        if self._L.input_codomain_shape != (
            self._k.output_shape_0 if self._reverse else self._k.output_shape_1
        ):
            raise ValueError()

        if self._reverse:
            input_shape_0 = self._L.output_domain_shape
            input_shape_1 = self._k.input_shape_1

            output_shape_0 = self._L.output_codomain_shape
            output_shape_1 = self._k.output_shape_1
        else:
            input_shape_0 = self._k.input_shape_0
            input_shape_1 = self._L.output_domain_shape

            output_shape_0 = self._k.output_shape_0
            output_shape_1 = self._L.output_codomain_shape

        super().__init__(
            input_shape_0=input_shape_0,
            input_shape_1=input_shape_1,
            output_shape_0=output_shape_0,
            output_shape_1=output_shape_1,
        )

    @property
    def k(self) -> covfuncs.CovarianceFunction:
        return self._k

    @property
    def L(self) -> linfuncops.LinearFunctionOperator:
        return self._L

    @property
    def reverse(self) -> bool:
        return self._reverse

    @property
    def argnum(self) -> int:
        return 0 if self.reverse else 1


class CovarianceFunction_LinearFunctionOperator(covfuncs.CovarianceFunction):
    def __init__(
        self,
        k: covfuncs.CovarianceFunction,
        L0: linfuncops.LinearFunctionOperator,
        L1: linfuncops.LinearFunctionOperator,
    ):
        self._k = k
        self._L0 = L0
        self._L1 = L1

        if self._L0.input_domain_shape != self._k.input_shape_0:
            raise ValueError()

        if self._L0.input_codomain_shape != self._k.output_shape_0:
            raise ValueError()

        if self._L1.input_domain_shape != self._k.input_shape_1:
            raise ValueError()

        if self._L1.input_codomain_shape != self._k.output_shape_1:
            raise ValueError()

        super().__init__(
            input_shape_0=self._L0.output_domain_shape,
            input_shape_1=self._L1.output_domain_shape,
            output_shape_0=self._L0.output_codomain_shape,
            output_shape_1=self._L1.output_codomain_shape,
        )

    @property
    def k(self) -> covfuncs.CovarianceFunction:
        return self._k

    @property
    def L0(self) -> linfuncops.LinearFunctionOperator:
        return self._L0

    @property
    def L1(self) -> linfuncops.LinearFunctionOperator:
        return self._L1
