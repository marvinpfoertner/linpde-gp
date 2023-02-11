import probnum as pn

from linpde_gp import domains, functions, linfuncops
from linpde_gp.typing import DomainLike


class LinearPDE:
    def __init__(
        self,
        domain: DomainLike,
        diffop: linfuncops.LinearDifferentialOperator,
        rhs: pn.functions.Function | None = None,
    ):
        self._domain = domains.asdomain(domain)

        if diffop.input_domain_shape != self._domain.shape:
            raise ValueError(
                "The shape of the domain of the differential operator's input "
                "function is not equal to the shape of the given domain object "
                f"({diffop.input_domain_shape} != {self._domain.shape})."
            )

        assert diffop.input_domain_shape == diffop.output_domain_shape, (
            "The domains of the input and output functions of a differential operator"
            "should be equal by definition."
        )

        self._diffop = diffop

        if rhs is None:
            rhs = functions.Zero(
                self._domain.shape,
                output_shape=self._diffop.output_codomain_shape,
            )

        if rhs.input_shape != self._domain.shape:
            raise ValueError(
                "The shape of the right-hand side function's domain is not equal to "
                "the shape of the given domain object "
                f"({rhs.input_shape} != {self._domain.shape})."
            )

        if rhs.output_shape != self._diffop.output_codomain_shape:
            raise ValueError(
                "The shape of the right-hand side function's codomain is not equal to "
                "the shape of the codomain of the differential operator's output "
                f"function ({rhs.output_shape} != "
                f"{self._diffop.output_codomain_shape})."
            )

        self._rhs = rhs

    @property
    def domain(self) -> domains.Domain:
        return self._domain

    @property
    def diffop(self) -> linfuncops.LinearDifferentialOperator:
        return self._diffop

    @property
    def rhs(self) -> pn.functions.Function:
        return self._rhs
