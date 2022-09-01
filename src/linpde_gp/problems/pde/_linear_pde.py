import dataclasses

import probnum as pn

from linpde_gp import domains, linfuncops


@dataclasses.dataclass(frozen=True)
class LinearPDE:
    domain: domains.Domain
    diffop: linfuncops.LinearDifferentialOperator
    rhs: pn.functions.Function | pn.randprocs.RandomProcess

    def __post_init__(self):
        if self.diffop.input_domain_shape != self.domain.shape:
            raise ValueError(
                "The shape of the domain of the differential operator's input "
                "function is not equal to the shape of the given domain object "
                f"({self.diffop.input_domain_shape} != {self.domain.shape})."
            )

        assert self.diffop.input_domain_shape == self.diffop.output_domain_shape, (
            "The domains of the input and output functions of a differential operator"
            "should be equal by definition."
        )

        if self.rhs.input_shape != self.domain.shape:
            raise ValueError(
                "The shape of the right-hand side function's domain is not equal to "
                "the shape of the given domain object "
                f"({self.rhs.input_shape} != {self.domain.shape})."
            )

        if self.rhs.output_shape != self.diffop.output_codomain_shape:
            raise ValueError(
                "The shape of the right-hand side function's codomain is not equal to "
                "the shape of the codomain of the differential operator's output "
                f"function ({self.rhs.output_shape} != "
                f"{self.diffop.output_codomain_shape})."
            )
