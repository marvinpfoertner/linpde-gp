import dataclasses
import functools
from typing import Optional, Union

import numpy as np
import probnum as pn


@dataclasses.dataclass(frozen=True)
class BayesCGBelief:
    mean: np.ndarray
    cov_unscaled: pn.linops.LinearOperator
    cov_scale: np.floating = 0.0
    num_steps: int = 0

    @functools.cached_property
    def cov(self) -> pn.linops.LinearOperator:
        if self.num_steps == 0:
            return self.cov_unscaled
        else:
            return (self.cov_scale / self.num_steps) * self.cov_unscaled

    @functools.cached_property
    def x(self) -> pn.randvars.Normal:
        # TODO: This should actually return a multivariate t-distribution
        return pn.randvars.Normal(self.mean, self.cov)

    @classmethod
    def from_linear_system(
        self,
        problem: pn.problems.LinearSystem,
        mean: Optional[Union[np.ndarray, pn.randvars.Constant]] = None,
    ) -> "BayesCGBelief":
        if mean is None:
            mean = np.zeros_like(problem.b)
        else:
            if isinstance(mean, pn.randvars.Constant):
                mean = mean.support

            mean = mean.astype(
                np.result_type(problem.A.dtype, problem.b.dtype),
                copy=True,
            )

        return BayesCGBelief(
            mean=mean,
            cov_unscaled=pn.linops.aslinop(problem.A).inv(),
        )
