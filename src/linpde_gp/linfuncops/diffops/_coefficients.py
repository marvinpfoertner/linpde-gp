from collections.abc import Mapping, Iterator
from copy import deepcopy

import numpy as np
from probnum.typing import ShapeType


class PartialDerivativeCoefficients(
    Mapping[ShapeType, Mapping[Tuple[int, ...], float]]
):
    r"""Partial derivative coefficients of a linear differential operator.

    Any linear differential operator can be written as a sum of partial derivatives.

    The coefficients are stored in a dictionary of the form
    {input_codomain_index: {multi_index: coefficient}}.

    For example, the Laplacian operator in 2D can be written as

    .. math::
        \Delta = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}

    and is stored as

    .. code-block:: python

        {
            (): {
                (2, 0): 1.0,
                (0, 2): 1.0,
            }
        }

    Parameters
    ----------
    coefficient_dict :
        Dictionary of coefficients.
    """

    def __init__(
        self,
        coefficient_dict: dict[ShapeType, dict[tuple[int, ...], float]],
        input_domain_shape: ShapeType,
        input_codomain_shape: ShapeType,
    ) -> None:
        if len(input_domain_shape) > 1:
            raise ValueError(
                f"Input domain must be R or R^n, but got shape {input_domain_shape}."
            )
        input_size = int(np.prod(input_domain_shape))
        self._num_entries = 0
        for codomain_idx in coefficient_dict.keys():
            if len(codomain_idx) != len(input_codomain_shape) or not all(
                x < y for x, y in zip(codomain_idx, input_codomain_shape)
            ):
                raise ValueError(
                    f"Codomain index {codomain_idx} does not match shape"
                    "{input_codomain_shape}."
                )
            for multi_index in coefficient_dict[codomain_idx].keys():
                if len(multi_index) != input_size:
                    raise ValueError(
                        f"Multi-index {multi_index} does not match input domain shape "
                        f"{input_domain_shape}."
                    )
                if any(x < 0 for x in multi_index):
                    raise ValueError(
                        f"Multi-index {multi_index} contains negative entries."
                    )
                self._num_entries += 1

        self._coefficient_dict = coefficient_dict
        self._input_domain_shape = input_domain_shape
        self._input_codomain_shape = input_codomain_shape

    @property
    def num_entries(self) -> int:
        return self._num_entries

    @property
    def input_domain_shape(self) -> ShapeType:
        return self._input_domain_shape

    @property
    def input_codomain_shape(self) -> ShapeType:
        return self._input_codomain_shape

    def __getitem__(self, key: ShapeType) -> dict[tuple[int, ...], float]:
        return self._coefficient_dict[key]

    def __len__(self) -> int:
        return len(self._coefficient_dict)

    def __iter__(self) -> Iterator[ShapeType]:
        return iter(self._coefficient_dict)

    def __neg__(self) -> "PartialDerivativeCoefficients":
        return -1.0 * self

    def __add__(self, other) -> "PartialDerivativeCoefficients":
        if isinstance(other, PartialDerivativeCoefficients):
            if self.input_domain_shape != other.input_domain_shape:
                raise ValueError(
                    "Cannot add coefficients with input domain shapes"
                    f"{self.input_domain_shape} != {other.input_domain_shape}"
                )
            if self.input_codomain_shape != other.input_codomain_shape:
                raise ValueError(
                    "Cannot add coefficients with input codomain shapes"
                    f"{self.input_codomain_shape} != {other.input_codomain_shape}"
                )

            new_dict = deepcopy(self._coefficient_dict)
            for codomain_idx in other.keys():
                if codomain_idx in new_dict.keys():
                    for multi_index in other[codomain_idx].keys():
                        if multi_index in new_dict[codomain_idx].keys():
                            new_dict[codomain_idx][multi_index] += other[codomain_idx][multi_index]
                        else:
                            new_dict[codomain_idx][multi_index] = other[codomain_idx][multi_index]
                else:
                    new_dict[codomain_idx] = deepcopy(other[codomain_idx])
            return PartialDerivativeCoefficients(
                new_dict, self.input_domain_shape, self.input_codomain_shape
            )
        return NotImplemented

    def __sub__(self, other) -> "PartialDerivativeCoefficients":
        return self + (-other)

    def __rmul__(self, other) -> "PartialDerivativeCoefficients":
        if np.ndim(other) == 0:
            scaled_dict = deepcopy(self._coefficient_dict)
            for codomain_idx in scaled_dict.keys():
                for multi_index in scaled_dict[codomain_idx].keys():
                    scaled_dict[codomain_idx][multi_index] *= other
            return PartialDerivativeCoefficients(
                scaled_dict, self.input_domain_shape, self.input_codomain_shape
            )
        return NotImplemented
