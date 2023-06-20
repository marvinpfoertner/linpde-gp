from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
from probnum.typing import ShapeType


class PartialDerivativeCoefficients:
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
        self, coefficient_dict: Dict[ShapeType, Dict[Tuple[int, ...], float]]
    ) -> None:
        self._input_codomain_shape_bound = None
        existing_key = None
        for key in coefficient_dict.keys():
            if existing_key is None:
                existing_key = key
                self._input_codomain_shape_bound = key
            elif len(key) != len(existing_key):
                raise ValueError("Codomain indices must all have the same length.")
            self._input_codomain_shape_bound = tuple(
                max(x, y) for x, y in zip(self._input_codomain_shape_bound, key)
            )

        self._num_entries = 0
        existing_key = None
        for sub_dict in coefficient_dict.values():
            for key in sub_dict.keys():
                if existing_key is None:
                    existing_key = key
                    self._input_domain_size = len(key)
                elif len(key) != len(existing_key):
                    raise ValueError("Multi-indices must all have the same length.")
                self._num_entries += 1

        self._coefficient_dict = coefficient_dict

    @property
    def as_dict(self) -> Dict[ShapeType, Dict[Tuple[int, ...], float]]:
        return self._coefficient_dict

    @property
    def num_entries(self) -> int:
        return self._num_entries

    @property
    def input_domain_size(self) -> int:
        return self._input_domain_size

    @property
    def input_codomain_ndim(self) -> int:
        return len(self._input_codomain_shape_bound)

    def validate_input_codomain_shape(self, input_codomain_shape: ShapeType) -> bool:
        if len(input_codomain_shape) != len(self._input_codomain_shape_bound):
            return False
        return all(
            x >= (y + 1)
            for x, y in zip(input_codomain_shape, self._input_codomain_shape_bound)
        )

    def validate_input_domain_shape(self, input_domain_shape: ShapeType) -> bool:
        if len(input_domain_shape) == 0:
            # R is isomorphic to R^1
            return self._input_domain_size == 1
        return (
            len(input_domain_shape) == 1
            and input_domain_shape[0] == self._input_domain_size
        )

    def __getitem__(self, key: ShapeType) -> Dict[Tuple[ShapeType, int], float]:
        return self._coefficient_dict[key]

    def __len__(self) -> int:
        return len(self._coefficient_dict)

    def __neg__(self) -> "PartialDerivativeCoefficients":
        return -1.0 * self

    def __add__(self, other) -> "PartialDerivativeCoefficients":
        if isinstance(other, PartialDerivativeCoefficients):
            if self.input_domain_size != other.input_domain_size:
                raise ValueError(
                    "Cannot add coefficients with input domain sizes"
                    f"{self.input_domain_size} != {other.input_domain_size}"
                )
            if self.input_codomain_ndim != other.input_codomain_ndim:
                raise ValueError(
                    "Cannot add coefficients with input codomain ndim"
                    f"{self.input_codomain_ndim} != {other.input_codomain_ndim}"
                )

            new_dic = deepcopy(self._coefficient_dict)
            for key in other.as_dict.keys():
                if key in new_dic.keys():
                    for sub_key in other.as_dict[key].keys():
                        if sub_key in new_dic[key].keys():
                            new_dic[key][sub_key] += other.as_dict[key][sub_key]
                        else:
                            new_dic[key][sub_key] = other.as_dict[key][sub_key]
                else:
                    new_dic[key] = deepcopy(other.as_dict[key])
            return PartialDerivativeCoefficients(new_dic)
        return NotImplemented

    def __sub__(self, other) -> "PartialDerivativeCoefficients":
        return self + (-other)

    def __rmul__(self, other) -> "PartialDerivativeCoefficients":
        if np.ndim(other) == 0:
            scaled_dic = deepcopy(self._coefficient_dict)
            for key in scaled_dic.keys():
                for sub_key in scaled_dic[key].keys():
                    scaled_dic[key][sub_key] *= other
            return PartialDerivativeCoefficients(scaled_dic)
        return NotImplemented
