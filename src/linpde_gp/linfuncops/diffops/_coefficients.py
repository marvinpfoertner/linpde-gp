from collections.abc import Iterator, Mapping
from copy import deepcopy
import functools

import numpy as np
from probnum.typing import ArrayLike, ShapeType


class MultiIndex:
    r"""Multi-index representation of a partial derivative.

    A multi-index is an array of non-negative integers which is used to represent
    a partial derivative of a function.

    For example, the multi-index :math:`(1, 2, 0)` represents the partial derivative
    :math:`\frac{\partial^3}{\partial x_1 \partial x_2^2}`
    of a function :math:`f: \mathbb{R}^3 \to \mathbb{R}`.
    """

    def __init__(self, multi_index: ArrayLike) -> None:
        self._multi_index = np.asarray(multi_index, dtype=int)
        if np.any(self._multi_index < 0):
            raise ValueError(f"Multi-index {multi_index} contains negative entries.")
        self._multi_index.setflags(write=False)

    @classmethod
    def from_index(
        cls, index: tuple[int, ...], shape: ShapeType, order: int
    ) -> "MultiIndex":
        multi_index = np.zeros(shape, dtype=int)
        multi_index[index] = order
        return cls(multi_index)

    @functools.cached_property
    def order(self) -> int:
        return np.sum(self._multi_index)

    @functools.cached_property
    def is_mixed(self) -> bool:
        return np.count_nonzero(self._multi_index) > 1

    @property
    def array(self) -> np.ndarray:
        return self._multi_index

    @property
    def shape(self) -> ShapeType:
        return self._multi_index.shape

    def __getitem__(self, index: tuple[int, ...]) -> int:
        return self._multi_index[index]

    def __hash__(self) -> int:
        return hash(self._multi_index.data.tobytes())

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, MultiIndex):
            return NotImplemented
        return np.all(self.array == __o.array)

    def __repr__(self) -> str:
        return f"MultiIndex({self._multi_index.tolist()})"


class PartialDerivativeCoefficients(Mapping[ShapeType, Mapping[MultiIndex, float]]):
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
                MultiIndex((2, 0)): 1.0,
                MultiIndex((0, 2)): 1.0,
            }
        }

    Parameters
    ----------
    coefficient_dict :
        Dictionary of coefficients.
    """

    def __init__(
        self,
        coefficient_dict: Mapping[ShapeType, Mapping[MultiIndex, float]],
        input_domain_shape: ShapeType,
        input_codomain_shape: ShapeType,
    ) -> None:
        self._num_entries = 0
        for codomain_idx in coefficient_dict.keys():
            if len(codomain_idx) != len(input_codomain_shape) or not all(
                x < y for x, y in zip(codomain_idx, input_codomain_shape)
            ):
                raise ValueError(
                    f"Codomain index {codomain_idx} does not match shape"
                    f"{input_codomain_shape}."
                )
            for multi_index in coefficient_dict[codomain_idx].keys():
                if multi_index.shape != input_domain_shape:
                    raise ValueError(
                        f"Multi-index shape {multi_index.shape} does not match "
                        f"input domain shape {input_domain_shape}."
                    )
                self._num_entries += 1

        self._coefficient_dict = coefficient_dict
        self._input_domain_shape = input_domain_shape
        self._input_codomain_shape = input_codomain_shape

    @property
    def num_entries(self) -> int:
        return self._num_entries

    @functools.cached_property
    def has_mixed(self) -> bool:
        return any(
            multi_index.is_mixed
            for codomain_idx in self._coefficient_dict
            for multi_index in self._coefficient_dict[codomain_idx]
        )

    @property
    def input_domain_shape(self) -> ShapeType:
        return self._input_domain_shape

    @property
    def input_codomain_shape(self) -> ShapeType:
        return self._input_codomain_shape

    def __getitem__(self, codomain_idx: ShapeType) -> Mapping[MultiIndex, float]:
        return self._coefficient_dict[codomain_idx]

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
                            new_dict[codomain_idx][multi_index] += other[codomain_idx][
                                multi_index
                            ]
                        else:
                            new_dict[codomain_idx][multi_index] = other[codomain_idx][
                                multi_index
                            ]
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
