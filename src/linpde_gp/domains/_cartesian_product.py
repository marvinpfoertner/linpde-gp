from collections.abc import Iterator, Sequence
import functools
import operator

import numpy as np
from probnum.typing import ArrayLike, ScalarType, ShapeLike

from . import _domain
from ._asdomain import DomainLike, asdomain


class CartesianProduct(_domain.Domain):
    def __init__(self, *domains: DomainLike) -> None:
        if len(domains) < 1:
            raise ValueError()

        self._domains = tuple(asdomain(domain) for domain in domains)

        if any(domain.ndims > 1 for domain in self._domains):
            raise ValueError()

        if any(domain.dtype != self._domains[0].dtype for domain in self._domains):
            raise ValueError()

        super().__init__(
            shape=(
                sum(
                    domain.shape[0] if domain.ndims == 1 else 1
                    for domain in self._domains
                ),
            ),
            dtype=self._domains[0].dtype,
        )

    @functools.cached_property
    def _as_box(self):
        from ._box import Box, Interval
        from ._point import Point

        if not all(isinstance(domain, (Interval, Box, Point)) for domain in self._domains):
            return None

        bounds = []

        for domain in self._domains:
            match domain:
                case Interval():
                    bounds.append(tuple(domain))
                case Box():
                    for interval in domain:
                        bounds.append(tuple(interval))
                case Point():
                    if domain.ndims == 0:
                        coord = np.asarray(domain)
                        bounds.append((coord, coord))
                    else:
                        coords = np.asarray(domain)
                        for coord in coords:
                            bounds.append((coord, coord))

        return Box(bounds)

    @property
    def boundary(self) -> Sequence[_domain.Domain]:
        raise NotImplementedError

    @property
    def volume(self) -> ScalarType:
        return functools.reduce(
            operator.mul, (domain.volume for domain in self._domains)
        )

    def __contains__(self, item: ArrayLike) -> bool:
        raise NotImplementedError

    def __eq__(self, other) -> bool:
        return isinstance(other, CartesianProduct) and all(
            domain_self == domain_other
            for domain_self, domain_other in zip(self._domains, other._domains)
        )

    def __len__(self) -> int:
        return len(self._domains)

    def __getitem__(self, idx) -> _domain.Domain:
        if isinstance(idx, int):
            return self._domains[idx]

        return CartesianProduct(*self._domains[idx])

    def __iter__(self) -> Iterator[_domain.Domain]:
        for idx in range(len(self)):
            yield self[idx]

    def uniform_grid(self, shape: ShapeLike, inset: ArrayLike = 0) -> np.ndarray:
        if self._as_box is not None:
            return self._as_box.uniform_grid(shape, inset=inset)

        raise NotImplementedError
