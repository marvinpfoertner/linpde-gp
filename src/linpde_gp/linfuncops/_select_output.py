import functools

import numpy as np
from probnum.typing import ShapeLike

from ._linfuncop import LinearFunctionOperator


class SelectOutput(LinearFunctionOperator):
    def __init__(
        self,
        input_shapes: tuple[ShapeLike, ShapeLike],
        idx,
    ) -> None:
        if isinstance(idx, tuple) and len(idx) == 1:
            idx = idx[0]
        self._idx = idx

        super().__init__(
            input_shapes,
            output_shapes=(
                input_shapes[0],
                np.empty(input_shapes[1], dtype=[])[self._idx],
            ),
        )

    @property
    def idx(self):
        return self._idx

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)

    def __repr__(self) -> str:
        return f"SelectOutput(idx={self.idx})"
