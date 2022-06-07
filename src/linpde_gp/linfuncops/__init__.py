from . import diffops
from ._arithmetic import (
    CompositeLinearFunctionOperator,
    ScaledLinearFunctionOperator,
    SumLinearFunctionOperator,
)
from ._identity import Identity
from ._linfuncop import LinearFunctionOperator
from .diffops import LambdaLinearDifferentialOperator, LinearDifferentialOperator
