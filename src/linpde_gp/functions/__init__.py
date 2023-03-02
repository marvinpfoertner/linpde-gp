from . import bases
from ._affine import Affine
from ._constant import Constant, Zero
from ._fourier import TruncatedSineSeries
from ._jax import JaxFunction, JaxLambdaFunction
from ._jax_arithmetic import JaxScaledFunction, JaxSumFunction
from ._piecewise import Piecewise, PiecewiseConstant, PiecewiseLinear
from ._polynomial import Monomial, Polynomial, RationalPolynomial
from ._stack import StackedFunction, stack
from ._truncated_gmm import TruncatedGaussianMixturePDF

from . import _linfunctls, linfuncops  # isort: skip
