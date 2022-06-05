from numpy import isin
import probnum as pn

from linpde_gp.typing import RandomProcessLike

from ._deterministic_process import DeterministicProcess


def asrandproc(f: RandomProcessLike) -> pn.randprocs.RandomProcess:
    match f:
        case pn.randprocs.RandomProcess():
            return f
        case pn.Function():
            return DeterministicProcess(f)

    raise TypeError(
        f"Cannot convert an object of type `{type(f)}` to a `RandomProcess`"
    )
