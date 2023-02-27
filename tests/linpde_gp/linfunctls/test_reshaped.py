import linpde_gp
import numpy as np
import pytest
from linpde_gp.linfunctls import DiracFunctional, ReshapedLinearFunctional


def test_flat_stays_unchanged():
    X = np.arange(3)
    f = linpde_gp.functions.Affine(1, 0)

    D = DiracFunctional((), (), X)
    D_res = D(f)

    L = ReshapedLinearFunctional(D)
    L_res = L(f)

    assert L_res.shape == D_res.shape
    np.testing.assert_allclose(L_res, D_res)

def test_tensor_becomes_flat():
    rng = np.random.default_rng(2196001)

    I, O = 2, 3
    N = 5
    X = np.arange(N*I).reshape((N, I), order='C')
    f = linpde_gp.functions.Affine(rng.normal(size=(O, I)), rng.normal(size=(O)))

    D = DiracFunctional((I,), (O,), X)
    D_res = D(f)
    assert len(D_res.shape) == 2

    L = ReshapedLinearFunctional(D)
    L_res = L(f)

    assert len(L_res.shape) == 1
    assert L_res.shape[0] == np.prod(D_res.shape)

    # Check C order
    for i in range(D_res.shape[0]):
        for j in range(D_res.shape[1]):
            np.testing.assert_allclose(D_res[i, j], L_res[i*D_res.shape[1] + j])