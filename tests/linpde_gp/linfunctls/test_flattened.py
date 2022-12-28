import pytest
import numpy as np

import linpde_gp
from linpde_gp.linfunctls import DiracFunctional, FlattenedLinearFunctional

def test_flat_stays_unchanged():
    X = np.array([1., 2., 3.])
    f = linpde_gp.functions.Zero((), ())

    D = DiracFunctional((), (), X)
    D_res = D(f)

    L = FlattenedLinearFunctional(D)
    L_res = L(f)

    assert L_res.shape == D_res.shape
    np.testing.assert_allclose(L_res, D_res)

def test_tensor_becomes_flat():
    I, O = 2, 3
    N = 5
    X = np.zeros((N, I))
    f = linpde_gp.functions.Zero((I,), (O,))

    D = DiracFunctional((I,), (O,), X)
    D_res = D(f)
    assert len(D_res.shape) == 2

    L = FlattenedLinearFunctional(D)
    L_res = L(f)

    assert len(L_res.shape) == 1
    assert L_res.shape[0] == np.prod(D_res.shape)

    # Check C order
    for i in range(D_res.shape[0]):
        for j in range(D_res.shape[1]):
            np.testing.assert_allclose(D_res[i, j], L_res[i*D_res.shape[1] + j])