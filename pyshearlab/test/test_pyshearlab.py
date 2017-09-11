import numpy as np
import pyshearlab
import pytest


@pytest.fixture(scope='module', params=['float32', 'float64'])
def dtype(request):
    return request.param


@pytest.fixture(scope='module', params=['64', '128'])
def shape(request):
    size = int(request.param)
    return (size, size)


@pytest.fixture(scope='module')
def shearletSystem(shape):
    scales = 2
    return pyshearlab.SLgetShearletSystem2D(0,
                                            shape[0], shape[1],
                                            scales)


def test_call(dtype, shearletSystem):
    shape = tuple(shearletSystem['size'])

    # load data
    X = np.random.randn(*shape).astype(dtype)

    # decomposition
    coeffs = pyshearlab.SLsheardec2D(X, shearletSystem)

    # Test parameters
    assert coeffs.dtype == X.dtype
    assert coeffs.shape == shape + (shearletSystem['nShearlets'],)


def test_adjoint(dtype, shearletSystem):
    shape = tuple(shearletSystem['size'])

    # load data
    X = np.random.randn(*shape).astype(dtype)

    # decomposition
    coeffs = pyshearlab.SLsheardec2D(X, shearletSystem)

    # adjoint
    Xadj = pyshearlab.SLshearadjoint2D(coeffs, shearletSystem)
    assert Xadj.dtype == X.dtype
    assert Xadj.shape == X.shape

    # <Ax, Ax> should equal <x, AtAx>
    assert (pytest.approx(np.vdot(coeffs, coeffs), rel=1e-3, abs=0) ==
            np.vdot(X, Xadj))


def test_inverse(dtype, shearletSystem):
    X = np.random.randn(*shearletSystem['size']).astype(dtype)

    # decomposition
    coeffs = pyshearlab.SLsheardec2D(X, shearletSystem)

    # reconstruction
    Xrec = pyshearlab.SLshearrec2D(coeffs, shearletSystem)
    assert Xrec.dtype == X.dtype
    assert Xrec.shape == X.shape

    assert np.linalg.norm(X - Xrec) < 1e-5 * np.linalg.norm(X)


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
