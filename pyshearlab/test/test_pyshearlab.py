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
    """Validate the regular call."""
    shape = tuple(shearletSystem['size'])

    # load data
    X = np.random.randn(*shape).astype(dtype)

    # decomposition
    coeffs = pyshearlab.SLsheardec2D(X, shearletSystem)

    # Test parameters
    assert coeffs.dtype == X.dtype
    assert coeffs.shape == shape + (shearletSystem['nShearlets'],)


def test_adjoint(dtype, shearletSystem):
    """Validate the adjoint."""
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
    """Validate the inverse."""
    X = np.random.randn(*shearletSystem['size']).astype(dtype)

    # decomposition
    coeffs = pyshearlab.SLsheardec2D(X, shearletSystem)

    # reconstruction
    Xrec = pyshearlab.SLshearrec2D(coeffs, shearletSystem)
    assert Xrec.dtype == X.dtype
    assert Xrec.shape == X.shape

    assert np.linalg.norm(X - Xrec) < 1e-5 * np.linalg.norm(X)


def test_adjoint_of_inverse(dtype, shearletSystem):
    """Validate the adjoint of the inverse."""
    X = np.random.randn(*shearletSystem['size']).astype(dtype)

    # decomposition
    coeffs = pyshearlab.SLsheardec2D(X, shearletSystem)

    # reconstruction
    Xrec = pyshearlab.SLshearrec2D(coeffs, shearletSystem)
    Xrecadj = pyshearlab.SLshearrecadjoint2D(Xrec, shearletSystem)
    assert Xrecadj.dtype == X.dtype
    assert Xrecadj.shape == coeffs.shape

    # <A^-1x, A^-1x> = <A^-* A^-1 x, x>.
    assert (pytest.approx(np.vdot(Xrec, Xrec), rel=1e-3, abs=0) ==
            np.vdot(Xrecadj, coeffs))


def test_inverse_of_adjoint(dtype, shearletSystem):
    """Validate the (pseudo-)inverse of the adjoint."""
    X = np.random.randn(*shearletSystem['size']).astype(dtype)

    # decomposition to create data.
    coeffs = pyshearlab.SLsheardec2D(X, shearletSystem)

    # Validate that the inverse works.
    Xadj = pyshearlab.SLshearadjoint2D(coeffs, shearletSystem)
    Xadjrec = pyshearlab.SLshearrecadjoint2D(Xadj, shearletSystem)
    assert Xadjrec.dtype == X.dtype
    assert Xadjrec.shape == coeffs.shape

    assert np.linalg.norm(coeffs - Xadjrec) < 1e-5 * np.linalg.norm(coeffs)


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
