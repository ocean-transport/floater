import numpy as np
import pytest
from floater import rclv

@pytest.fixture()
def psi():
    ny, nx = 100, 100
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x = (x-5)*2*np.pi/80
    y = (y - nx/2)*2*np.pi/100
    # Kelvin's cats eyes flow
    a = 0.8
    psi = np.log(np.cosh(y) + a*np.cos(x)) - np.log(1 + a)
    return psi

@pytest.fixture()
def square_verts():
    return np.array([[0,0], [1,0], [1,1], [0,1], [0,0]])


def test_polygon_area(square_verts):
    assert rclv.polygon_area(square_verts) == 1.0
    # try without the last vertex
    assert rclv.polygon_area(square_verts[:-1]) == 1.0


def test_get_local_region():
    # create some data
    n = 10
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    (j,i), x_reg = rclv.get_local_region(x, (2,2), border_j=(2,2), border_i=(2,2))
    assert x_reg.shape == (5,5)
    assert x_reg[j,i] == 0
    assert x_reg[j,0] == 2
    assert x_reg[j,-1] == -2

    with pytest.raises(ValueError) as ve:
        (j,i), x_reg = rclv.get_local_region(x, (2,2), border_j=(3,2), border_i=(2,2))
    with pytest.raises(ValueError) as ve:
        (j,i), x_reg = rclv.get_local_region(x, (2,2), border_j=(2,7), border_i=(2,2))
    with pytest.raises(ValueError) as ve:
        (j,i), x_reg = rclv.get_local_region(x, (2,2), border_j=(2,2), border_i=(3,2))
    with pytest.raises(ValueError) as ve:
        (j,i), x_reg = rclv.get_local_region(x, (2,2), border_j=(2,2), border_i=(2,7))


def test_is_contour_closed(square_verts):
    assert rclv.is_contour_closed(square_verts)
    assert not rclv.is_contour_closed(square_verts[:-1])


def test_point_in_contour(square_verts):
    assert rclv.point_in_contour(square_verts, (0.5, 0.5))
    assert not rclv.point_in_contour(square_verts, (1.5, 0.5))
