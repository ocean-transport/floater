import numpy as np
import pytest
from floater import rclv

@pytest.fixture()
def sample_data_and_maximum():
    ny, nx = 100, 100
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x = (x-5)*2*np.pi/80
    y = (y - nx/2)*2*np.pi/100
    # Kelvin's cats eyes flow
    a = 0.8
    psi = np.log(np.cosh(y) + a*np.cos(x)) - np.log(1 + a)
    # want the extremum to be positive, need to reverse sign
    psi = -psi
    # max located at psi[50, 45] = 2.1972245773362196
    ji = (50,45)
    return psi, ji, psi[ji]

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


def test_contour_area(square_verts):
    region_area, hull_area, convex_def = rclv.contour_area(square_verts)
    assert region_area == 1.0
    assert hull_area == 1.0
    assert convex_def == 0.0


def test_contour_around_maximum(sample_data_and_maximum):
    psi, ji, psi_max = sample_data_and_maximum

    # we should get an error if the contour intersects the domain boundary
    with pytest.raises(ValueError):
        _ = rclv.find_contour_around_maximum(psi, ji, psi_max + 0.1)

    con, region_data, border_i, border_j = rclv.find_contour_around_maximum(
                                                            psi, ji, psi_max/2)

    # region data should be normalized to have the center point 0
    assert region_data[border_i[1], border_j[0]] == 0.0
    assert region_data.shape == (sum(border_j)+1, sum(border_j)+1)

    # the contour should be closed
    assert rclv.is_contour_closed(con)

    # check size against reference solution
    region_area, hull_area, convex_def = rclv.contour_area(con)
    assert region_area == 575.02954788959767
    assert hull_area == 575.0296629815823
    assert convex_def == (hull_area - region_area) / region_area


def test_convex_contour_around_maximum(sample_data_and_maximum):
    psi, ji, psi_max = sample_data_and_maximum

    # step determines how precise the contour identification is
    step = 0.001
    con, area = rclv.convex_contour_around_maximum(psi, ji, step)

    # check against reference solution
    assert area == 2693.8731123245125
    assert len(con) == 261

    # for this specific psi, contour should be symmetric around maximum
    assert tuple(con[:-1].mean(axis=0).astype('int')) == ji


def test_find_convex_contours(sample_data_and_maximum):
    psi, ji, psi_max = sample_data_and_maximum
    res =list(rclv.find_convex_contours(psi, step=0.001))

    assert len(res) == 1

    ji_found, con, area = res[0]
    assert tuple(ji_found) == ji
    assert len(con) == 261
    assert area == 2693.8731123245125
