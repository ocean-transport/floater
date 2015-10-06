import unittest
import numpy as np
from floater import hexgrid

def _get_test_array(shape=(3,3), nonzero=[(1,1)], dtype=np.float64):
    a = np.zeros(shape, dtype)
    for (j,i) in nonzero:
        a[j,i] = 1
    return a

class HexArrayTester(unittest.TestCase):

    def setUp(self):
        pass

    def test_neighbors(self):
        for ha in [hexgrid.HexArray(shape=(3,3)),
                   hexgrid.HexArray(np.empty((3,3), dtype=np.float64))]:
            for n in xrange(9):
                if n == 4:
                    self.assertSequenceEqual(
                        list(ha.neighbors(n)),
                        [0, 1, 5, 7, 6, 3]
                    )
                else:
                    self.assertEqual(len(ha.neighbors(n)), 0)

    def test_critical_points(self):
        a = _get_test_array()
        ha = hexgrid.HexArray(a)
        c = ha.classify_critical_points()

        self.assertSequenceEqual(a.shape, c.shape)
        # c should actually come out the same as a
        np.testing.assert_array_equal(a.astype('i4'), c)

    def test_maxima(self):
        a = _get_test_array()
        ha = hexgrid.HexArray(a)
        maxima = ha.maxima()
        self.assertEqual(len(maxima), 1)
        self.assertEqual(maxima[0], 4)


class HexArrayRegionTester(unittest.TestCase):

    def setUp(self):
        self.ha = hexgrid.HexArray(_get_test_array())

    def test_add_point(self):
        hr = hexgrid.HexArrayRegion(self.ha)
        self.assertNotIn(4, hr)
        hr.add_point(4)
        self.assertIn(4, hr)
        hr.add_point(5)
        self.assertIn(5, hr)

    def test_exterior_boundary(self):
        # points on the boundary should give no boundary
        hr = hexgrid.HexArrayRegion(self.ha)
        hr.add_point(0)
        self.assertSetEqual(hr.exterior_boundary(), set())
        # points in the middle should
        hr = hexgrid.HexArrayRegion(self.ha)
        hr.add_point(4)
        self.assertSetEqual(hr.exterior_boundary(), {0,1,5,7,6,3})

    def test_interior_exterior_boundary(self):
        a = np.array([[0,0,0,0,0],
                      [0,0,2,2,0],
                      [0,2,1,2,0],
                      [0,0,2,2,0],
                      [0,0,0,0,0]])
        ha = hexgrid.HexArray(shape=a.shape)
        hr = hexgrid.HexArrayRegion(ha)

        a = a.ravel()
        cpt = np.nonzero(a==1)[0][0]
        hr.add_point(cpt)
        bd = set(np.nonzero(a==2)[0])
        eb = hr.exterior_boundary()
        self.assertSetEqual(eb, bd)
        for pt in bd:
            hr.add_point(pt)
        ib = hr.interior_boundary()
        print hr.members
        self.assertSetEqual(ib, bd)

boundary_examples = """
Exterior Boundary
  o   o   .
o   x   o
  o   o   .

Interior Boundary
  .   .   .   .   .
.   .   o   o   .
  .   o   x   o   .
.   .   o   o   .
  .   .   .   .   .
"""
