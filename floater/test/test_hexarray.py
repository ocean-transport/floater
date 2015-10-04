import unittest
import numpy as np
from floater import hexgrid

class HexArraylTester(unittest.TestCase):

    def setUp(self):
        pass

    def test_neighbors(self):
        ha = hexgrid.HexArray(3,3)
        for n in xrange(9):
            if n == 4:
                self.assertSequenceEqual(
                    list(ha.neighbors(n)),
                    [0, 1, 5, 7, 6, 3]
                )
            else:
                self.assertEqual(len(ha.neighbors(n)), 0)

    def test_critical_points(self):
        a = np.array([[0,0,0],
                      [0,1,0],
                      [0,0,0]], np.float64)
        ha = hexgrid.HexArray(*a.shape)
        with self.assertRaises(ValueError):
            ha.classify_critical_points(np.zeros((4,4), dtype=np.float64))
        c = ha.classify_critical_points(a)

        self.assertSequenceEqual(a.shape, c.shape)
        # c should actually come out the same as a
        np.testing.assert_array_equal(a.astype('i4'), c)

    def test_maxima(self):
        a = np.array([[0,0,0],
                      [0,1,0],
                      [0,0,0]], np.float64)
        ha = hexgrid.HexArray(*a.shape)
        maxima = ha.maxima(a)
        self.assertEqual(len(maxima), 1)
        self.assertEqual(maxima[0], 4)
