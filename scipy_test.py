import unittest
import numpy as np
from numpy.testing import assert_equal

from scipy import misc, linalg, imag, signal


class Scipy_Test(unittest.TestCase):

    def setUp(self):

        return


    def test_convolve(self):

        x = np.array([1.0, 2.0, 3.0])
        h = np.array([0.0, 1.0, 0.0, 0.0, 0.0])

        print(x)
        assert_equal(signal.convolve(x, h),
                          [0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0])


if __name__ == '__main__':

    unittest.main()

