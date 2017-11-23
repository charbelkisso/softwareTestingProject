from scipy import signal
import numpy as np
import unittest
from numpy.testing import assert_equal, assert_almost_equal


class Test_Convolve(unittest.TestCase):

    def setUp(self):

        self.empty1 = np.array([])
        self.empty2 = np.array([])
        self.empty_res = np.array([])

        self.x = np.array([1.0, 2.0, 3.0])
        self.h = np.array([0.0, 0.5, 0.5])
        self.res_full = [0.0, 0.5, 1.5, 2.5, 1.5]
        self.res_same = [0.5, 1.5, 2.5]
        self.res_valid = [1.5]

    def test_default(self):
        """
        this is the defult mode of convolve function with correct value
        and expecting correct return

        """
        assert_equal(signal.convolve(self.x, self.h), self.res_full)



    def test_empty_with_direkt(self):
        """
        this test check if we could pass an empty array as input 
        with direct is not allowed to have empty arrays as input
        """
        self.assertRaisesRegexp(ValueError, "math domain error", signal.convolve, self.empty1, self.empty2)
        self.assertRaisesRegexp(ValueError, "a cannot be empty",
                        signal.convolve, self.empty1, self.empty2, 'same', 'direct')
        self.assertRaisesRegexp(ValueError, "a cannot be empty",
                        signal.convolve, self.empty1, self.empty2,'valid', 'direct')


    def test_empty_with_fft(self):
        """
        this test check if we could pass an empty array as input
        with fft is allowed to have empty array as input
        """
        assert_equal(signal.convolve(self.empty1, self.empty2, 'valid', 'fft'), self.empty_res)
        assert_equal(signal.convolve(self.empty1, self.empty2, 'same', 'fft'), self.empty_res)
        assert_equal(signal.convolve(self.empty1, self.empty2, 'full', 'fft'), self.empty_res)



    def test_with_one_empty(self):
        """
        testing the function with one empty input

        :return:
        """
        assert_equal(signal.convolve(self.empty1, self.h, 'full', 'fft'), self.empty_res)
        assert_equal(signal.convolve(self.empty1, self.h, 'same', 'fft'), self.empty_res)
        assert_equal(signal.convolve(self.empty1, self.h, 'valid', 'fft'), self.empty_res)
        assert_equal(signal.convolve(self.x, self.empty2, 'full', 'fft'), self.empty_res)
        assert_equal(signal.convolve(self.x, self.empty2, 'same', 'fft'), self.empty_res)
        assert_equal(signal.convolve(self.x, self.empty2, 'valid', 'fft'), self.empty_res)
        
    def test_modes_methods(self):
        """
        check if the convolve function will hold write result with
        different modes and methods
        :return:
        """
        assert_almost_equal(signal.convolve(self.x, self.h, 'full', 'fft'), self.res_full)
        assert_almost_equal(signal.convolve(self.x, self.h, 'same', 'fft'), self.res_same)
        assert_almost_equal(signal.convolve(self.x, self.h, 'valid', 'fft'), self.res_valid)
        assert_almost_equal(signal.convolve(self.x, self.h, 'full', 'direct'), self.res_full)
        assert_almost_equal(signal.convolve(self.x, self.h, 'same', 'direct'), self.res_same)
        assert_almost_equal(signal.convolve(self.x, self.h, 'valid', 'direct'), self.res_valid)

if __name__ == '__main__':
    unittest.main()