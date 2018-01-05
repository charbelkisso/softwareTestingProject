from scipy import signal
import numpy as np
import unittest
from numpy.testing import assert_equal, assert_almost_equal


class Test_ConvolveCorrealte(unittest.TestCase):
    """
    Tests are designed to test convolve & correlate functions from signal library from scipy
    """

    def setUp(self):
        """
        Def arrays:
        """
        self.empty1 = np.array([])
        self.empty2 = np.array([])

        self.x = np.array([1.0, 2.0, 3.0])
        self.h = np.array([0.0, 0.5, 0.5])
        """
        Def Convolve Expected Results:
        """
        self.conv_res_full = [0.0, 0.5, 1.5, 2.5, 1.5]
        self.conv_res_empty = np.array([])
        self.conv_res_same = [0.5, 1.5, 2.5]
        self.conv_res_valid = [1.5]
        """
        Def Correlate Expected Results:
        """
        self.corr_res_full = [0.5,1.5,2.5,1.5,0.0]
        self.corr_res_empty = np.array([])
        self.corr_res_same = [1.5, 2.5, 1.5]
        self.corr_res_valid = [2.5]


    def test_default(self):
        """
        this is the defult mode of convolve function with correct value
        and expecting correct return

        """
        assert_equal(signal.convolve(self.x, self.h), self.conv_res_full)
        """
        this is the defult mode of correalte function with correct value
        and expecting correct return

        """
        assert_equal(signal.correlate(self.x, self.h), self.corr_res_full)



    def test_empty_with_direkt(self):
        """
        this test check if we could pass an empty array as input
        with direct is not allowed to have empty arrays as input
        in covolve function
        """
        self.assertRaisesRegexp(ValueError, "math domain error", signal.convolve, self.empty1, self.empty2)
        self.assertRaisesRegexp(ValueError, "a cannot be empty",
                        signal.convolve, self.empty1, self.empty2, 'same', 'direct')
        self.assertRaisesRegexp(ValueError, "a cannot be empty",
                        signal.convolve, self.empty1, self.empty2,'valid', 'direct')

        """
        this test check if we could pass an empty array as input
        with direct is not allowed to have empty arrays as input
        in correlate function
        """
        self.assertRaisesRegexp(ValueError, "math domain error", signal.correlate, self.empty1, self.empty2)


    def test_empty_with_fft(self):
        """
        this test check if we could pass an empty array as input
        with fft is allowed to have empty array as input in convolve
        """
        assert_equal(signal.convolve(self.empty1, self.empty2, 'valid', 'fft'), self.conv_res_empty)
        assert_equal(signal.convolve(self.empty1, self.empty2, 'same', 'fft'), self.conv_res_empty)
        assert_equal(signal.convolve(self.empty1, self.empty2, 'full', 'fft'), self.conv_res_empty)

        """
        this test check if we could pass an empty array as input
        with fft is allowed to have empty array as input in correlate
        """
        assert_equal(signal.correlate(self.empty1, self.empty2, 'valid', 'fft'), self.corr_res_empty)
        assert_equal(signal.correlate(self.empty1, self.empty2, 'same', 'fft'), self.corr_res_empty)
        assert_equal(signal.correlate(self.empty1, self.empty2, 'full', 'fft'), self.corr_res_empty)

    def test_with_one_empty(self):
        """
        testing the convolve function with one empty input
        """
        assert_equal(signal.convolve(self.empty1, self.h, 'full', 'fft'), self.conv_res_empty)
        assert_equal(signal.convolve(self.empty1, self.h, 'same', 'fft'), self.conv_res_empty)
        assert_equal(signal.convolve(self.empty1, self.h, 'valid', 'fft'), self.conv_res_empty)
        assert_equal(signal.convolve(self.x, self.empty2, 'full', 'fft'), self.conv_res_empty)
        assert_equal(signal.convolve(self.x, self.empty2, 'same', 'fft'), self.conv_res_empty)
        assert_equal(signal.convolve(self.x, self.empty2, 'valid', 'fft'), self.conv_res_empty)

        """
        testing the correlate with one empty input
        """
        assert_equal(signal.correlate(self.empty1, self.h, 'full', 'fft'), self.corr_res_empty)
        assert_equal(signal.correlate(self.empty1, self.h, 'same', 'fft'), self.corr_res_empty)
        assert_equal(signal.correlate(self.empty1, self.h, 'valid', 'fft'), self.corr_res_empty)
        assert_equal(signal.correlate(self.x, self.empty2, 'full', 'fft'), self.corr_res_empty)
        assert_equal(signal.correlate(self.x, self.empty2, 'same', 'fft'), self.corr_res_empty)
        assert_equal(signal.correlate(self.x, self.empty2, 'valid', 'fft'), self.corr_res_empty)

    def test_modes_methods(self):
        """
        check if the convolve function will hold write result with
        different modes and methods
        """
        assert_almost_equal(signal.convolve(self.x, self.h, 'full', 'fft'), self.conv_res_full)
        assert_almost_equal(signal.convolve(self.x, self.h, 'same', 'fft'), self.conv_res_same)
        assert_almost_equal(signal.convolve(self.x, self.h, 'valid', 'fft'), self.conv_res_valid)
        assert_almost_equal(signal.convolve(self.x, self.h, 'full', 'direct'), self.conv_res_full)
        assert_almost_equal(signal.convolve(self.x, self.h, 'same', 'direct'), self.conv_res_same)
        assert_almost_equal(signal.convolve(self.x, self.h, 'valid', 'direct'), self.conv_res_valid)

        """
        check if the correlate function will hold write result with
        different modes and methods
        """
        assert_almost_equal(signal.correlate(self.x, self.h, 'full', 'fft'), self.corr_res_full)
        assert_almost_equal(signal.correlate(self.x, self.h, 'same', 'fft'), self.corr_res_same)
        assert_almost_equal(signal.correlate(self.x, self.h, 'valid', 'fft'), self.corr_res_valid)
        assert_almost_equal(signal.correlate(self.x, self.h, 'full', 'direct'), self.corr_res_full)
        assert_almost_equal(signal.correlate(self.x, self.h, 'same', 'direct'), self.corr_res_same)
        assert_almost_equal(signal.correlate(self.x, self.h, 'valid', 'direct'), self.corr_res_valid)

    def main(self):
        unittest.main()

