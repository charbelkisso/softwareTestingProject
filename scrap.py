from scipy import signal
import numpy as np
import unittest
from numpy.testing import assert_equal


class Test_Convolve(unittest.TestCase):

    def setUp(self):

        self.empty1 = np.array([])
        self.empty2 = np.array([])
        self.empty_res = np.array([])

        self.x = np.array([1.0, 2.0, 3.0])
        self.h = np.array([0.0, 0.5, 0.5])
        self.res1 = [0.0, 0.5, 1.5, 2.5, 1.5]


    def test_default(self):
        """
        this is the defult mode of convolve function with correct value
        and expecting correct return

        """
        assert_equal(signal.convolve(self.x, self.h, 'fft'), self.res1)

    def test_empty_all_modes(self):

        """
        this test check if we could pass an empty array as input 

        """
        self.assertRaisesRegexp(ValueError, "a cannot be empty",
                        signal.convolve, self.empty1, self.empty2)
        self.assertRaisesRegexp(ValueError, "a cannot be empty",
                        signal.convolve, self.empty1, self.empty2, 'same')
        self.assertRaisesRegexp(ValueError, "a cannot be empty",
                        signal.convolve, self.empty1, self.empty2, 'valid')
        self.assertRaisesRegexp(ValueError, "a cannot be empty",
                        signal.convolve, self.empty1, self.empty2, 'direct')
        self.assertRaisesRegexp(ValueError, "a cannot be empty",
                        signal.convolve, self.empty1, self.empty2, 'fft')
        #self.empty_res = signal.convolve(self.empty1, self.empty2)

        

if __name__ == '__main__':
    unittest.main()