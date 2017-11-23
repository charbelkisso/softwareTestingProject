"""
Software Testing Project - Scipy TestUnit

Participant:

- Charbel Kisso
- Salman Akhtar Warraich
- add you names here

"""

"""
importing important libraries for testing
"""
import unittest
import numpy as np
"""
built-in assert function works with arrays and ndArrays 
"""
from numpy.testing import assert_equal
"""
functions to test:

ToDo: add function here to build test cases for it  
"""
from scipy import misc, linalg, imag, signal
import scipy


class Scipy_Test(unittest.TestCase):


    def setUp(self):
        """

        :return:
        """

        return


    def test_convolve(self):
        """
        this test unit meant to test convolve function from scypi.signal

        function description:
        ---------------------
        Todo: add function description

        :return: this function has no return is meant to be a test case for
        the convolve function.
        """
        x = np.array([1.0, 2.0, 3.0])
        h = np.array([0.0, 0.5, 0.5])

        assert_equal(signal.convolve(x, h),
                          [0.0, 0.5, 1.5, 2.5, 1.5])

        assert_equal(len(signal.convolve(x, h)), 5)



    def test_2(self):



"""
    def test_full(self):

        scipy.test('full')
"""

if __name__ == '__main__':

    unittest.main()

