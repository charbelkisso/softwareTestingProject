

"""
Software Testing Project - Scipy TestUnit

Participant:

- Charbel Kisso
- add you names

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


class Scipy_Test(unittest.TestCase):


    """
    Scope to define global variable are used in the test unit
    """
    def setUp(self):

        return


    def test_convolve(self):

        x = np.array([1.0, 2.0, 3.0])
        h = np.array([0.0, 1.0, 0.0, 0.0, 0.0])

        assert_equal(signal.convolve(x, h),
                          [0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0])


if __name__ == '__main__':

    unittest.main()

