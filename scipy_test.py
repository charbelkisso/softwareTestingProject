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
import math
import numpy as np
"""
built-in assert function works with arrays and ndArrays 
"""
from numpy.testing import assert_equal, assert_almost_equal
"""
functions to test:
ToDo: add function here to build test cases for it  
"""
from scipy import misc, linalg, imag, signal, integrate

import scipy


class Scipy_Test(unittest.TestCase):


    def setUp(self):
        """
        :return:
        """
        self.quad_func0 = lambda x: 1
        self.quad_func1 = lambda x: np.exp(-x)
        self.quad_func2 = lambda x: x**2
        self.quad_func3 = lambda x,a: a*x

        self.dblquad_func0 = lambda y, x: 1
        self.dblquad_func1 = lambda y, x: 9 * (x**3) * (y**2)
        self.dblquad_func1_lo = lambda x: 2
        self.dblquad_func1_up = lambda x: 4

        self.dblquad_func2 = lambda t, x: np.exp(-x*t)/t**4
        self.dblquad_func2_lo = lambda x: 1
        self.dblquad_func2_up = lambda x: np.inf

        self.dblquad_func3 = lambda x, y: x*y
        self.dblquad_func3_lo = lambda x: 0
        self.dblquad_func3_up = lambda x: 1 - 2*x


        self.tplquad_func0 = lambda y, x, z: 1


#        self.dblquad_func2 = lambda x,y: x**2 + y**2
#        self.dblquad_func3 = lambda x,a: a*x

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
    """
    Quad Testing
    """
    def test_quad_Constant_function(self):
        assert_almost_equal(integrate.quad(self.quad_func0, 0, 10),
                          [10.0, 0.0], decimal = 9)
        assert_almost_equal(integrate.quad(self.quad_func0, -5, 5),
                          [10, 0.0], decimal = 9)
        assert_almost_equal(integrate.quad(self.quad_func0, 5, -5),
                          [-10, 0.0], decimal = 6)
#        assert_almost_equal(integrate.quad(self.quad_func0, 0, np.inf),
#                          [np.inf, 0.0], decimal = 6)

    def test_quad_different_functions(self):
        assert_almost_equal(integrate.quad(self.quad_func1, 0, np.inf),
                          [1.0, 0.0], decimal = 9)
        assert_almost_equal(integrate.quad(self.quad_func2, 0, 4),
                          [64/3, 0.0], decimal = 9)
        assert_almost_equal(integrate.quad(self.quad_func3, 0, 1, args = (1,)),
                          [0.5, 0.0], decimal = 9)

    def test_quad_different_functions_inf_limits(self):
        assert_almost_equal(integrate.quad(self.quad_func1, -np.inf, np.inf),
                          [np.inf, np.nan], decimal = 9)

    def test_quad_different_functions_negative_limits(self):
        assert_almost_equal(integrate.quad(self.quad_func2, -4, 0),
                          [64/3, 0.0], decimal = 9)
        assert_almost_equal(integrate.quad(self.quad_func3, -1, 0, args = (1,)),
                          [-0.5, 0.0], decimal = 9)

    def test_quad_different_functions_same_limits(self):
        assert_almost_equal(integrate.quad(self.quad_func1, 0, 0),
                          [0.0, 0.0], decimal = 9)
        assert_almost_equal(integrate.quad(self.quad_func2, 0, 0),
                          [0, 0.0], decimal = 9)
        assert_almost_equal(integrate.quad(self.quad_func3, 0, 0, args = (1,)),
                          [0, 0.0], decimal = 9)

    def test_quad_different_functions_missing_arguments(self):
        with self.assertRaises(Exception) as context:
           integrate.quad(self.quad_func3, 0, 1)
        self.assertEqual("<lambda>() missing 1 required positional argument: 'a'", str(context.exception))

    """
    Double Quad Testing
    """
    def test_dblquad_Constant_function(self):
        assert_almost_equal(integrate.dblquad(self.dblquad_func0, 0, 10, lambda x:0, lambda x:10),
                          [100, 0.0], decimal = 6)
        assert_almost_equal(integrate.dblquad(self.dblquad_func0, -5, 5, lambda x:-5, lambda x:5),
                          [100, 0.0], decimal = 6)
        assert_almost_equal(integrate.dblquad(self.dblquad_func0, 5, -5, lambda x:5, lambda x:-5),
                          [100, 0.0], decimal = 6)
#        assert_almost_equal(integrate.dblquad(self.dblquad_func0, 0, np.inf, lambda x:0, lambda x: np.inf),
#                          [np.inf, 0.0], decimal = 6)

    def test_dblquad_different_functions(self):
        # func1 with constant limits
        assert_almost_equal(integrate.dblquad(self.dblquad_func1, 1, 3, self.dblquad_func1_lo, self.dblquad_func1_up),
                          [3360, 0.0], decimal = 6)
        # func2 with infinty limits
        assert_almost_equal(integrate.dblquad(self.dblquad_func2, 0, np.inf, self.dblquad_func2_lo, self.dblquad_func2_up),
                          [0.25, 0.0], decimal = 6)
        # func3 with expression in the limit
        assert_almost_equal(integrate.dblquad(self.dblquad_func3, 0, 0.5, self.dblquad_func3_lo, self.dblquad_func3_up),
                          [1/96, 0.0], decimal = 6)

    def test_dblquad_with_all_limits_zero(self):
        # func1 with constant limits
        assert_almost_equal(integrate.dblquad(self.dblquad_func1, 0, 0, lambda x:0, lambda x:0),
                          [0, 0.0], decimal = 6)
#        # func2 with infinty limits func2 is nan at zero so skipping it
#        assert_almost_equal(integrate.dblquad(self.dblquad_func2, 0, 0, lambda x:0, lambda x:0),
#                          [0.0, 0.0], decimal = 6)
        # func3 with expression in the limit
        assert_almost_equal(integrate.dblquad(self.dblquad_func3, 0, 0.0, lambda x:0, lambda x:0),
                          [0, 0.0], decimal = 6)

    def test_dblquad_with_all_limits_same_minus_one(self):
        # func1 with constant limits
        assert_almost_equal(integrate.dblquad(self.dblquad_func1, -1, -1, lambda x:-1, lambda x:-1),
                          [0, 0.0], decimal = 6)
        # func2 with infinty limits
        assert_almost_equal(integrate.dblquad(self.dblquad_func2, -1, -1, lambda x:-1, lambda x:-1),
                          [0.0, 0.0], decimal = 6)
        # func3 with expression in the limit
        assert_almost_equal(integrate.dblquad(self.dblquad_func3, -1, -1, lambda x:-1, lambda x:-1),
                          [0, 0.0], decimal = 6)

    def test_dblquad_with_all_limits_same_plus_one(self):
        # func1 with constant limits
        assert_almost_equal(integrate.dblquad(self.dblquad_func1, 1, 1, lambda x:1, lambda x:1),
                          [0, 0.0], decimal = 6)
        # func2 with infinty limits
        assert_almost_equal(integrate.dblquad(self.dblquad_func2, 1, 1, lambda x:1, lambda x:1),
                          [0.0, 0.0], decimal = 6)
        # func3 with expression in the limit
        assert_almost_equal(integrate.dblquad(self.dblquad_func3, 1, 1, lambda x:1, lambda x:1),
                          [0, 0.0], decimal = 6)

    """
    Tripple Quad Testing
    """
    def test_tplquad_Constant_function(self):
        assert_almost_equal(integrate.tplquad(self.tplquad_func0, 0, 6, lambda x:0, lambda x:6, lambda x,y:0, lambda x,y:6),
                          [216, 0.0], decimal = 6)
        assert_almost_equal(integrate.tplquad(self.tplquad_func0, -3, 3, lambda x:-3, lambda x:3, lambda x,y:-3, lambda x,y:3),
                          [216, 0.0], decimal = 6)
        assert_almost_equal(integrate.tplquad(self.tplquad_func0, 3, -3, lambda x:3, lambda x:-3, lambda x,y:3, lambda x,y:-3),
                          [-216, 0.0], decimal = 6)
#        assert_almost_equal(integrate.tplquad(self.tplquad_func0, 0, np.inf, lambda x: 0, lambda x: np.inf, lambda x,y:0, lambda x,y: np.inf),
#                          [np.inf, 0.0], decimal = 6)

#        assert_almost_equal(integrate.quad(self.quad_func2, 0, 4),
#                          [64/3, 0.0], decimal = 9)
#        assert_almost_equal(integrate.quad(self.quad_func3, 0, 1, args = (1,)),
#                          [0.5, 0.0], decimal = 9)



if __name__ == '__main__':

    unittest.main()
