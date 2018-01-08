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
from scipy import misc, linalg, imag, signal, integrate, stats
import warnings as w
import scipy


class Scipy_Test(unittest.TestCase):


    def setUp(self):
        """
        :return:
        """
        w.filterwarnings('ignore')
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

        
        self.R1 = 2
        self.H1 = 3
        self.ConeVolume1 = (1 / 3) * math.pi * (self.R1**2) * self.H1

        self.R2 = 1
        self.H2 = 0
        self.ConeVolume2 = (1 / 3) * math.pi * (self.R2**2) * self.H2

        self.R3 = 1000
        self.H3 = 1000
        self.ConeVolume3 = (1 / 3) * math.pi * (self.R3**2) * self.H3


        self.tplquad_func0 = lambda z, y, x: y + x + z



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
    Tripple Quad Testing Calculating the valolume of cube
    """
#   Calculating the volume of cube
    def test_tplquad_Volume_of_Cube_function(self):
        assert_almost_equal(integrate.tplquad(lambda z, y, x: 1, 0, 6, lambda x:0, lambda x:6, lambda x,y:0, lambda x,y:6),
                          [6*6*6, 0.0], decimal = 6)
        assert_almost_equal(integrate.tplquad(lambda z, y, x: 1, -3, 3, lambda x:-3, lambda x:3, lambda x,y:-3, lambda x,y:3),
                          [6*6*6, 0.0], decimal = 6)


# Making any one side of cube equals to zero
    def test_tplquad_Volume_of_Cube_With_One_Side_Zero(self):
        assert_almost_equal(integrate.tplquad(lambda z, y, x: 1, 0, 0, lambda x:0, lambda x:6, lambda x,y:0, lambda x,y:6),
                          [0, 0.0], decimal = 6)
        assert_almost_equal(integrate.tplquad(lambda z, y, x: 1, -3, 3, lambda x:0, lambda x:0, lambda x,y:-3, lambda x,y:3),
                          [0, 0.0], decimal = 6)
        assert_almost_equal(integrate.tplquad(lambda z, y, x: 1, 3, -3, lambda x:3, lambda x:-3, lambda x,y:0, lambda x,y:0),
                          [0, 0.0], decimal = 6)

#   Calculating the volume of cone using tripple integral
#   Now we have functions in the limits
#   Rather than constants opposed to caluclation of volume of Cube
    def test_tplquad_Volume_of_Cone_Small(self):
#       Small Numbers
        assert_almost_equal(integrate.tplquad(lambda z, y, x: 1, -self.R1, self.R1, 
                            lambda x: -math.sqrt(self.R1**2 - x**2), lambda x: math.sqrt(self.R1**2 - x**2),
                            lambda x,y: (self.H1 / self.R1) * math.sqrt(x**2 + y**2), lambda x,y: self.H1),
                          [self.ConeVolume1, 0.0], decimal = 6)
#       Making Height of the cone equals to zero
    def test_tplquad_Volume_of_Cone_Zero(self):
        assert_almost_equal(integrate.tplquad(lambda z, y, x: 1, -self.R2, self.R2, 
                            lambda x: -math.sqrt(self.R2**2 - x**2), lambda x: math.sqrt(self.R2**2 - x**2),
                            lambda x,y: (self.H2 / self.R2) * math.sqrt(x**2 + y**2), lambda x,y: self.H2),
                          [self.ConeVolume2, 0.0], decimal = 6)
#       Large Numbers
    def test_tplquad_Volume_of_Cone_Large(self):
        assert_almost_equal(integrate.tplquad(lambda z, y, x: 1, -self.R3, self.R3, 
                            lambda x: -math.sqrt(self.R3**2 - x**2), lambda x: math.sqrt(self.R3**2 - x**2),
                            lambda x,y: (self.H3 / self.R3) * math.sqrt(x**2 + y**2), lambda x,y: self.H3),
                          [self.ConeVolume3, 0.0], decimal = -1)

    """
    Tripple Quad Testing Simple Summing Function
    """
    def test_tplquad_Simple_function(self):
        assert_almost_equal(integrate.tplquad(self.tplquad_func0, 0, 6, lambda x:0, lambda x:6, lambda x,y:0, lambda x,y:6),
                          [1944, 0.0], decimal = 6)
        assert_almost_equal(integrate.tplquad(self.tplquad_func0, -3, 3, lambda x:-3, lambda x:3, lambda x,y:-3, lambda x,y:3),
                          [0, 0.0], decimal = 6)

    """
    Moments
    """
    def test_different_order_moments_with_default_arguments(self):
        assert_almost_equal(stats.moment([1, 2, 3], moment = 0),
                          1.0, decimal = 9)
        assert_almost_equal(stats.moment([1, 2, 3], moment = 1),
                          0.0, decimal = 9)
        assert_almost_equal(stats.moment([1, 2, 3], moment = 2),
                          0.6666666666, decimal = 9)
        assert_almost_equal(stats.moment([1, 2, 3], moment = 1000),
                          0.6666666666, decimal = 9)
        assert_almost_equal(stats.moment([1, 2, 3], moment = -1),
                          0.6666666666, decimal = 9)
        assert_almost_equal(stats.moment([1, 2, 3], moment = -1000),
                          0.6666666666, decimal = 9)


    def test_moments_with_NAN(self):
        assert_almost_equal(stats.moment([1, 2, np.nan, 3], moment = 2, axis = 0, nan_policy = 'omit'),
                          0.6666666666, decimal = 9)
        assert_almost_equal(stats.moment([1, 2, np.nan, 3], moment = 2, axis = 0, nan_policy = 'propagate'),
                          np.nan, decimal = 9)
        assert_almost_equal(stats.moment([1, 2, np.nan, 3], moment = 2),
                          np.nan, decimal = 9)

    def test_moments_with_multi_dimension_array(self):
        assert_almost_equal(stats.moment([[1, 2, 3], [2,4,6]], moment = 2, axis = 1),
                          [0.666666666, 2.666666666], decimal = 9)
        assert_almost_equal(stats.moment([[1, 2, 3], [2,4,6]], moment = 2, axis = 0),
                          [0.25, 1.0, 2.25], decimal = 9)


if __name__ == '__main__':

    unittest.main()
