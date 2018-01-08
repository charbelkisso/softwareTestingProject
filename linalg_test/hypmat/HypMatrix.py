
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
import scipy.linalg


class HyperbolicMatrix_Test(unittest.TestCase):
    """
    These tests are created to test sine,cosine and tangent-matrix functions from linalg from scipy
    """

    def setUp(self):
        """
        :Def Arrays:
        """
        self.a = np.array([[1.0, 2.0],[-1.0, 3.0]])
        self.one_zero = np.array([[1.0, 2.0],[-1.0, 0.0]])
        self.array_zero = np.array([[1.0, 2.0],[0.0, 0.0]])
        self.all_zero = np.array([[0, 0],[0, 0]])


        """
        :Def Expected Sine Results:
        """
        self.sin_res = np.array([[1.89,-0.97],
                                [0.48,0.91]])
        self.sin_res_one_zero = np.array([[ 1.54,2.31],
                                         [-1.16,0.39]])
        self.sin_res_array_zero = np.array([[0.84,1.68],
                                           [0,0]])
        self.sin_res_all_zero = np.array([[0,0],
                                [0,0]])

        """
        :Def Expected Cosine Results:
        """
        self.cos_res =np.array([[ 0.42, -2.13],
                                [ 1.06, -1.71]])
        self.cos_res_one_zero= np.array([[1.45,-1.26],
                                         [0.63,2.08]])
        self.cos_res_array_zero= np.array([[0.54,-0.92],
                                           [0,1]])
        self.cos_res_all_zero= np.array([[1,0],
                                         [0,1]])

        """
        :Def Expected tangent Results:
        """
        self.tan_res= np.array([[-1.41,2.33],
                                [-1.17,0.92]])
        self.tan_res_one_zero= np.array([[0.46,1.39],
                                         [-0.7,-0.24]])
        self.tan_res_array_zero= np.array([[1.56,3.11],
                                           [0,0]])
        self.tan_res_all_zero= np.array([[0,0],
                                         [0,0]])

    def test_sinm(self):

        """
        this test unit meant to test sinm function from scypi.linalg

        function description:
        ---------------------
        Compute the matrix sine
        """
        assert_almost_equal(scipy.linalg.sinm(self.a), self.sin_res, decimal=2)

        """
        this test unit meant to test sinm function from scypi.linalg

        function description:
        ---------------------
        Compute the matrix sine with one zero element
        """
        assert_almost_equal(scipy.linalg.sinm(self.one_zero), self.sin_res_one_zero, decimal=2)

        """
        this test unit meant to test sinm function from scypi.linalg

        function description:
        ---------------------
        Compute the matrix sine with one zero array
        """
        assert_almost_equal(scipy.linalg.sinm(self.array_zero), self.sin_res_array_zero, decimal=2)

        """
        this test unit meant to test sinm function from scypi.linalg

        function description:
        ---------------------
        Compute the matrix sine with all zero arrays
        """
        assert_equal(scipy.linalg.sinm(self.all_zero), self.sin_res_all_zero)





    def test_cosm(self):
        """
        this test unit meant to test cosm function from scypi.linalg

        function description:
        ---------------------
        Compute the matrix cosine
        """
        assert_almost_equal(scipy.linalg.cosm(self.a), self.cos_res, decimal=2)

        """
        this test unit meant to test cosm function from scypi.linalg

        function description:
        ---------------------
        Compute the matrix cosine with one zero element
        """
        assert_almost_equal(scipy.linalg.cosm(self.one_zero), self.cos_res_one_zero, decimal=2)

        """
        this test unit meant to test cosm function from scypi.linalg

        function description:
        ---------------------
        Compute the matrix cosine with one zero array
        """
        assert_almost_equal(scipy.linalg.cosm(self.array_zero), self.cos_res_array_zero, decimal=2)

        """
        this test unit meant to test cosm function from scypi.linalg

        function description:
        ---------------------
        Compute the matrix cosine with all zero arrays
        """
        assert_equal(scipy.linalg.cosm(self.all_zero), self.cos_res_all_zero)



    def test_tanm(self):

        """
        this test unit meant to test tanm function from scypi.linalg

        function description:
        ---------------------
        Compute the matrix tangent
        """
        assert_almost_equal(scipy.linalg.tanm(self.a), self.tan_res, decimal=2)

        """
        this test unit meant to test tanm function from scypi.linalg

        function description:
        ---------------------
        Compute the matrix tangent with one zero element
        """
        assert_almost_equal(scipy.linalg.tanm(self.one_zero), self.tan_res_one_zero, decimal=2)



        """
        this test unit meant to test tanm function from scypi.linalg

        function description:
        ---------------------
        Compute the matrix tangent with one zero array
        """
        assert_almost_equal(scipy.linalg.tanm(self.array_zero), self.tan_res_array_zero, decimal=2)

        """
        this test unit meant to test tanm function from scypi.linalg

        function description:
        ---------------------
        Compute the matrix tangent with all zero arrays
        """
        assert_equal(scipy.linalg.tanm(self.all_zero), self.tan_res_all_zero)

    def main(self):
        unittest.main()

    def runTest(self):
        pass

