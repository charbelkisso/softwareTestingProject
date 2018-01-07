import unittest
import numpy as np
from scipy import linalg
from numpy.testing import *
from numpy import random

class Test_Inv(unittest.TestCase):

    def setUp(self):
        """
        test cases:
            case1 : value in the input range
            case2 : identity matrix
            case3 : zero matrix
        :return:
        """

        self.case01 = np.array([
            [1, 2, 4],
            [3, 4, 5],
            [6, 8, 9]
        ])

        self.case02 = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        self.case03 = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])

    def test_inv_out(self):
        """
        case01: testing with case one and compare with calculated value
                the value calculated by hand
        case02: testing with the identity matrix the answer should the
                same as input

        :return:
        """

        res_case01 = np.array([
            [-2, 7, -3],
            [1.5, -7.5, 3.5],
            [0, 2, -1]
        ])

        res1 = linalg.inv(self.case01)
        res2 = linalg.inv(self.case02)

        assert_almost_equal(res_case01,res1)
        assert_equal(self.case02,res2)

    def test_inv_property(self):
        """
        testing the inverse function from it's property

        A * A^-1 = I

        since the identity matrix consist of integers we will ignore the floating point

        :return:
        """

        inv = linalg.inv(self.case01)
        id = self.case01.dot(inv)
        assert_almost_equal(self.case02, id, decimal=2)

    def test_inv_with_zeros(self):

        """
        the test function should throw exception since zero matrix has no inverse
        :return:
        """

        try:
            inv = linalg.inv(self.case03)
        except Exception as e:
            self.assertEqual(np.linalg.linalg.LinAlgError, e.__class__)

    def test_inv_out_shape(self):

        """
        random test to compare the dimension of both input and output
        :return:
        """

        for i in range(0, 10):
            shape = random.randint(2, 5)
            case04 = random.rand(shape,shape)
            res = linalg.inv(case04)
            self.assertEqual(case04.shape, res.shape)



    def runTest(self):
        pass

    def main(self):
        unittest.main()