import unittest
import numpy as np
from scipy import linalg
from numpy.testing import *

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
        the test function should throw exepssion since zero matrix has no inverse
        :return:
        """

        try:
            inv = linalg.inv(self.case03)
        except Exception as e:
            self.assertEqual(np.linalg.linalg.LinAlgError, e.__class__)

    def runTest(self):
        pass

"""
class myClass():

    def matrix_mult(self,a,b):

        result = [
            [0,0,0],
            [0,0,0],
            [0,0,0]
        ]
        for i in range (3):
            for j in range(3):
                for k in range (3):
                    result[i][j] += round(a[i][k]*b[k][j],3)

        return result

    def main(self):

        a = np.array([[1,2,4],[3,4,5],[6,8,9]])
        b = np.array([[3,5,6],[1,4,8],[4,2,6]])

        a_b = self.matrix_mult(a,b)

        c = linalg.inv(a)

        b_new = self.matrix_mult(a_b,c)

        i = a.dot(c)

        i_prime = self.matrix_mult(a,c)



        print "a= {}".format(a)
        print "b= {}".format(b)
        print "c= {}".format(b_new)
        print "i= {}".format(i_prime)


"""


if __name__ == '__main__':
   unittest.main()