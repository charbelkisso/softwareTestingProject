import numpy as np
from scipy import linalg
import unittest
from numpy import random
from numpy.testing import *

class Case_Generator():

    A = None
    B = None
    shape = None
    expected_result = None

    def __init__(self):
        pass

    def buildCases(self):

        self.shape = random.randint(2,8)

        self.A = random.rand(self.shape, self.shape)
        self.B = random.rand(self.shape, 1)

        inv_a = linalg.inv(self.A)

        self.expected_result = inv_a.dot(self.B)

        return self


class Test_Solve(unittest.TestCase):

    def setUp(self):

        self.randomCase = Case_Generator()      # used to generate random test cases
        """
            this calculated by hand
        """
        self.case01 = Case_Generator()          # specific values
        self.case01.shape = 3
        self.case01.A = np.array([
            [1, 3, 5],
            [2, 5, 1],
            [2, 3, 8]
        ])
        self.case01.B = np.array([
            [10],
            [8],
            [3]
        ])

        self.case01.expected_result = np.array([
            [-9.28],
            [5.16],
            [0.76]
        ])


        """
            this from the property solve(I, B) = B
            where I is the identity matrix
        """
        self.case02 = Case_Generator()
        self.case02.shape = 3
        self.case02.A = np.identity(3)
        self.case02.B = np.array([
            [3],
            [5],
            [1]
        ])

        self.case02.expected_result = self.case02.B

    def test_solver_output_value(self):

        res = linalg.solve(self.case01.A, self.case01.B)
        assert_almost_equal(self.case01.expected_result, res, decimal=3)

    def test_run_random_values(self):

        for i in range(0,100):
            self.randomCase = self.randomCase.buildCases()
            res = linalg.solve(self.randomCase.A, self.randomCase.B)
            assert_almost_equal(self.randomCase.expected_result, res, decimal=3)
            self.assertEqual(self.randomCase.expected_result.shape, res.shape)

    def test_feature(self):

        res = linalg.solve(self.case02.A, self.case02.B)
        assert_almost_equal(self.case02.expected_result, res, decimal=3)


    def runTest(self):
        pass
    def main(self):
        unittest.main()