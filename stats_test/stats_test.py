from scipy import stats
import numpy as np
from numpy.testing import assert_raises, assert_almost_equal
import random
import math
import unittest
import warnings as w

class Test_Stats(unittest.TestCase):

    def setUp(self):
        w.filterwarnings('ignore')

    def math_pdf(self, x):
        """
        this function is obtained from the definition of pdf
        used to calculate pdf mathimatecly in order to test the actual functiom

        """
        y = math.   pi)
        return y

    def math_cdf(self, x):
        """
        this function is obtained from the definition of cdf
        used to calculate pdf mathimatecly in order to test the actual functiom
        """
        return (1.0+ math.erf(x/math.sqrt(2.0)))/2.0

    def generate_random(self, count):
        """
        generate random integers corresponding to the given count
        :param count: integer
        :return: random numbers stored nd array
        """
        out = np.ndarray(count)

        for i in range(0, count):
            out.put(i,random.randint(1,25))

        return out

    def test_pdf_single_value(self):
        """
        testing the pdf with random values
        :return:
        """
        for i in range(0,10):
            x = self.generate_random(1)
            y_math = self.math_pdf(x)
            y_pdf = stats.norm.pdf(x)
            self.assertAlmostEqual(y_math,y_pdf)

    def test_pdf_array(self):
        array_len = 3
        y_math = np.ndarray(array_len)
        for i in range (0,10):
            x = self.generate_random(array_len)
            for j in range(0, array_len):
                y_math.put(j,self.math_pdf(x[j]))
            y_pdf = stats.norm.pdf(x)
            assert_almost_equal(y_math,y_pdf,decimal=6)

    def test_pdf_with_loc_scale(self):
        """
        testing the pdf with loc and scale argument
        from the library manual
        stats.norm.pdf(y, loc, scale) is equal to
        norm.pdf(y) / scale where y = (x-loc)/scale
        """

        array_len = 3
        y_math = np.ndarray(array_len)
        for i in range(0, 10):
            loc = self.generate_random(1)
            scale = self.generate_random(1)
            x = self.generate_random(array_len)
            for j in range(0, array_len):
                res_x = (x[j] - loc) / scale
                y_math.put(j,self.math_pdf(res_x)/scale)
            y_pdf = stats.norm.pdf(x, loc, scale)
            assert_almost_equal(y_math,y_pdf,decimal=6)

    def test_cdf_single_value(self):
        for i in range(0,10):
            x = self.generate_random(1)
            y_math = self.math_cdf(x[0])
            y_pdf = stats.norm.cdf(x[0])
            self.assertAlmostEqual(y_math,y_pdf)

    def test_cdf_array(self):
        array_len = 3
        y_math = np.ndarray(array_len)
        for i in range(0, 10):
            x = self.generate_random(array_len)
            for j in range(0, array_len):
                y_math.put(j, self.math_cdf(x[j]))
            y_pdf = stats.norm.cdf(x)
            assert_almost_equal(y_math, y_pdf, decimal=6)

    def test_cdf_loc_scale(self):
        array_len = 3
        y_math = np.ndarray(array_len)
        for i in range(0, 10):
            loc = self.generate_random(1)
            scale = self.generate_random(1)
            x = self.generate_random(array_len)
            for j in range(0, array_len):
                res_x = (x[j] - loc) / scale
                y_math.put(j, self.math_cdf(res_x))
            y_pdf = stats.norm.cdf(x, loc, scale)
            assert_almost_equal(y_math, y_pdf, decimal=6)

    def test_moment_out(self):
        """
        this test will test the moment by observing output
        the test case values calculated by hand according to the library reference
        """

        case01 = [1,2,3,4]
        case02 = [2,6,7,8]

        y_case01 = 1.25
        y_case02 = 5.1875

        y1 = stats.moment(case01,2)
        y2 = stats.moment(case02,2)

        self.assertEqual(y_case01,y1)
        self.assertEqual(y_case02,y2)

    def test_moment_out_shape(self):

        for i in range(0, 10):
            shape = random.randint(0,10)
            moment = np.ndarray(shape)
            for j in range(0, shape):
                moment.put(j,random.randint(0,10))
            y = stats.moment([1,2,3,4],moment)
            self.assertEqual(y.shape, moment.shape)

    def test_wrong_parameters(self):
        """
        here we will try to give the function some wrong value
        """

        assert_raises(ValueError, stats.moment, [1,2,3,4], 2, nan_policy='2')
        assert_raises(ValueError, stats.moment, [1,2,3,4], 1.5)

    def main(self):
        unittest.main()

    def runTest(self):
        pass