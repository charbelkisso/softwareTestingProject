import unittest
from scipy.ndimage import laplace, correlate1d
import numpy as np
from numpy.testing import *


class Laplace_Test(unittest.TestCase):

    """
    This class is testing the Laplace filter

    Function name: laplace()

    Laplace filter is used to detect the density in the pixel of an image
    by make it more significant from the pixels that surrounding it.
    One of the most used application for the laplace filter in image processing
    for edge detection.


    Test Plan:
    ==========

    1. testing the output given from the laplace filter.
    2. testing the functionality of the laplace over many data types.
    3. testing the shape of the output array, must be equal to the input array.

    """
    def setUp(self):
        np.random.seed(24568)
        self.input_array_2d = np.random.randint(0, 25, (100, 100))
        self.input_array_3d = np.random.randint(0, 25, (100, 100, 100))
        pass


    def test_laplace_output_2d(self):
        """
        Test plan 1

        this test is testing the output of the laplace filter
        by self calculating the 2nd derivative order of the given
        array and compare it with the laplace output
        """
        res_ax = correlate1d(self.input_array_2d, [1, -2, 1], 0)  #first order derivative for the axis 1
        res_ax += correlate1d(self.input_array_2d, [1, -2, 1], 1) #first order derivative for the axis 2
        result = laplace(self.input_array_2d)
        assert_array_almost_equal(result, res_ax)

        #same approach for 3D arrays

        res_ax = correlate1d(self.input_array_3d, [1, -2, 1], 0)  #first order derivative for the axis 1
        res_ax += correlate1d(self.input_array_3d, [1, -2, 1], 1)  # first order derivative for the axis 2
        res_ax += correlate1d(self.input_array_3d, [1, -2, 1], 2)  # first order derivative for the axis 3
        result = laplace(self.input_array_3d)
        assert_array_almost_equal(result, res_ax)



    def test_laplace_types(self):

        """
        Test plane 2:

        this test is testing the quality of types for both input and output
        """
        types = [np.int32, np.float32, np.float64]

        for type in types:
            t_input = np.asarray([
                [15, 2, 6],
                [4, 25, 9],
                [5, 2, 6]
            ], type)
            t_output = laplace(t_input)
            self.assertEqual(type, t_output.dtype)




    def test_laplace_shape(self):

        """
        Test plan 3:

        making sure that the shape of the output array matches the shape of the input array

        """

        in_shape = self.input_array_2d.shape
        out_shape = laplace(self.input_array_2d).shape
        self.assertEqual(in_shape, out_shape)
        in_shape = self.input_array_3d.shape
        out_shape = laplace(self.input_array_3d).shape
        self.assertEqual(in_shape, out_shape)

    def runTest(self):
        pass

    def main(self):
        unittest.main()
