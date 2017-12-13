import unittest
from scipy.ndimage import gaussian_filter
import numpy as np
from numpy.testing import * 



class Gaussian_Filter_Test(unittest.TestCase):

    """
    this class is testing the gaussian filter

    Function name: gaussian_filter()

    Gaussian filter used for reduce noises into images. the input is presented as nd array
    with n dimensions and the output is a nd array correspond to the filtered input with same 
    dimensions.

    Test Plan:
    =========
    1. testing the return result of the function. 
        a. on one dimension input shall return one dimension output.
        b. on two dimension input shall return two dimension output.
        c. on 3X3X3 dimension input array
    2. testing exceptions when the function is called without parameters.
    3. testing truncate option by counting the non-zero items in the result array.
    4. testing where the sigma is set to zero (No filtering), input shall be equal to output


    since the output value of this function is an approximation of the gaussian second deviation
    to calculated by hand and compare result is quit difficult and it doesn't give any testing benefits
    """

    def setUp(self):
        self.x = [11, 4, 5, 12, 6, 8, 25]
        self.y = np.arange(50, step=2).reshape(5, 5)
        self.z = (5 * np.random.random_sample(27)).reshape(3, 3, 3) 
        self.res = np.array([])
        
    
    def test_gaussian_output_shape(self):
        """ Test Plan 1

        this test will pass if the dimension of the input array
        and the output array is identical equal
        """
        self.assertEqual(gaussian_filter(self.x, 1).shape, (7,))
        self.assertEqual(gaussian_filter(self.y, 1).shape, (5, 5))
        self.assertEqual(gaussian_filter(self.z, 1).shape, (3, 3, 3))

    
    @unittest.expectedFailure
    def test_gaussian_no_input(self):
        """ Test Plan 2

        this test will pass if the function raise an error 
        corresponding to the missing argument
        """
        self.assertEqual(gaussian_filter(self.a), seld.res)
        self.assertEqual(gaussian_filter(), self.res)

    
    def test_gaussian_truncate(self):
        """ Test Plan 3

        this test will pass if the non-zero output element are correct
        the equation is non-zeros = ( 2 * ( sigma * truncate ) + 1 ) ^ num_of_axis
        whene the evaluation of the equation exceeds the size of the array all the elemnt will be non-zero
        """

        # array with 2 axix
        input_array = np.zeros((100, 100), np.float)
        input_array[50, 50] = 1
        # sigma = 5 , truncate = 2 
        non_zero = (gaussian_filter(input_array, sigma= 5, truncate=2) > 0.0).sum()
        non_zero_calc = (2* int(5 * 2) + 1)**2
        self.assertEqual(non_zero, non_zero_calc)
        non_zero = (gaussian_filter(input_array, sigma= 3.5, truncate=5.5) > 0.0).sum()
        non_zero_calc = (2* int(3.5 * 5.5) + 1)**2
        self.assertEqual(non_zero, non_zero_calc)

        #array with 3 axix

        input_array = np.zeros((100, 100, 100), np.float)
        input_array[50, 50, 50] = 1
        # sigma = 5 , truncate = 2 
        non_zero = (gaussian_filter(input_array, sigma= 5, truncate=2) > 0.0).sum()
        non_zero_calc = (2* int(5 * 2) + 1)**3
        self.assertEqual(non_zero, non_zero_calc)
        # sigma = 3.5 , truncate = 5.5
        non_zero = (gaussian_filter(input_array, sigma= 3.5, truncate=5.5) > 0.0).sum()
        non_zero_calc = (2* int(3.5 * 5.5) + 1)**3
        self.assertEqual(non_zero, non_zero_calc)

  
    def test_gaussian_sigma_influent(self):
        """ Test Plan 4

        this test will pass if the input and the output are the same
        """

        self.res = gaussian_filter(self.x, 0)
        assert_array_almost_equal(self.res, self.x)
        self.res = np.array([])

    def main(self):
        unittest.main()

    