import unittest
from scipy.ndimage import gaussian_filter
from scipy import misc
import numpy as np
from numpy.testing import * 



class Gaussian_Filter_Test(unittest.TestCase):

    """
    this class is testing the gaussian filter

    Function: name gaussian_filter()

    Guassian filter used for reduce noises into images. the input is presented as nd array
    with n dimensions and the output is a nd array correspond to the filtered input with same 
    dimensions.

    Test Plan:
    =========
    1. testing the return result of the function. 
        a. on one dimension input shall return one dimension output.
        b. on two dimention input shall return two dimension output.
        c. on 3X3X3 dimension input array
    2. testing exeptions whene the function is called without parameters.
    3. testing truncate option by counting the non-zero items in the result array.

    since the output value of this function is an approximation of the gaussian secon deviration
    to calculated by hand and compaire result is quit difficult and it doesn't give any testing benifits
    """

    def setUp(self):
        self.x = [11, 4, 5, 12, 6, 8, 25]
        self.y = np.arange(50, step=2).reshape(5, 5)
        self.z = (5 * np.random.random_sample(27)).reshape(3, 3, 3) 
        
        
    """ Test Plan 1
    """
    def test_gaussian_output_shape(self):
        self.assertEqual(gaussian_filter(self.x, 1).shape, (7,))
        self.assertEqual(gaussian_filter(self.y, 1).shape, (5, 5))
        self.assertEqual(gaussian_filter(self.z, 1).shape, (3, 3, 3))

    """ Test Plan 2
    """
    def test_gaussian_no_input(self):
        return

    def test_gaussian_truncate(self):
        return

    def main(self):
        unittest.main()

    