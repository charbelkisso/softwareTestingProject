import unittest
from scipy.ndimage import gaussian_filter
from scipy import misc
import numpy as np
from numpy.testing import * 



class Gaussian_Filter_Test(unittest.TestCase):

    """
    this class is testing the gaussian filter

    Function: name gaussian_filter()

    Test Plan:
    =========

    1. testing the return result of the function 
        a. on one dimension
        b. on two dimention
        the result test will 





    """
    def setUp(self):
        self.x = [11, 4, 5, 12, 6, 8]
        self.y = np.arange(50, step=2)
        self.res1 = []
        

    def test_filter_result_with_sigma(self):
        self.res_test_1 = gaussian_filter(self.x, 1)
        self.res_test_3 = gaussian_filter(self.x, 3)
        self.res_test_8 = gaussian_filter(self.x, 8)
        
    def test_2d(self):
        return

    def main():
        unittest.main()

if __name__ == '__main__':
    Gaussian_Filter_Test.main()
    