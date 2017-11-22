from numpy import *
import numpy as np
from scipy import signal, misc
import matplotlib.pyplot as plt
import unittest

class test_scipy(unittest.TestCase):

    def test_convolve(self):

        x = np.array([1.0, 2.0, 3.0])
        h = np.array([0.0, 1.0, 0.0, 0.0, 0.0])

        print(x)
        self.assertEquals(signal.convolve(x, h),
                          [ 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0])

if __name__ == '__main__':

    unittest.main()


