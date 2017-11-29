import unittest
import math
import numpy as np

from numpy.testing import assert_equal, assert_almost_equal

from scipy import misc, linalg, imag, signal, integrate

import scipy



class Test_Convolve(unittest.TestCase):

	
	def test_convolve(self):
		"""
		this test unit meant to test convolve function from scypi.signal
		function description:
		---------------------
		Todo: add function description
		:return: this function has no return is meant to be a test case for
		the convolve function.
		"""
		x = np.array([1.0, 2.0, 3.0])
		h = np.array([0.0, 0.5, 0.5])

		self.assertEqual(signal.convolve(x, h),
		                  [0.0, 0.5, 1.5, 2.5, 1.5])

		assert_equal(len(signal.convolve(x, h)), 5)
		
if __name__ == '__main__':

    unittest.main()
