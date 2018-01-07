import unittest
from scipy import linalg
import numpy as np
from numpy.testing import assert_raises
import warnings as w

class Test_Pascal(unittest.TestCase):

## Function return pascal matrix size n x n
## Stress testing on invalid or negative n, 
## as well as kind of pascal matrix

	def setUp(self):
		w.filterwarnings('ignore')
		self.pascalarr = np.array([[1, 1, 1],[1, 2, 3],[1, 3, 6]], dtype=np.uint64)
		self.pascalobj = np.array(112186277816662845432, dtype=np.object)
		self.pascalarrzero = np.empty(shape=(0, 0))
		self.pascalarrlower = np.array([[1, 0, 0],[1, 1, 0],[1, 2, 1]], dtype=np.uint64)
		self.pascalarrupper = np.array([[1, 1, 1],[0, 1, 2],[0, 0, 1]], dtype=np.uint64)

	def testdefault(self):
		np.testing.assert_array_equal(linalg.pascal(3),self.pascalarr)
		np.testing.assert_array_equal(linalg.pascal(36)[-1,-1],self.pascalobj)
		np.testing.assert_array_equal(linalg.pascal(3,kind='symmetric'), self.pascalarr)
		np.testing.assert_array_equal(linalg.pascal(3,kind='lower'), self.pascalarrlower)
		np.testing.assert_array_equal(linalg.pascal(3,kind='upper'), self.pascalarrupper)
		np.testing.assert_array_equal(linalg.pascal(3, exact='True'), self.pascalarr)

	def testzero(self):
		np.testing.assert_array_equal(linalg.pascal(0),self.pascalarrzero)

	def testinvalid(self):
		assert_raises(ValueError, linalg.pascal, -1)
		assert_raises(ValueError, linalg.pascal, 2, "kind=qwerty")

	def main(self):
		unittest.main()


