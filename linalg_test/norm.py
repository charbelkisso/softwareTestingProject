import unittest
from scipy import linalg
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_raises
import warnings as w
class Test_norms(unittest.TestCase):

## The function computes matrix or vector norm 


	def setUp(self):
		w.filterwarnings('ignore')
		self.A = np.array([[1,2], [3,4]])

	def testdefault(self):
		self.assertAlmostEqual(linalg.norm(self.A), 5.4772256)
		#max column sum
		self.assertEqual(linalg.norm(self.A, 1), 6)
		#min column sum
		self.assertEqual(linalg.norm(self.A, -1), 4)
		self.assertEqual(linalg.norm(self.A, np.inf), 7)
		# frobenius norm by default
		self.assertAlmostEqual(linalg.norm(self.A, 'fro'), 5.4772256)
		assert_almost_equal(linalg.norm(self.A, axis=1), np.array([2.236068, 5.]))
		assert_equal(linalg.norm(self.A, axis=(0,1)), np.array(5.477225575051661))
		assert_equal(linalg.norm(self.A, keepdims=True), np.array(5.477225575051661))

	def testinvalid(self):
		# order of norm should be non-zero int, 'fro', inf, -inf

		assert_raises(ValueError, linalg.norm, self.A, 0)
		assert_raises(ValueError, linalg.norm, self.A, 3.4)
		assert_raises(ValueError, linalg.norm, self.A, 'flo')

		assert_raises(ValueError, linalg.norm, self.A, axis=3)
		assert_raises(ValueError, linalg.norm, self.A, axis=(1,2))
		assert_raises(ValueError, linalg.norm, self.A, axis=(1,1))

	def main(self):
		unittest.main()

	def runTest(self):
		pass
		

