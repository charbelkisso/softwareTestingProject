import unittest
from scipy import linalg
import numpy as np
from numpy.testing import assert_raises
import warnings as w

class Test_subspace_angles(unittest.TestCase):

	## The function compute subspace angles between 2 matricies
	## Tests try it on matricies with orthogonal and non-orthogonal columns,
	## as well as computing subspace angle of a matrix to itself

	def setUp(self):
		w.filterwarnings('ignore')
		#Hadamard matrix with orthogonal columns
		self.A = linalg.hadamard(4)
		self.A2 = linalg.hadamard(8)
		self.P = np.array([90., 90.])
		self.Z = np.array([True, True], dtype=np.bool)
		self.X = np.random.RandomState(0).randn(4, 3)
		self.zero = np.empty(shape=(0, 0))

	def testdefault(self):
		#angle between orthogonal columns
		np.testing.assert_almost_equal(np.rad2deg(linalg.subspace_angles(self.A[:,:2], self.A[:,2:])), self.P)
		#Subspace angle of a matrix to itself should be zero (array([True, True])), but a matrix with orthogonal columns size 4 fail the test
		np.testing.assert_array_equal(linalg.subspace_angles(self.A[:,:2], self.A[:,:2]) <= 2*np.finfo(float).eps, self.Z)
		np.testing.assert_array_equal(linalg.subspace_angles(self.A2[:,:2], self.A2[:,:2]) <= 2*np.finfo(float).eps, self.Z)
		#angle between non-orthogonal columns
		np.testing.assert_almost_equal(np.rad2deg(linalg.subspace_angles(self.X[:, :2], self.X[:, [2]])), np.array([55.83245]))

	def testinvalid(self):
		#self.assertRaisesRegexp(ValueError, "expected 2D array, got shape ()", linalg.subspace_angles, -1, 0)
		#self.assertRaisesRegexp(ValueError, "On entry to DGESDD parameter number 5 had an illegal value", linalg.subspace_angles, self.zero, self.zero)
		assert_raises(ValueError, linalg.subspace_angles, -1, 0)
		assert_raises(ValueError, linalg.subspace_angles, self.zero, self.zero)

	def main(self):
		unittest.main()

	def runTest(self):
		pass
