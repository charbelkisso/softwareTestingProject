"""
importing important libraries for testing
"""
import unittest
import warnings as ws
import math
import numpy as np
"""
built-in assert function works with arrays and ndArrays
"""
from numpy.testing import assert_equal, assert_almost_equal, assert_raises
"""
functions to test:

ToDo: add function here to build test cases for it
"""
from scipy.optimize import least_squares

class Test_Least_Squares(unittest.TestCase):

	def setUp(self):
		ws.filterwarnings('ignore')
		self.res_test = [0.19280596, 0.19130423, 0.12306063, 0.13607247]
		self.res_unbound_self_test = ([  1.0117800995e+06,  -1.4075881180e+01,
							-4.5790318134e+07, -2.8564061567e+07])
		self.res_unbound_test = ([  1.9070429319e+05,  -1.4075872912e+01,
							-8.6307328042e+06, -5.3838624284e+06])
		self.u = np.array([4.0, 2.0, 1.0, 5.0e-1, 2.5e-1, 1.67e-1, 1.25e-1, 1.0e-1,
            				8.33e-2, 7.14e-2, 6.25e-2])
		self.y = np.array([1.957e-1, 1.947e-1, 1.735e-1, 1.6e-1, 8.44e-2, 6.27e-2,
            				4.56e-2, 3.42e-2, 3.23e-2, 2.35e-2, 2.46e-2])
		self.x0 = np.array([2.5, 3.9, 4.15, 3.9])
		self.x0_complex = np.array([1+2j, 3+4j, 5+6j])
		self.x0_ndim = np.array([
			[1, 2, 3],
			[4, 5, 6]
		])

	def model (self, x, u):

		return x[0] * (u ** 2 + x[1] * u) / (u ** 2 + x[2] * u + x[3])

	def fun(self, x, u, y):

		return self.model(x, u) - y

	def jac(self, x, u, y):
		J = np.empty((u.size, x.size))
		den = u ** 2 + x[2] * u + x[3]
		num = u ** 2 + x[1] * u
		J[:, 0] = num / den
		J[:, 1] = x[0] * u / den
		J[:, 2] = -x[0] * num * u / den ** 2
		J[:, 3] = -x[0] * num / den ** 2
		return J

	def jac_dunm(self, x, u, y):
		# this will not do anything
		return 0

	def test1_red(self):
		"""
		1: RED Test Path: This test will cover the parameters check at the begining and end up in error state.
		"""
		# invalid method name.
		assert_raises(ValueError, least_squares, self.fun, self.x0, method='not valid')
		# invalid jacobian.
		assert_raises(ValueError, least_squares, fun=self.fun, x0=self.x0, jac= 'self.jac_dunm')
		# invalid solver.
		assert_raises(ValueError, least_squares, self.fun, self.x0, tr_solver= 'not valid')
		# invalid loss.
		assert_raises(ValueError, least_squares, self.fun, self.x0, loss= 'not implemented')
		# invalid combination
		assert_raises(ValueError, least_squares, self.fun, self.x0, method= 'lm', loss= 'huber')
		# invalid vebrose
		assert_raises(ValueError, least_squares, self.fun, self.x0, verbose= 4)
		#invalid bound
		assert_raises(ValueError, least_squares, self.fun, self.x0, bounds=[0])
		#invalid max_nfev
		assert_raises(ValueError, least_squares, self.fun, self.x0, max_nfev= -4)
		#complex x0
		assert_raises(ValueError, least_squares, self.fun, self.x0_complex)
		#n-dim x0
		assert_raises(ValueError, least_squares, self.fun, self.x0_ndim)
		#invalid method with bound
		assert_raises(ValueError, least_squares, self.fun, self.x0, method='lm', bounds=(0, 100))
		#invalid bound low > high
		assert_raises(ValueError, least_squares, self.fun, self.x0, bounds=(100, 50))
		# x0 not in bound
		assert_raises(ValueError, least_squares, self.fun, self.x0, bounds=(0, 2))

	def test2_green(self):
		"""
		2: Green Test Path: This test will execute least_square function with default values
		(i.e call least square with the default parameters: method= trf , default jacobian (2-point),
		valid bounds, tr_solve=none, loss='linear' and verbose>=1)
		"""
		res = least_squares(self.fun, self.x0, bounds=(0, 100),args=(self.u, self.y))
		assert_almost_equal(res.x, self.res_test,decimal=3)

	def test3_blue(self):
		"""
		3: Blue Test Path: This test will cover:
		jacobian='cs'; default method(i.e 'trf'), jac_sparsity=None, verbose=2
		"""
		res = least_squares(self.fun, self.x0,jac='cs', bounds=(0, 100),loss='linear',method='trf', args=(self.u, self.y),jac_sparsity=None, verbose=2)
		assert_almost_equal(res.x, self.res_test,decimal=6)

	def test4_yellow(self):
		"""
		4: Yellow Test Path: This test will cover:
		jacobian='2-point', method=='lm', no bounds,tr_solve='exact', verbose=default,
		"""
		res = least_squares(self.fun, self.x0, jac='2-point',method='lm',tr_solver='exact', args=(self.u, self.y))
		assert_almost_equal(res.x, self.res_unbound_test,decimal=4)

	def test5_cyan(self):
		"""
		4: Yellow Test Path: This test will cover:
		callable jacobian=self, method=='dogbox',bounded,loss='linear',tr_solve='lsmr',jac_sparsity=array_like, verbose=default,
		"""
		res = least_squares(self.fun, self.x0, jac=self.jac,bounds=(0, 100), method='dogbox',loss='linear',tr_solver='lsmr', args=(self.u, self.y),jac_sparsity='sparse matrix')
		assert_almost_equal(res.x, self.res_test,decimal=3)

	def test5_grey(self):
		"""
		4: Grey Test Path: This test will cover:
		jacobian='3-point', method=default, bounded, loss='linear',tr_solve=none, verbose=default,
		"""
		res = least_squares(self.fun, self.x0, jac='3-point',bounds=(0, 100),loss='linear',tr_solver=None, args=(self.u, self.y))
		assert_almost_equal(res.x, self.res_test,decimal=3)

	def runTest(self):
		pass
	def main(self):
		unittest.main()


if __name__ == '__main__':
	unittest.main()

