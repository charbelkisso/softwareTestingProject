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

from least_square import least_squares

class Test_Least_Squares(unittest.TestCase):

	def setUp(self):
		ws.filterwarnings('ignore')
		self.res_test = [0.19280596, 0.19130423, 0.12306063, 0.13607247]
		self.res_unbound_self_test = ([  1.0117800995e+06,  -1.4075881180e+01,
							-4.5790318134e+07, -2.8564061567e+07])
		self.res_test_lm = ([ 1.0117800995e+06, -1.4075881180e+01, -4.5790318134e+07,
        					-2.8564061567e+07])
		self.res_temp = [9.40418096e-04,1.71030814e+02,-5.20017963e+00,8.09010276e+00]
		self.res_temp_x = [0.2329967442,-0.5000017786,-0.0358009854,-0.2321008023]
		self.res_unbound_test = ([  1.9070429319e+05,  -1.4075872912e+01,
							-8.6307328042e+06, -5.3838624284e+06])

		self.res_sparse = [ 2.5 , 3.8472489507,  5.1140583493,  5.4414322215]
		self.u = np.array([4.0, 2.0, 1.0, 5.0e-1, 2.5e-1, 1.67e-1, 1.25e-1, 1.0e-1,
            				8.33e-2, 7.14e-2, 6.25e-2])

		self.u_s = np.array([4.0, 2.0, 1.0, 6.25e-2])

		self.y = np.array([1.957e-1, 1.947e-1, 1.735e-1, 1.6e-1, 8.44e-2, 6.27e-2,
            				4.56e-2, 3.42e-2, 3.23e-2, 2.35e-2, 2.46e-2])
		self.y_s = np.array([1.957e-1, 1.947e-1,2.35e-2, 2.46e-2])

		self.x0 = np.array([2.5, 3.9, 4.15, 3.9])
		self.x0_s = np.array([2.5, 3.9])

		self.x0_large = np.array([2,5,3,9,4,15,3,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5])
		self.x0_complex = np.array([1+2j, 3+4j, 5+6j])
		self.x0_ndim = np.array([
			[1, 2, 3],
			[4, 5, 6]
		])



	def cauchy_x(self, z, rho= np.empty((3, 11)), cost_only=False):

		rho[0] = np.log1p(z)
		if cost_only:
			return
		t = 1 + z
		rho[1] = 1 / t
		rho[2] = -1 / t ** 2
		return rho

	def cauchy_y(self, z, rho, cost_only=False):
		rho[0] = np.log1p(z)
		if cost_only:
			return
		t = 0* 1 + z
		rho[1] = 0* 1 / t
		rho[2] = 0* -1 / t ** 2

	def cauchy_z(self, z, rho= np.empty((3, 11)), cost_only=False):

		rho[0] = np.log1p(z)
		if cost_only:
			return z
		t = 1 + z
		rho[1] = 1 / t
		rho[2] = -1 / t ** 2
		return z


	def model (self, x, u):

		return x[0] * (u ** 2 + x[1] * u) / (u ** 2 + x[2] * u + x[3])

	def fun(self, x, u, y):

		return self.model(x, u) - y

	def fun_not_returning(self, x, u, y):
		pass

	def fun_ndim(self, x, u, y):

		return self.x0_ndim


	def jac(self, x, u, y):
		J = np.empty((u.size, x.size))
		den = u ** 2 + x[2] * u + x[3]
		num = u ** 2 + x[1] * u
		J[:, 0] = num / den
		J[:, 1] = x[0] * u / den
		J[:, 2] = -x[0] * num * u / den ** 2
		J[:, 3] = -x[0] * num / den ** 2
		return J

	def jac_s(self, x, u, y):
		J = np.empty((u.size, x.size))
		den = u ** 2 + x[2] * u + x[3]
		num = u ** 2 + x[1] * u
		J[:, 0] = 10* num / den
		J[:, 1] = 10* x[0] * u / den
		J[:, 2] = 10* -x[0] * num * u / den ** 2
		J[:, 3] = 10* -x[0] * num / den ** 2
		return J

	def jac_wrong(self, x, u, y):
		J = np.empty((u.size, x.size-1))
		den = u ** 2 + x[2] * u + x[3]
		num = u ** 2 + x[1] * u
		J[:, 0] = num / den
		J[:, 1] = x[0] * u / den
		J[:, 2] = -x[0] * num * u / den ** 2
		#J[:, 3] = -x[0] * num / den ** 2
		return J

	def jac_zero(self, x, u, y):
		J = np.empty((u.size, x.size))
		den = u ** 2 + x[2] * u + x[3]
		num = u ** 2 + x[1] * u
		J[:, 0] = 0 * num / den
		J[:, 1] = 0 * u / den
		J[:, 2] = 0 * num * u / den ** 2
		J[:, 3] = 0 * num / den ** 2
		return J

	def jac_sparse(self, x, u, y):
		J = np.empty((u.size, x.size))
		J = np.empty((u.size, x.size))
		den = u ** 2 + x[2] * u + x[3]
		num = u ** 2 + x[1] * u
		J[:, 0] = 0 * num / den
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
		# method = lm and jac sparsity = none
		assert_raises(ValueError, least_squares, self.fun, self.x0, method = 'lm', jac_sparsity = 'sparse matrix', bounds=(0, 2))
		# method = lm and jac sparsity != none and tr_solver = exact
		#assert_raises(ValueError, least_squares, self.fun, self.x0, method = 'lm', jac_sparsity = 'sparse matrix' , tr_solver = 'exact', bounds=(0, 2))
		# loss function is none
		assert_raises(ValueError, least_squares, self.fun, self.x0, jac='3-point', bounds=(0, 100), loss=None, tr_solver=None,
						args=(self.u, self.y))
		# wrong shape of jacobian
		assert_raises(ValueError, least_squares, self.fun, self.x0, jac=self.jac_wrong,bounds=(0, 100), method='dogbox',loss='linear',tr_solver='lsmr', args=(self.u, self.y),jac_sparsity='sparse matrix')

		assert_raises(ValueError, least_squares, self.fun, self.x0, jac=self.jac_zero,bounds=(0, 100), method='lm',loss='linear', args=(self.u, self.y),jac_sparsity='sparse matrix')

		assert_raises(ValueError, least_squares, self.fun, self.x0, jac=self.jac,bounds=(0, 100),x_scale=-1.0, method='dogbox',loss='linear',tr_solver='lsmr', args=(self.u, self.y),jac_sparsity='sparse matrix')

		assert_raises(ValueError, least_squares, self.fun, self.x0,bounds=(0, 100), method='dogbox',loss='linear',tr_solver='lsmr', args=(self.u, self.y),jac_sparsity=self.x0)

		assert_raises(TypeError, least_squares, self.fun, self.x0, jac='3-point',bounds=(0, 100), loss=self.cauchy_y, tr_solver=None, args=(self.u, self.y))

		assert_raises(TypeError, least_squares, self.fun_not_returning, self.x0, jac=self.jac_zero,bounds=(0, 100), method='dogbox',loss='huber',tr_solver='lsmr', args=(self.u, self.y),jac_sparsity='sparse matrix')

		#fun must rerun atmost-1d arrays
		assert_raises(ValueError, least_squares, self.fun_ndim, self.x0,jac='cs', bounds=(0, 100),loss='linear',method='trf', args=(self.u, self.y),jac_sparsity=None, verbose=2)

		#method='lm' does not support `jac_sparsity`.
		assert_raises(ValueError, least_squares, self.fun, self.x0,diff_step=1e-8, method='lm',tr_solver='exact', jac_sparsity='sparse matrix', args=(self.u, self.y))

		#tr_solver='exact' is incompatible with `jac_sparsity` when method !=lm
		assert_raises(ValueError, least_squares, self.fun, self.x0,bounds=(0, 100), method='dogbox',tr_solver='exact', args=(self.u, self.y),jac_sparsity='sparse matrix')
		#warn("The keyword 'regularize' in `tr_options` is not relevant "
		assert_raises(AttributeError, least_squares, self.fun, self.x0,bounds=(0, 100), method='dogbox',tr_solver='lsmr', tr_options='regularize', args=(self.u, self.y))
		#Method 'lm' doesn't work when the number of residuals is less than the number of variables
		assert_raises(ValueError, least_squares, self.fun, self.x0_large, method='lm', args=(self.u, self.y))
		#The return value of `loss` callable has wrong shape" both two cases
		assert_raises(ValueError, least_squares, self.fun, self.x0,method='lm,', jac='3-point',bounds=(0, 100), loss=self.cauchy_z, tr_solver=None,
							args=(self.u, self.y))
		assert_raises(ValueError, least_squares, self.fun, self.x0, jac='3-point', bounds=(0, 100), loss=self.cauchy_z, tr_solver=None,
							args=(self.u, self.y))



	def test2_green(self):
		"""
		2: Green Test Path: This test will execute least_square function with default values
		(i.e call least square with the default parameters: method= trf , default jacobian (2-point),
		valid bounds, tr_solve=none, loss='linear' and verbose>=1)
		"""
		res = least_squares(self.fun, self.x0, bounds=(0, 100),args=(self.u, self.y))
		assert_almost_equal(res.x, self.res_test,decimal=3)
		res = least_squares(self.fun, self.x0, bounds=(0, 100), args=(self.u, self.y), ftol=1e-20, xtol=1e-24, gtol=1e-23)
		assert_almost_equal(res.x, self.res_test, decimal=3)

	def test3_blue(self):
		"""
		3: Blue Test Path: This test will cover:
		jacobian='cs'; default method(i.e 'trf'), jac_sparsity=None, verbose=2
		"""
		res = least_squares(self.fun, self.x0,jac='cs', bounds=(0, 100),loss='linear',method='trf', args=(self.u, self.y),jac_sparsity=None, verbose=2)
		assert_almost_equal(res.x, self.res_test,decimal=6)

		res = least_squares(self.fun, self.x0,x_scale='jac', args=(self.u, self.y))
		assert_almost_equal(res.x, self.res_temp_x,decimal=3)

		res = least_squares(self.fun, self.x0,x_scale='jac', args=(self.u, self.y))
		assert_almost_equal(res.x, self.res_temp_x,decimal=3)



	def test4_yellow(self):
		"""
		4: Yellow Test Path: This test will cover:
		jacobian='2-point', method=='lm', no bounds,tr_solve='exact', verbose=default,
		"""
		res = least_squares(self.fun, self.x0, jac='2-point',method='lm',tr_solver='exact', args=(self.u, self.y))
		assert_almost_equal(res.x, self.res_unbound_test,decimal=4)

		res = least_squares(self.fun, self.x0, jac='2-point',diff_step=1e-8, method='lm',tr_solver='exact', args=(self.u, self.y))
		assert_almost_equal(res.x, self.res_unbound_test,decimal=4)

		res = least_squares(self.fun, self.x0, jac=self.jac,diff_step=1e-14,max_nfev=None, method='lm',tr_solver='exact', args=(self.u, self.y))
		assert_almost_equal(res.x, self.res_test_lm,decimal=2)

		#warn("jac='{0}' works equivalently to '2-point'
		res = least_squares(self.fun, self.x0, jac='3-point', method='lm', args=(self.u, self.y))
		assert_almost_equal(res.x, self.res_unbound_test,decimal=4)




	def test5_cyan(self):
		"""
		4: Cyan Test Path: This test will cover:
		callable jacobian=self, method=='dogbox',bounded,loss='linear',tr_solve='lsmr',jac_sparsity=array_like, verbose=default,
		"""
		res = least_squares(self.fun, self.x0, jac=self.jac,bounds=(0, 100), method='dogbox',loss='linear',tr_solver='lsmr', args=(self.u, self.y),jac_sparsity='sparse matrix')
		assert_almost_equal(res.x, self.res_test,decimal=3)

		res = least_squares(self.fun, self.x0, jac=self.jac_zero,bounds=(0, 100), method='dogbox',loss='huber',tr_solver='lsmr', args=(self.u, self.y),jac_sparsity='sparse matrix')
		assert_almost_equal(res.x, self.x0,decimal=3)

		res = least_squares(self.fun, self.x0, jac=self.jac_s,diff_step=20,args=(self.u_s, self.y_s))
		assert_almost_equal(res.x, self.res_temp,decimal=3)

	def test6_grey(self):
		"""
		4: Grey Test Path: This test will cover:
		jacobian='3-point', method=default, bounded, loss='linear',tr_solve=none, verbose=default,
		"""
		res = least_squares(self.fun, self.x0, jac='3-point',bounds=(0, 100),loss='linear',tr_solver=None, args=(self.u, self.y))
		assert_almost_equal(res.x, self.res_test,decimal=3)
		res = least_squares(self.fun, self.x0, jac='3-point', bounds=(0, 100), loss='huber', tr_solver=None,
							args=(self.u, self.y))
		assert_almost_equal(res.x, self.res_test, decimal=3)
		res = least_squares(self.fun, self.x0, jac='3-point', bounds=(0, 100), loss='cauchy', tr_solver=None,
							args=(self.u, self.y))
		assert_almost_equal(res.x, self.res_test, decimal=3)
		res = least_squares(self.fun, self.x0, jac='3-point', bounds=(0, 100), loss='soft_l1', tr_solver=None,
							args=(self.u, self.y))
		assert_almost_equal(res.x, self.res_test, decimal=3)

		res = least_squares(self.fun, self.x0, jac='3-point',bounds=(0, 100), loss='arctan', tr_solver=None,
							args=(self.u, self.y))
		assert_almost_equal(res.x, self.res_test, decimal=3)

		res = least_squares(self.fun, self.x0, jac='3-point',bounds=(0, 100), loss=self.cauchy_x, tr_solver=None,
							args=(self.u, self.y))
		assert_almost_equal(res.x, self.res_test, decimal=5)

	def runTest(self):
		pass
	def main(self):
		unittest.main()


if __name__ == '__main__':
	unittest.main()
