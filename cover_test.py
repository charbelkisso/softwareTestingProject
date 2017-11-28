import unittest
import numpy as np
from scipy.optimize import least_squares
from numpy.testing import assert_equal, assert_almost_equal

def count_white_spaces(str):

	count = 0

	for i in range (0, len(str)):

		if (str[i] == ' '):
			count = count + 1

	return (count)

class Test_Count(unittest.TestCase):

	
		
	testStr1 = " "
	testStr2 = "h"
	testStr3 = ""

	def test1(self):

		self.assertEqual(count_white_spaces(self.testStr1), 1)
		self.assertEqual(count_white_spaces(self.testStr2), 0)
		self.assertEqual(count_white_spaces(self.testStr3), 0)
		
class Test_Least_Squares(unittest.TestCase):

	def setUp(self):

		self.res_test = [0.19280596, 0.19130423, 0.12306063, 0.13607247]
		self.u = np.array([4.0, 2.0, 1.0, 5.0e-1, 2.5e-1, 1.67e-1, 1.25e-1, 1.0e-1,
            8.33e-2, 7.14e-2, 6.25e-2])
		self.y = np.array([1.957e-1, 1.947e-1, 1.735e-1, 1.6e-1, 8.44e-2, 6.27e-2,
            4.56e-2, 3.42e-2, 3.23e-2, 2.35e-2, 2.46e-2])
		self.x0 = np.array([2.5, 3.9, 4.15, 3.9])
	
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

	def test1(self):
		res = least_squares(self.fun, self.x0, jac=self.jac, bounds=(0, 100), args=(self.u, self.y), verbose=1)
		assert_almost_equal(res.x, self.res_test)




if __name__ == '__main__':
	unittest.main()

