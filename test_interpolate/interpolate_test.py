import numpy as np
from scipy import interpolate
import unittest

class Test_Interpolate(unittest.TestCase):

    def setUp(self):
        self.step = 1
        self.x = np.arange(0, 4 * np.pi, self.step)
        self.y = np.sin(self.x)
        self.f = interpolate.interp1d(self.x, self.y)
        self.f2 = interpolate.interp1d(self.x, self.y, kind='cubic')

    def test_min_val_lin_mode(self):
        """
        testing if the both given and generated point has the same minimum point
        this valid for linear mode
        :return:
        """
        xnew = np.arange(0,4*np.pi-self.step/2,self.step/10)
        ynew = self.f(xnew)
        self.assertEqual(ynew.min(), self.y.min())

    def test_max_val_lin_mode(self):
        """
        testing if the both given and generated point has the same maximum point
        this valid for linear mode
        :return:
        """
        xnew = np.arange(0, 4 * np.pi - self.step / 2, self.step / 10)
        ynew = self.f(xnew)
        self.assertEqual(ynew.max(), self.y.max())

    def compare_y(self, a, b):
        """
        this method compare the y axis for both given and generated point
        :param a: the given y axis
        :param b: the generated y axis
        :return: True if y:s of A array are included in y:s of B array
                 False if not
        """
        compare_result = [False] * a.size
        for i in range(0, a.size):
            for j in range(0, b.size):
                if a[i] == b[j]:
                    compare_result[i] = True
                    break
        if compare_result:
            return True
        else:
            return False

    def test_y_any_mode(self):

        xnew = np.arange(0, 4 * np.pi - self.step / 2, self.step / 10)
        ynew = self.f(xnew)

        self.assertTrue(self.compare_y(self.y,ynew))

        ynew_c = self.f2(xnew)

        self.assertTrue(self.compare_y(self.y,ynew_c
                                       ))

    def main(self):
        unittest.main()

    def runTest(self):
        pass