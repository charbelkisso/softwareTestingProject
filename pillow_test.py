import unittest

import numpy as np
from  numpy.testing import assert_equal


from scipy import misc

import scipy


class Test_scipy_resize(unittest.TestCase):

    def test_imresize(self):

        im = np.random.random((10,20))

        for T in np.sctypes['float'] + [float]:
            im1 = misc.imresize(im,T(2.0))
            self.assertEqual(im1.shape, (20, 40))

    def test_imresize4(self):

        im = np.array([[1, 2],
                       [3, 4]])

        res = np.array([[1., 1.25, 1.75, 2.],
                        [1.5, 1.75, 2.25, 2.5],
                        [2.5, 2.75, 3.25, 3.5],
                        [3., 3.25, 3.75, 4.]], dtype=np.float32)

        im2 = misc.imresize(im, (4, 4), mode='F')  # output size
        im3 = misc.imresize(im, 2., mode='F')  # fraction
        im4 = misc.imresize(im, 200, mode='F')  # percentage

        assert_equal(im2, res)
        assert_equal(im3, res)
        assert_equal(im4, res)

if __name__ == '__main__' :

    unittest.main()