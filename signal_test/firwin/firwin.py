"""
importing important libraries for testing
"""
import unittest
import numpy as np
"""
built-in assert function works with arrays and ndArrays
"""
from numpy.testing import assert_equal, assert_almost_equal
"""
functions to test:

ToDo: add function here to build test cases for it
"""
from scipy import signal, misc
import scipy

class TestFIR(unittest.TestCase):

    def setUp(self):
        """
        Def inputs for filter
        """
        self.numtaps = 3
        self.numtaps_1 = 5
        self.numtaps_zero= 0
        self.freq= 0.1
        self.freq_1 = 0.2
        self.freq_zero= 0

        """
        Expected outputs for filter
        """
        self.ans_fir= [0.067,0.864,0.067]
        self.ans_fir_1= [0.028,0.237,0.469,0.237,0.028]
        self.res_zero= np.array([])

    def test_default(self):
        """
        this test unit meant to test firwin function from scypi.signal

        function description:
        ---------------------
        FIR filter design using the window method.
        """
        assert_almost_equal(signal.firwin(self.numtaps, self.freq), self.ans_fir, decimal=3)
        assert_almost_equal(signal.firwin(self.numtaps_1, self.freq_1), self.ans_fir_1, decimal=3)
    def test_numptaps_zero(self):
        """
        this test unit meant to test firwin fuction with zero numtaps
        """

        assert_equal(signal.firwin(self.numtaps_zero, self.freq), self.res_zero)

    def test_freq_zero(self):
        """
        this test unit meant to test firwin fuction with zero frequency
        must araise an error because zero frequency is unrealistic
        """
        try:
            signal.firwin(self.numtaps,self.freq_zero)
        except Exception as e:
            self.assertEqual(e.__class__, ValueError)

        try:
            signal.firwin(self.numtaps_zero, self.freq_zero)
        except Exception as e:
            self.assertEqual(e.__class__, ValueError)

    def main(self):
        unittest.main()

