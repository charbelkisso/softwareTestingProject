from signal_test.convolve.ConvCorr import *
from signal_test.firwin.firwin import *

class test_signal:

    def __init__(self):
        self.convcorr = Test_ConvolveCorrealte()
        self.firwin = TestFIR()

    def main(self):
        self.convcorr.main()
        self.firwin.main()
