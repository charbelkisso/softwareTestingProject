from ndimage.ndimage_test import *
from signal_test.test_signal import *
from linalg_test.test_linalg import *
from integrate_test.test_integrate import *



class Main():

    def __init__(self):
        self.nd_test = Test_ndimage()
        self.sig_test = test_signal()
        self.linalg_test = Test_linalg()
        self.integ_test = Test_integrate()

    def main(self):
        self.nd_test.main()
        self.sig_test.main()
        self.linalg_test.main()
        self.integ_test.main()


if __name__ == '__main__':

    mainEX = Main()
    mainEX.main()

    