from ndimage.ndimage_test import *
from signal_test.test_signal import *
from linalg_test.test_linalg import *
class Main():

    def __init__(self):
        self.nd_test = Test_ndimage()
        self.sig_test = test_signal()
        self.linalg_test = Test_linalg()

    def main(self):
        self.nd_test.main()
        self.sig_test.main()
        self.linalg_test.main()

if __name__ == '__main__':

    mainEX = Main()
    mainEX.main()

    