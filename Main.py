from ndimage.ndimage_test import *
from signal_test.test_signal import *
class Main():

    def __init__(self):
        self.nd_test = Test_ndimage()
        self.sig_test = test_signal()

    def main(self):
        self.nd_test.main()
        self.sig_test.main()

if __name__ == '__main__':

    mainEX = Main()
    mainEX.main()

    