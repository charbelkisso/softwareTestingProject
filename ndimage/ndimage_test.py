from ndimage.gaussian.gaussian_filter_test import *
from ndimage.laplace.laplace_test import *

class Test_ndimage():

    def __init__(self):
        self.gaussian = Gaussian_Filter_Test()
        self.laplace = Laplace_Test()
        
    def main(self):
        self.gaussian.main()
        self.laplace.main()
