from linalg_test.hypmat.HypMatrix import *
from linalg_test.norm import *
from linalg_test.subspace_angles import *
from linalg_test.pascal import  *
from linalg_test.inv import *

class Test_linalg():

    def __init__(self):
        self.hypmat = HyperbolicMatrix_Test()
        self.norm = Test_norms()
        self.sbangl = Test_subspace_angles()
        self.pascal = Test_Pascal()
        self.inv_test = Test_Inv()

    def main(self):
        self.hypmat.main()
        self.norm.main()
        self.sbangl.main()
        self.pascal.main()
        self.inv_test.main()