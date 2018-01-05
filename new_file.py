import numpy as np
from scipy import  integrate
quad_func0 = lambda x: 1
quad_func1 = lambda x: np.exp(-x)
quad_func2 = lambda x: x**2
quad_func3 = lambda x: 5*x
print(integrate.quad(quad_func1,-np.inf,np.inf))
print (np.inf)