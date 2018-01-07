import numpy as np
from scipy import  interpolate
import matplotlib.pyplot as plt

step = 1
x = np.arange(0,4*np.pi,step)
y = np.cos(-x**2/9.0)
f = interpolate.interp1d(x,y, kind='cubic')
x_real = np.arange(0,4*np.pi,0.1)
y_real = np.cos(-x_real**2/9.0)

xnew = np.arange(0,4*np.pi-step/2,step/10)
ynew = f(xnew)
plt.plot(x,y,'o',label='point sample')
plt.plot(xnew,ynew,'-',label='approximated function')
plt.plot(x_real,y_real,'--',label='actual function')
plt.legend()
plt.title("function approximation using scipy interpolate")
plt.show()
print(y.min())
print(ynew.min())
print(ynew.max())
print(y.max())
ynew_av = ynew.sum()/ynew.shape
y_av = y.sum()/y.shape
print(ynew_av)
print(y_av)

def compare_y(a,b):
    compare_result = [False ]*a.size
    for i in range(0, a.size):
        for j in range(0,b.size):
            if a[i] == b[j]:
                compare_result[i] = True
                break
    if compare_result:
        return True
    else:
        return False

print(compare_y(y,ynew))


import scipy


print(scipy.__version__)