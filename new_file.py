import numpy as np
from scipy import  interpolate
import matplotlib.pyplot as plt

step = 1
x = np.arange(0,4*np.pi,step)
y = np.sin(x)
f = interpolate.interp1d(x,y)

xnew = np.arange(0,4*np.pi-step/2,step/10)
ynew = f(xnew)
plt.plot(x,y,'o', xnew, ynew, '-')
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

