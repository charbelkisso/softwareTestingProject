from scipy import ndimage
import numpy as np

a = [[0,-1,0],[-1,4,-1],[0,-1,0]]
a = np.asarray(a, dtype=np.float)

tmp1 = ndimage.correlate1d(a,[1, -2, 1], 0 )
tmp2 = ndimage.correlate1d(a,[1, -2, 1], 1 )
b1 = ndimage.laplace(a)
b2 = tmp1 + tmp2


print(b1)
print(b2)
