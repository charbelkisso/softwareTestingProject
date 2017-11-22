from numpy import *
from scipy import signal, misc
import matplotlib.pyplot as plt


image = misc.face(gray=True).astype(float32)
derfilt = array([1.0,-2,1.0],float32)
ck = signal.cspline2d(image,8.0)
deriv = signal.sepfir2d(ck, derfilt, [1]) +  signal.sepfir2d(ck, [1], derfilt)


plt.figure()
plt.imshow(image)

plt.gray()
plt.title("Original Image")
plt.show()



