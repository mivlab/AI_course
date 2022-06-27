import numpy as np
arr = np.arange(0, 12, 1.0)
a = arr.reshape(3, 4)
b = np.ones((3, 4), dtype=np.int32)
c = a + b
print(arr)
print(arr.shape)
print(np.mean(a))
print(c[1:2, :3])

aa = np.arange(0, 3.14, 30.0/ 180. * 3.14)
print(np.sin(aa))
