import numpy as np
a = np.arange(1.0, 19.0, 1.0).reshape(3, 6)
b = np.random.rand(6,6) * 3
c = np.ones((6, 6))
print(np.dot(b[0:2,-2:], c[0:2,-3:]))







