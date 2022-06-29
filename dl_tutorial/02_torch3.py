import numpy as np
import torch
a = torch.Tensor(np.arange(1, 65, 1).reshape(8, 8))
print(torch.stack([a[0:2, 0:3], a[0:2, -3:]]))
