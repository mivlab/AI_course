
import torch
import numpy as np

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data) # 把list转换为Tensor
np_array = np.array(data)
x_np = torch.from_numpy(np_array) # 把ndarray转换为Tensor
shape = (2,3,)
rand_tensor = torch.rand(shape) # 根据形状创建Tensor
ones_tensor = torch.ones(shape, dtype=torch.uint8)
zeros_tensor = torch.zeros(shape)

print(x_np)
print(rand_tensor)
print(ones_tensor)
print(zeros_tensor)

