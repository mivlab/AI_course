
import torch
import numpy as np

# 这个例子程序是展示创建Tensor的各种方法

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data, dtype=torch.uint8) # 把list转换为Tensor
np_array = np.array(data)
x_np = torch.from_numpy(np_array) # 把ndarray转换为Tensor
shape = (2,3)
rand_tensor = torch.rand(shape) # 根据形状创建随机数Tensor
ones_tensor = torch.ones(shape, dtype=torch.uint8) # 根据形状创建全1 的Tensor
zeros_tensor = torch.zeros(shape) # 根据形状创建全0的Tensor

print(x_data)
print(x_np)
print(rand_tensor)
print(ones_tensor)
print(zeros_tensor)

