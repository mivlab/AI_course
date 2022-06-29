import numpy as np
import torch

#分别用python和numpy实现两个矩阵相加
def matrix_add(m1, m2):
    if len(m1) != len(m2):
        return None
    if len(m1[0]) != len(m2[0]):
        return None
    m = []
    for i in range(len(m1)):
        row = []
        for j in range(len(m1[i])):
            row.append(m1[i][j] + m2[i][j])
        m.append(row)
    return m


def matrix_add_np(m1, m2):
    return m1 + m2


if __name__ == '__main__':
    m1 = [[0, 1, 2], [4, 5, 6]]
    m2 = [[3, 4, 5], [6, 7, 8]]
    print(matrix_add(m1, m2))
    print(matrix_add_np(np.array(m1), np.array(m2)))
    print(matrix_add_np(torch.Tensor(m1), torch.Tensor(m2)))
