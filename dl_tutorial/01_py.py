a = [1,2,3]
b = [4,5,6]
print([i * i for i in a])

#以下两种写法均可，推荐第1种
print([i * i + j * j for (i, j) in zip(a, b)])
print([a[i]**2 + b[i]**2 for i in range(len(a))])
