import torch

x = torch.tensor(3)
y = torch.tensor(2)
print(x + y)

A = torch.arange(16).reshape(4, 4)
B = torch.arange(16).reshape(4, 4)
print(A.T+B)

X = torch.arange(20, dtype=torch.float64).reshape(5, 4)
Y = X.clone()
print(X)
print(Y)

sum_A = A.sum(axis=1, keepdims=True)
print(sum_A)

print(torch.mm(A, B))# 向量点积dot(),矩阵乘向量：mv()，矩阵乘矩阵mm()










