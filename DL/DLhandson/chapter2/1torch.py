import torch

x = torch.arange(12)
print(x)
print(x.shape)

X = x.reshape(3, 4)
print(X)
print(X.shape)

Y=torch.randn(3, 4)#ones, zeros都可以
print(Y)
print(Y.shape)
print(X>Y)

xx=Y+X
print(xx)

y=torch.exp(Y)
print(y)

yy=y.sum()

a=torch.arange(12).reshape(12,1)
b=torch.arange(18)
print(a+b)

c=a+b
d=c.numpy()
print(type(c))
print(type(d))