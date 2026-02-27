import torch

x=torch.arange(5.,requires_grad=True)
print(x)

y=2*torch.dot(x,x)
print(y)
print(y.backward())
print(x.grad)

print(x.grad==4*x)

x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
print(x.grad == u)

x.grad.zero_()
y.sum().backward()
print(x.grad == 2)
