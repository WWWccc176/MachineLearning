import random
import torch
import matplotlib.pyplot as plt

def synthesize_data(w, b, n):  
    X = torch.normal(0, 1, (n, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2.0, -3.4])
true_b = 4.2
features, labels = synthesize_data(true_w, true_b, 1000)#这里是真实的数据

print('features:', features[0], '\nlabel:', labels[0])#[0]的意义是取个样，不全部输出
print("-" * 50)

plt.figure(figsize=(5, 3.5))

plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), s=1)

plt.xlabel('feature 2 (features[:, 1])')
plt.ylabel('label')
plt.title('Linear Relationship')

plt.show()

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

w = torch.normal(0,0.01, size=(2,1), requires_grad = True)
b = torch.zeros(1, requires_grad = True)

def linreg(X,w,b):
    return torch.matmul(X,w)+b

def squ_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(paras, lr, batch_size):
    with torch.no_grad():
        for para in paras:
            para -= lr * para.grad / batch_size
            para.grad.zero_()

lr = 0.03
num_epochs = 5
net = linreg
loss = squ_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # 计算小批量损失
        l.sum().backward()
        
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
        
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

# 打印模型最终学到的参数
print(f'最终学到的 w:{w.detach().numpy()},最终学到的 b:, {b.detach().numpy()}')
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
