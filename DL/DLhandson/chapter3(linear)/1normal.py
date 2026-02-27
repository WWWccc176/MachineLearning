import torch as tc
import math  # 替换为内置的 math 库
import matplotlib.pyplot as plt

def normal(x, miu, sigma):
    const = 1 / math.sqrt(2 * math.pi * sigma**2)
    return const * tc.exp(-(x - miu)**2 / (sigma**2 * 2))

x = tc.arange(-7, 7, 0.01)
params = [(0, 1), (0, 2), (3, 1)]

plt.figure(figsize=(4.5, 2.5))

# 绘制每条曲线
for mu, sigma in params:
    y = normal(x, mu, sigma)
    plt.plot(x.numpy(), y.numpy(), label=f'mean {mu}, std {sigma}')

# 添加标签和图例
plt.xlabel('x')
plt.ylabel('p(x)')
plt.legend()
plt.show()