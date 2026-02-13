import torch
import matplotlib
matplotlib.use('Qt5Agg')#linux version
import matplotlib.pyplot as plt
from torch.distributions import multinomial

# 设置随机种子以便结果可复现
torch.manual_seed(42)

# 1. 定义公平骰子的概率向量
fair_probs = torch.ones([6]) / 6

# 2. 单个样本（掷一次）
single_sample = multinomial.Multinomial(1, fair_probs).sample()
print("单次投掷结果：", single_sample)

# 3. 一次抽取10个样本
ten_samples = multinomial.Multinomial(10, fair_probs).sample()
print("10次投掷计数：", ten_samples)

# 4. 模拟1000次投掷，计算相对频率
counts_1000 = multinomial.Multinomial(1000, fair_probs).sample()
relative_freq = counts_1000 / 1000
print("1000次投掷的相对频率：", relative_freq)

# 5. 500组实验，每组抽取10个样本
counts = multinomial.Multinomial(10, fair_probs).sample((500,))  # 形状: (500, 6)
cum_counts = counts.cumsum(dim=0)                               # 沿实验组累积求和
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)   # 计算累积相对频率

# 6. 用 Matplotlib 绘制6条概率估计曲线
plt.figure(figsize=(6, 4.5))
for i in range(6):
    plt.plot(estimates[:, i].numpy(), label=f"P(die={i + 1})")

# 添加理论值水平线（1/6 ≈ 0.167）
plt.axhline(y=1/6, color='black', linestyle='dashed', label='True probability (1/6)')

plt.xlabel('Groups of experiments')
plt.ylabel('Estimated probability')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()
