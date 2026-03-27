import numpy as np

np.random.seed(0)

x = np.random.randn(2000)

mean = np.mean(x)
std_dev = np.std(x, ddof=1)

print(mean)
print(std_dev)
