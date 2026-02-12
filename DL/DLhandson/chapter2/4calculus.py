import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 3 * x**2 - 4 * x


def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h


h = 0.1
for i in range(5):
    print(f"h={h:.10f}, numerical limit={numerical_lim(f, 1, h):.10f}")
    h *= 0.05

x = np.arange(0, 3, 0.1)
y1 = f(x)
y2 = 6 * x - 8.25

plt.figure(figsize=(6, 4))

plt.plot(x, y1, "-", label="f(x)")

plt.plot(x, y2, "m--", label="Tangent line (x=1.5)")

plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()  # 图像的标签
plt.grid(True)

plt.show()
