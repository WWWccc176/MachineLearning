import sympy as sp

beta1, beta2 = sp.symbols("beta1 beta2", real=True)
y1, y2 = sp.symbols("y1 y2", real=True)
x11, x12, x21, x22 = sp.symbols("x11 x12 x21 x22", real=True)

y = sp.Matrix([y1, y2])
x = sp.Matrix([[x11, x12], [x21, x22]])
beta = sp.Matrix([beta1, beta2])

S_manual = (y1 - beta1 * x11 - beta2 * x12) ** 2 + (y2 - beta1 * x21 - beta2 * x22) ** 2
S_manual = sp.expand(S_manual)  # 展开便于比较

residual = y - x * beta
S_matrix = (residual.T * residual)[0]  # 结果为1x1矩阵，取第一个元素
S_matrix = sp.expand(S_matrix)

print("\n=== 第二部分：验证矩阵形式与求和形式相等 ===")
print("手动求和 S =", S_manual)
print("矩阵形式 S =", S_matrix)
print("相等性:", sp.simplify(S_manual - S_matrix) == 0)

dS_dbeta1 = sp.diff(S_matrix, beta1)
dS_dbeta2 = sp.diff(S_matrix, beta2)
dS_dbeta = sp.Matrix([dS_dbeta1, dS_dbeta2])

dS_dbeta_simplified = sp.simplify(dS_dbeta)
print("\n导数向量 dS/dβ =", dS_dbeta_simplified)

expected_gradient = -2 * x.T * (y - x * beta)
expected_gradient_simplified = sp.simplify(expected_gradient)
print("\n理论值 -2 x^T (y - xβ) =", expected_gradient_simplified)
print(
    "相等性:",
    sp.simplify(dS_dbeta_simplified - expected_gradient_simplified)
    == sp.Matrix([0, 0]),
)
