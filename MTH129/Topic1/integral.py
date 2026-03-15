import sympy as sp

# Define symbols
x = sp.symbols("x")

# Define function
f = sp.Piecewise((x**2, x < 0), ((3 * x) / 4, x > 3), ((x**2) / (x + 1), True))
df = sp.diff(f, x)
dfv = df.subs(x, 3)
print(df)
print(dfv)