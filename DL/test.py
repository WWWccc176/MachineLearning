import torch

print("d2l 导入成功！")
print(f"PyTorch version: {torch.__version__}")
print("hello, there")

print("This is a code testing whether your number is 13.")
a = int(input("input your number:"))

if a - (-1) != 14:
    print("your number isn't 13!")
else:
    print("your number is 13!")

