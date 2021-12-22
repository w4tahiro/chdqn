import numpy as np

a = [[[0 for j in range(1)]for i in range(10)]for k in range(6)]

for i in range(6):
    for j in range(10):
        for k in range(3):
            a[i][j].append(k)

print(a)