# %% 库

import numpy as np
import matplotlib.pyplot as plt
import time

# %% 参数

# 运动学粘度
ν = 0.001

# 网格数 (N * N)
N = 50

# 网格长度
h = 1 / (N - 1)

# x方向速度
u = np.zeros((N, N))

# y方向速度
v = np.zeros((N, N))

# 压力场
p = np.zeros((N, N))

# 最大迭代次数
max_iter = 10000

# 边界条件
# 上边界： u(x) = sin(np.pi * x) ** 2, v = 0
# 下、左、右边界: u, v = 0

# 设置边界条件
x_coord = np.linspace(0, )


# %% 迭代求解

for iters in range(max_iter):
    
    
    
    
