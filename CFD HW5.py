# %% 库

import numpy as np
import matplotlib.pyplot as plt

# %% 参数设置

N = 101                        # 网格数 (N x N)
h = 1 / (N-1)                 # 网格步长
nu = 0.001                    # 运动粘度
dt = 0.001                    # 时间步长
max_steps = 10000             # 最大时间步数
tol = 1e-5                    # 收敛容差
u = np.zeros((N, N))          # x方向速度
v = np.zeros((N, N))          # y方向速度
p = np.zeros((N, N))          # 压力场

# %% 设置上边界速度条件

x_coords = np.linspace(0, 1, N)
u[:, -1] = np.sin(np.pi * x_coords)**2  # u_top = sin²(πx)

# %% 离散 Laplace 算子（五点格式）

def laplace_5(f):
    laplace = np.zeros_like(f)
    laplace[1:-1, 1:-1] = (f[2:, 1:-1] + f[:-2, 1:-1] + f[1:-1, 2:] + f[1:-1, :-2] - 4*f[1:-1, 1:-1]) / h**2
    return laplace

# %% 对流项（中心差分）

def advection(u, v, field):
    
    # 初始化导数和对流项
    dfield_dx = np.zeros_like(field)
    dfield_dy = np.zeros_like(field)
    adv = np.zeros_like(field)
    
    # x方向导数
    # 内部点（中心差分）
    dfield_dx[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2 * h)
    # 左边界（前向差分）
    dfield_dx[0, :] = (field[1, :] - field[0, :]) / h
    # 右边界（后向差分）
    dfield_dx[-1, :] = (field[-1, :] - field[-2, :]) / h
    
    # y方向导数
    # 内部点（中心差分）
    dfield_dy[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2 * h)
    # 下边界（前向差分）
    dfield_dy[:, 0] = (field[:, 1] - field[:, 0]) / h
    # 上边界（后向差分）
    dfield_dy[:, -1] = (field[:, -1] - field[:, -2]) / h
    
    # 合成对流项
    adv = u * dfield_dx + v * dfield_dy
    return adv

# %% 使用Jacobi迭代求解压力泊松方程

def solve_pressure(u_star, v_star):
    p_new = np.zeros_like(p)
    rhs = (np.roll(u_star, -1, axis=0) - np.roll(u_star, 1, axis=0) + 
          np.roll(v_star, -1, axis=1) - np.roll(v_star, 1, axis=1)) / (2*h*dt)
    for _ in range(100):
        p_new = (np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0) + 
                np.roll(p, 1, axis=1) + np.roll(p, -1, axis=1) - 
                h**2 * rhs) / 4
        p[:, :] = p_new
    return p

# %% 主循环

for step in range(max_steps):
    
    # 保存上一步速度场用于收敛判断
    u_prev = u.copy()
    v_prev = v.copy()

    # 用显式Euler法计算中间速度
    u_star = u + dt * (nu * laplace_5(u) - advection(u, v, u))
    v_star = v + dt * (nu * laplace_5(v) - advection(u, v, v))

    # 求解压力泊松方程
    p = solve_pressure(u_star, v_star)

    # 速度修正
    grad_p_x = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / (2*h)
    grad_p_y = (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) / (2*h)
    u = u_star - dt * grad_p_x
    v = v_star - dt * grad_p_y

    # 边界条件
    # 固定边界（下、左、右）
    u[:, 0] = 0; v[:, 0] = 0       # 下边界
    u[0, :] = 0; v[0, :] = 0       # 左边界
    u[-1, :] = 0; v[-1, :] = 0     # 右边界
    
    # 上边界（u给定，v = 0）
    u[:, -1] = np.sin(np.pi * x_coords)**2
    v[:, -1] = 0

    # 压力边界
    p[:, 0] = p[:, 1]              # 下边界
    p[0, :] = p[1, :]              # 左边界
    p[-1, :] = p[-2, :]            # 右边界
    p[:, -1] = p[:, -2]            # 上边界

    # 检查收敛
    if step % 100 == 0:
        du = np.max(np.abs(u - u_prev))
        dv = np.max(np.abs(v - v_prev))
        if du < tol and dv < tol:
            print(f"收敛于第 {step} 步")
            break
        
# %% 流线图绘制

# 计算流函数
omega = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) - (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2*h)     
psi = np.zeros_like(u)
for _ in range(1000):  # Jacobi迭代求解ψ
    psi_new = (np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0) + 
              np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1) + 
              h**2 * omega) / 4
    psi = psi_new

# %% 绘制结果

x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)
plt.contour(X, Y, psi, levels=30, colors='blue')
plt.axvline(x = 0.6, color = 'black', linestyle = '--', linewidth = 1)
plt.title(f"Stream Line Graph (Grid = {N-1} | Central)")
plt.xlabel("x"); plt.ylabel("y")
plt.show()


