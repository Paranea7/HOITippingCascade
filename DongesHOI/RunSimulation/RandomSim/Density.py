import numpy as np
from numba import njit
from multiprocessing import Pool
import matplotlib.pyplot as plt

# -----------------------------
# 参数生成
# -----------------------------
def generate_parameters(s, mu_d, sigma_d, mu_e, sigma_e):
    # 生成 d 矩阵
    d = np.random.normal(mu_d / s, sigma_d / np.sqrt(s), (s, s))
    np.fill_diagonal(d, 0)

    # 生成 e 张量
    e = np.random.normal(mu_e / s, sigma_e / s ** 2, (s, s, s))

    for i in range(s):
        e[i, i, :] = 0  # 设置 e[i, i, k] = 0 (前两个索引相同)
        e[i, :, i] = 0  # 设置 e[i, j, i] = 0 (首尾索引相同)
        e[:, i, i] = 0  # 设置 e[k, i, i] = 0 (后两个索引相同)

    return d, e

# -----------------------------
# ODE + RK4 (numba)
# -----------------------------
@njit
def dxdt_numba(x, d, e):
    s = len(x)
    out = -x**3 + x + d @ x
    for i in range(s):
        acc = 0.0
        for j in range(s):
            for k in range(s):
                acc += e[i, j, k] * x[j] * x[k]
        out[i] += acc
    return out

@njit
def rk4_step_numba(x, d, e, dt):
    k1 = dxdt_numba(x, d, e)
    k2 = dxdt_numba(x + 0.5 * dt * k1, d, e)
    k3 = dxdt_numba(x + 0.5 * dt * k2, d, e)
    k4 = dxdt_numba(x + dt * k3, d, e)
    return x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

# -----------------------------
# 单次模拟（只返回最终状态）
# -----------------------------
def simulate_once(args):
    s, t_steps, dt, mu_d, sigma_d, mu_e, sigma_e = args
    d, e = generate_parameters(s, mu_d, sigma_d, mu_e, sigma_e)
    x = np.full(s, -0.6)
    for _ in range(t_steps):
        x = rk4_step_numba(x, d, e, dt)
    return x

# -----------------------------
# 多进程模拟
# -----------------------------
def run_parallel(batch, s, t_steps, dt, mu_d, sigma_d, mu_e, sigma_e, n_jobs=4):
    args_list = [(s, t_steps, dt, mu_d, sigma_d, mu_e, sigma_e) for _ in range(batch)]
    with Pool(n_jobs) as pool:
        results = pool.map(simulate_once, args_list)
    return np.array(results)

# -----------------------------
# 绘制 density
# -----------------------------
def plot_density(xs_final):
    flat = xs_final.flatten()
    plt.figure(figsize=(6, 4))
    plt.hist(flat, bins=40, density=True)
    plt.xlabel("x")
    plt.ylabel("PDF")
    plt.tight_layout()
    plt.show()

# -----------------------------
# 主程序（计算 + density）
# -----------------------------
if __name__ == "__main__":
    s = 100
    t_steps = 2500
    dt = 0.01

    mu_d = 0.3
    sigma_d = 0.3
    mu_e = 0.3
    sigma_e = 0.4

    xs_final = run_parallel(
        batch=20,
        s=s,
        t_steps=t_steps,
        dt=dt,
        mu_d=mu_d,
        sigma_d=sigma_d,
        mu_e=mu_e,
        sigma_e=sigma_e,
        n_jobs=4
    )

    plot_density(xs_final)