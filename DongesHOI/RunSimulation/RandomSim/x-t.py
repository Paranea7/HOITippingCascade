import numpy as np
from numba import njit
from multiprocessing import Pool
import matplotlib.pyplot as plt

############################################
# 参数生成
############################################

def generate_parameters(s, mu_d, sigma_d, mu_e, sigma_e):
    d = np.random.normal(mu_d / s, sigma_d / np.sqrt(s), (s, s))
    np.fill_diagonal(d, 0)
    e = np.random.normal(mu_e / s, sigma_e / s**1.5, (s, s, s))
    for i in range(s):
        e[i, i, i] = 0
    return d, e

############################################
# numba ODE
############################################

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

############################################
# 单次模拟（返回完整轨迹）
############################################

def simulate_trajectory(s, t_steps, dt, d, e):
    x = np.full(s,-0.6)
    traj = np.zeros((t_steps, s))
    for t in range(t_steps):
        traj[t] = x
        x = rk4_step_numba(x, d, e, dt)
    return traj

############################################
# 多进程运行（只输出最终状态）
############################################

def simulate_once(args):
    s, t_steps, dt, mu_d, sigma_d, mu_e, sigma_e = args
    d, e = generate_parameters(s, mu_d, sigma_d, mu_e, sigma_e)
    x = np.full(s,-0.6)
    for _ in range(t_steps):
        x = rk4_step_numba(x, d, e, dt)
    return x

def run_parallel(batch, s, t_steps, dt, mu_d, sigma_d, mu_e, sigma_e, n_jobs=4):
    args_list = [(s, t_steps, dt, mu_d, sigma_d, mu_e, sigma_e) for _ in range(batch)]
    with Pool(n_jobs) as pool:
        results = pool.map(simulate_once, args_list)
    return np.array(results)

############################################
# 主程序 + 绘图
############################################

if __name__ == "__main__":
    s = 50
    t_steps = 2000
    dt = 0.01

    mu_d = 0.4
    sigma_d = 0.6
    mu_e = 0.3
    sigma_e = 0.4

    print("Generating parameters...")
    d, e = generate_parameters(s, mu_d, sigma_d, mu_e, sigma_e)

    print("Simulating one full trajectory for plotting...")
    traj = simulate_trajectory(s, t_steps, dt, d, e)

    print("Running batch simulations...")
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

    ############################################
    # 图1：时间序列 x_i(t)
    ############################################
    plt.figure(figsize=(8, 4))
    for i in range(s):
        plt.plot(traj[:, i], label=f"x_{i}")

    plt.xlabel("t")
    plt.ylabel("x")
    plt.show()

    ############################################
    # 图2：相图 (x_i, x_j)
    ############################################
    i, j = 1, 49
    plt.figure(figsize=(4, 4))
    plt.plot(traj[:, i], traj[:, j])
    plt.xlabel(f"x_{i}")
    plt.ylabel(f"x_{j}")
    plt.tight_layout()
    plt.show()

    ############################################
    # 图3：多次模拟最终状态分布
    ############################################
    plt.figure(figsize=(6, 4))
    flat = xs_final.flatten()
    plt.hist(flat, bins=40, density=True)

    plt.xlabel("x")
    plt.ylabel("PDF")
    plt.tight_layout()
    plt.show()

    print("Finished.")