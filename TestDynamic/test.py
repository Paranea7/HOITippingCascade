#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transient_HOI.py  修正版：函数先定义，后调用
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import circulant
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import os

# -------------------- 参数区 --------------------
α       = 0.7
N       = 3
t_max   = 2000
rtol    = atol = 1e-8
tol     = 1e-6
n_stat  = 20
g2_grid = np.linspace(0.05, 1.0, 20)
dir_out = 'fig_out'
os.makedirs(dir_out, exist_ok=True)

# #################### 工具函数 ####################
def make_B1_B2(A):
    B1 = A - A.T
    B2 = np.empty((N, N, N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                B2[i, j, k] = 2 * A[i, j] * A[i, k] - A[j, i] * A[j, k] - A[k, i] * A[k, j]
    return B1, B2

def make_B3(A):
    N = A.shape[0]
    B3 = np.zeros((N, N, N, N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    B3[i, j, k, l] = 3 * A[i, j] * A[i, k] * A[i, l] - A[j, i] * A[j, k] * A[j, l] - \
                                     A[k, i] * A[k, j] * A[k, l] - A[l, i] * A[l, j] * A[l, k]
    return B3

def dxdt(t, x, g1, g2, B1, B2):
    term1 = B1 @ x
    term2 = np.einsum('ijk,j,k->i', B2, x, x)
    return x * (g1 * term1 + g2 * term2)

def dxdt_tri(t, x, g1, g2, g3, B1, B2, B3):
    t1 = B1 @ x
    t2 = np.einsum('ijk,j,k->i', B2, x, x)
    t3 = np.einsum('ijkl,j,k,l->i', B3, x, x, x)
    return x * (g1 * t1 + g2 * t2 + g3 * t3)

def simulate(x0, g2, B1, B2):
    g1 = 1 - g2
    sol = solve_ivp(dxdt, [0, t_max], x0, args=(g1, g2, B1, B2),
                    method='RK45', rtol=rtol, atol=atol, t_eval=np.linspace(0, t_max, 5000))
    return sol.t, sol.y.T

def simulate_tri(x0, g1, g2, g3, B1, B2, B3):
    sol = solve_ivp(dxdt_tri, [0, t_max], x0, args=(g1, g2, g3, B1, B2, B3),
                    method='RK45', rtol=rtol, atol=atol, t_eval=np.linspace(0, t_max, 5000))
    return sol.t, sol.y.T

def transient_time(traj):
    eq = np.full(N, 1 / N)
    diff = np.linalg.norm(traj - eq, axis=1)
    idx = np.where(diff < tol)[0]
    return idx[0] if idx.size else len(traj)

def jacobian_eq(g2, B1, B2):
    eq = np.full(N, 1 / N)
    g1 = 1 - g2
    J = np.zeros((N, N))
    for i in range(N):
        dj = 0.0
        for k in range(N):
            dj += g1 * B1[i, k] * eq[k] + 2 * g2 * np.sum(B2[i, k, :] * eq * eq)
        J[i, i] = dj
        for j in range(N):
            if i == j: continue
            J[i, j] = eq[i] * (g1 * B1[i, j] + 2 * g2 * np.sum(B2[i, j, :] * eq))
    return J

def jacobian_tri(g1, g2, g3, B1, B2, B3):
    eq = np.full(N, 1 / N)
    J = np.zeros((N, N))
    for i in range(N):
        dj = 0.0
        for k in range(N):
            dj += g1 * B1[i, k] * eq[k] + 2 * g2 * np.sum(B2[i, k, :] * eq * eq) + 3 * g3 * np.sum(B3[i, k, :, :] * eq * eq * eq)
        J[i, i] = dj
        for j in range(N):
            if i == j: continue
            J[i, j] = eq[i] * (g1 * B1[i, j] + 2 * g2 * np.sum(B2[i, j, :] * eq) + 3 * g3 * np.sum(B3[i, j, :, :] * eq * eq))
    return J

# #################### 主程序 ####################
def main():
    print('>>> 构建 A 矩阵 & B 系数 ...')
    A = circulant([0.5, α, 1 - α])
    np.fill_diagonal(A, 0.5)
    B1, B2 = make_B1_B2(A)
    B3 = make_B3(A)

    print('>>> Figure 1 轨迹叠堆 ...')
    fig, ax = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
    x0 = np.array([0.45, 0.3, 0.25])
    for col, (g2, title) in enumerate([(0.0, 'pairwise (γ₁=1)'),
                                       (1.0, 'higher-order (γ₂=1)'),
                                       (0.5, 'mixed (γ₁=γ₂=0.5)')]):
        t, traj = simulate(x0, g2, B1, B2)
        for i in range(N):
            ax[col].plot(t, traj[:, i], lw=2, label=f'sp{i+1}')
        ax[col].set_title(title);
        ax[col].set_xlabel('time');
        ax[0].set_ylabel('density')
    plt.tight_layout();
    plt.savefig(f'{dir_out}/Fig1_traces.png', dpi=300)

    print('>>> Figure 3 箱线统计 ...')
    T_dict = {g2: [] for g2 in g2_grid}
    for g2 in g2_grid:
        for _ in range(n_stat):
            x0 = np.random.dirichlet(np.ones(N))
            t, traj = simulate(x0, g2, B1, B2)
            T_dict[g2].append(transient_time(traj))
    T_mean = np.array([np.mean(T_dict[g]) for g in g2_grid])
    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot([T_dict[g] for g in g2_grid], positions=g2_grid, widths=0.02, patch_artist=True)
    for patch in bp['boxes']: patch.set_facecolor('lightblue')
    ax.plot(g2_grid, T_mean, 'ro-', lw=2, label='mean')
    ax.set_xlabel('γ₂'); ax.set_ylabel('transient time'); ax.legend()
    plt.tight_layout();
    plt.savefig(f'{dir_out}/Fig3_box.png', dpi=300)

    print('>>> Figure 4 双曲线 + 特征值线性 ...')
    c, _ = curve_fit(lambda g, c: c / g, g2_grid, T_mean)
    g2_fine = np.linspace(0.05, 1, 100)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(g2_grid, T_mean, 'ro', label='simulation')
    ax1.plot(g2_fine, c[0] / g2_fine, 'k--', label=f'hyperbola c={c[0]:.1f}')
    ax1.set_xlabel('γ₂'); ax1.set_ylabel('mean transient time'); ax1.legend()
    re_lam = np.array([jacobian_eq(g2, B1, B2).real.min() for g2 in g2_grid])
    slope, *_ = np.polyfit(g2_grid, re_lam, 1)
    ax2.plot(g2_grid, re_lam, 'bo', label='numerical')
    ax2.plot(g2_grid, slope * g2_grid, 'k--', label=f'linear slope={slope:.4f}')
    ax2.set_xlabel('γ₂'); ax2.set_ylabel('Re(λ_max)'); ax2.legend()
    plt.tight_layout();
    plt.savefig(f'{dir_out}/Fig4_fit.png', dpi=300)

    print('>>> Figure 6 三角形热力 ...')
    n_t = 25
    g2_tri = np.linspace(0.05, 0.95, n_t)
    g3_tri = np.linspace(0.05, 0.95, n_t)
    T_map = np.full((n_t, n_t), np.nan)
    Lam_map = np.full_like(T_map, np.nan)
    for i, g3 in enumerate(g3_tri):
        for j, g2 in enumerate(g2_tri):
            g1 = 1 - g2 - g3
            if g1 <= 0: continue
            T_list = []
            for _ in range(n_stat // 2):
                x0 = np.random.dirichlet(np.ones(N))
                t, traj = simulate_tri(x0, g1, g2, g3, B1, B2, B3)
                T_list.append(transient_time(traj))
            T_map[i, j] = np.mean(T_list)
            J = jacobian_tri(g1, g2, g3, B1, B2, B3)
            Lam_map[i, j] = np.linalg.eigvals(J)[np.linalg.eigvals(J).real.argmin()].real
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    im1 = ax1.contourf(g2_tri, g3_tri, T_map, levels=50, cmap='viridis')
    plt.colorbar(im1, ax=ax1); ax1.set_xlabel('γ₂'); ax1.set_ylabel('γ₃'); ax1.set_title('transient time')
    im2 = ax2.contourf(g2_tri, g3_tri, Lam_map, levels=50, cmap='plasma')
    plt.colorbar(im2, ax=ax2); ax2.set_xlabel('γ₂'); ax2.set_ylabel('γ₃'); ax2.set_title('Re(λ_max)')
    plt.tight_layout();
    plt.savefig(f'{dir_out}/Fig6_triangle.png', dpi=300)

    print('>>> 完成！文件已输出到', dir_out)

if __name__ == '__main__':
    main()