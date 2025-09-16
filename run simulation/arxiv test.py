#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
复现 Ghosh & Shrimali 2025 arXiv:2509.07802
Fig.1 – Fig.4  （单文件版）
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.sparse import random as sprand
from scipy.sparse.linalg import eigsh

# -------------------------------------------------
# 通用参数
# -------------------------------------------------
a, b, x0 = 4.0, 1.0, 0.5
r_crit = np.sqrt(4*b**3/(27*a))          # ≈0.192
N      = 100                             # 网络规模（Fig.2/3）
T      = 200
dt     = 0.1
tspan  = np.arange(0, T+dt, dt)

# -------------------------------------------------
# Fig.1  单节点尖点突变图（稳态分支）
# -------------------------------------------------
def fig1():
    r_vals = np.linspace(-0.5, 0.5, 1000)
    x_up, x_down = [], []
    for r in r_vals:
        coeffs = [-a, 0, b, r]
        roots = np.roots(coeffs)
        real_roots = np.real(roots[np.abs(np.imag(roots)) < 1e-8])
        real_roots = real_roots[(real_roots >= -0.5) & (real_roots <= 1.5)]
        if real_roots.size == 0:          # 防止空数组
            x_down.append(np.nan)
            x_up.append(np.nan)
            continue
        if len(real_roots) >= 2:          # 双稳
            x_down.append(real_roots.min())
            x_up.append(real_roots.max())
        else:                             # 单稳
            x_down.append(real_roots[0])
            x_up.append(real_roots[0])
    plt.figure(figsize=(4, 3))
    plt.plot(r_vals, x_up, 'r', lw=2, label='upper')
    plt.plot(r_vals, x_down, 'r', lw=2, label='lower')
    plt.axvline(r_crit, ls='--', c='k')
    plt.axvline(-r_crit, ls='--', c='k')
    plt.fill_between(r_vals, x_down, x_up, color='blue', alpha=0.1)
    plt.xlabel('r'); plt.ylabel('x'); plt.title('Fig.1  Cusp bifurcation')
    plt.tight_layout(); plt.show()

# -------------------------------------------------
# 网络构建
# -------------------------------------------------
def make_er(N, kavg, seed=42):
    np.random.seed(seed)
    p = kavg/(N-1)
    A = np.random.rand(N,N)<p
    A = np.triu(A,1)
    A = A + A.T
    return A.astype(int)

def make_ws(N, k, p, seed=42):
    # Watts-Strogatz 简化版
    np.random.seed(seed)
    A = np.zeros((N,N), int)
    for i in range(N):
        for j in range(1, k//2+1):
            A[i, (i+j)%N] = A[i, (i-j)%N] = 1
    # rewire
    for i in range(N):
        for j in range(i+1, N):
            if A[i,j] and np.random.rand()<p:
                A[i,j]=A[j,i]=0
                new = np.random.randint(N)
                A[i,new]=A[new,i]=1
    return A

def make_ba(N, m, seed=42):
    np.random.seed(seed)
    A = np.zeros((N,N), int)
    deg = np.zeros(N)
    # 初始完全图
    for i in range(m+1):
        for j in range(i+1, m+1):
            A[i,j]=A[j,i]=1
    deg[:m+1] = m
    for i in range(m+1, N):
        targets = np.random.choice(N, m, p=deg/deg.sum(), replace=False)
        for t in targets:
            A[i,t]=A[t,i]=1
        deg[i] = m
        deg[targets] += 1
    return A

# -------------------------------------------------
# 动力学方程（Eq.2）
# -------------------------------------------------
def dyn(x, t, A, eps1, eps2, r_vec):
    N = len(x)
    dx = np.zeros_like(x)
    # 高阶项预计算：B_ijk = A_ij A_jk A_ki
    # 实用技巧：先算 x_j x_k 再收缩
    xx = np.outer(x, x)
    # 对每一个 i，计算 sum_{j,k} B_ijk x_j x_k
    ho = np.zeros(N)
    for i in range(N):
        mask = A[i].astype(bool)
        neigh = np.where(mask)[0]
        if len(neigh)<2: continue
        # 提取子矩阵
        sub = A[np.ix_(neigh, neigh)]
        xx_sub = xx[np.ix_(neigh, neigh)]
        tri = sub * sub.T * np.triu(sub,1)   # 保证三角形
        ho[i] = eps2 * np.tensordot(tri, xx_sub)
    # 主方程
    for i in range(N):
        dx[i] = -a*(x[i]-x0)**3 + b*(x[i]-x0) + r_vec[i] + eps1*np.dot(A[i],x) + ho[i]
    return dx

# -------------------------------------------------
# Fig.2  时间级联（ER/BA/WS）
# -------------------------------------------------
def fig2():
    eps1 = 0.05
    r_vec = np.zeros(N); r_vec[0]=0.2
    nets = {'ER': make_er(N,6),
            'BA': make_ba(N,2),
            'WS': make_ws(N,6,0.2)}
    eps2s = {'ER':0.5, 'BA':0.5, 'WS':0.3}
    plt.figure(figsize=(9,3))
    for idx, (name,A) in enumerate(nets.items()):
        eps2 = eps2s[name]
        x0init = np.zeros(N)
        traj = odeint(dyn, x0init, tspan, args=(A,eps1,eps2,r_vec))
        F = (traj>0.5).sum(axis=1)/N
        plt.subplot(1,3,idx+1)
        plt.plot(tspan, F, lw=2)
        plt.title(f'Fig.2 {name}'); plt.xlabel('time'); plt.ylabel('F')
    plt.tight_layout(); plt.show()

# -------------------------------------------------
# Fig.3  参数相图（ER）
# -------------------------------------------------
def fig3():
    kavg = 4
    M = 20
    eps1_range = np.linspace(0, 0.2, M)
    eps2_range = np.linspace(0, 0.5, M)
    result = np.zeros((M,M))
    for i, e1 in enumerate(eps1_range):
        for j, e2 in enumerate(eps2_range):
            cnt = 0
            for run in range(20):
                A = make_er(N,kavg,seed=run)
                r_vec = np.zeros(N); r_vec[0]=0.2
                x0init = np.zeros(N)
                traj = odeint(dyn, x0init, tspan, args=(A,e1,e2,r_vec))
                if (traj[-1]>0.5).sum()>=4: cnt+=1
            result[j,i] = cnt/20
    plt.figure(figsize=(4,3))
    plt.imshow(result, aspect='auto', origin='lower', cmap='RdBu_r', vmin=0, vmax=1,
               extent=[eps1_range[0], eps1_range[-1], eps2_range[0], eps2_range[-1]])
    plt.colorbar(label='cascade prob')
    plt.xlabel('ε₁'); plt.ylabel('ε₂'); plt.title('Fig.3  Phase diagram')
    plt.tight_layout(); plt.show()

# -------------------------------------------------
# Fig.4  分叉图（N=3 全局耦合）
# -------------------------------------------------
def fig4():
    # 全局耦合
    A = np.ones((3,3)) - np.eye(3)
    r_vec = np.array([0.2, 0, 0])
    # 延续参数：eps1 或 eps2
    def cont(eps_range, which_par, eps_fix):
        branch = []
        xinit = np.array([0,0,0])
        for ep in eps_range:
            if which_par=='e1':
                traj = odeint(dyn, xinit, np.linspace(0,100,5000), args=(A, ep, eps_fix, r_vec))
            else:
                traj = odeint(dyn, xinit, np.linspace(0,100,5000), args=(A, eps_fix, ep, r_vec))
            xss = traj[-1]
            branch.append(xss.copy())
            xinit = xss     # 伪弧长
        return np.array(branch)
    plt.figure(figsize=(10,4))
    # 三列：仅e1 / 混合 / 仅e2
    eps1_r = np.linspace(0, 0.3, 200)
    eps2_r = np.linspace(0, 1.0, 200)
    # 第1列
    B1 = cont(eps1_r, 'e1', 0)
    plt.subplot(2,3,1); plt.plot(eps1_r, B1[:,0], 'r', lw=2); plt.title('x1 (ε₂=0)'); plt.xlabel('ε₁')
    plt.subplot(2,3,4); plt.plot(eps1_r, B1[:,1], 'r', lw=2); plt.title('x2,x3 (ε₂=0)'); plt.xlabel('ε₁')
    # 第2列
    B2 = cont(eps1_r, 'e1', 0.2)
    plt.subplot(2,3,2); plt.plot(eps1_r, B2[:,0], 'r', lw=2); plt.title('x1 (ε₂=0.2)'); plt.xlabel('ε₁')
    plt.subplot(2,3,5); plt.plot(eps1_r, B2[:,1], 'r', lw=2); plt.title('x2,x3 (ε₂=0.2)'); plt.xlabel('ε₁')
    # 第3列
    B3 = cont(eps2_r, 'e2', 0)
    plt.subplot(2,3,3); plt.plot(eps2_r, B3[:,0], 'r', lw=2); plt.title('x1 (ε₁=0)'); plt.xlabel('ε₂')
    plt.subplot(2,3,6); plt.plot(eps2_r, B3[:,1], 'r', lw=2); plt.title('x2,x3 (ε₁=0)'); plt.xlabel('ε₂')
    plt.suptitle('Fig.4  Bifurcation diagrams')
    plt.tight_layout(); plt.show()

# -------------------------------------------------
# 主程序：依次弹出 4 张图
# -------------------------------------------------
if __name__=='__main__':
    fig1()
    fig2()
    fig3()
    fig4()