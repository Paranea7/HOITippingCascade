#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# -------------------- 参数 --------------------
c = np.array([0.4, 0.0, 0.0])
e = 0.2
ds0 = 0.01
ds_min, ds_max = 1e-4, 0.05
max_arclen = 3.0
d_min, d_max = -0.6, 0.6
# --------------------------------------------

def rhs(x):
    x1, x2, x3 = x
    d_mat = d * (1 - np.eye(3))
    ho = e * np.array([x2*x3, x1*x3, x1*x2])
    return -x**3 + x + c + d_mat @ x + ho

def F(x):
    return rhs(x)

def jac(x):
    x1, x2, x3 = x
    J = np.zeros((3, 3))
    for i in range(3):
        J[i, i] = -3*x[i]**2 + 1
    J += d * (1 - np.eye(3))

    J[0, 1] += e*x3; J[0, 2] += e*x2
    J[1, 0] += e*x3; J[1, 2] += e*x1
    J[2, 0] += e*x2; J[2, 1] += e*x1
    return J

def newton(x, tol=1e-10, maxit=20):
    for _ in range(maxit):
        dx = np.linalg.solve(jac(x), -F(x))
        x = x + dx
        if np.linalg.norm(dx) < tol:
            break
    return x

def stability(x):
    """特征值判断稳定性"""
    eigs = np.linalg.eigvals(jac(x))
    if np.max(eigs.real) < 0:
        return 1   # 稳定
    else:
        return 0   # 不稳定

def pseudo_arc_continuation(x0, d0, ds0):
    global d

    x, d = x0.copy(), d0
    ds = ds0

    x = newton(x)

    X_hist = [x.copy()]
    D_hist = [d]
    S_hist = [stability(x)]

    # --- 初始切向量 ---
    eps = 1e-6
    d_plus = d + eps
    d = d_plus; x_plus = newton(x.copy())
    d = d - eps; x_minus = newton(x.copy())
    d = D_hist[-1]

    tangent = np.hstack([(x_plus - x_minus) / (2*eps), 1.0])
    tangent /= np.linalg.norm(tangent)

    arclen = 0.0
    while arclen < max_arclen:

        if d < d_min or d > d_max:
            break

        # --- 预测 ---
        x_pred = x + ds * tangent[:3]
        d_pred = d + ds * tangent[3]
        d = d_pred

        # --- 修正（Newton + 伪弧长约束） ---
        for _ in range(15):

            if d < d_min or d > d_max:
                break

            r = F(x_pred)
            J = jac(x_pred)

            J_aug = np.hstack([J, np.zeros((3, 1))])
            aug = np.vstack([J_aug, tangent])
            rhs_vec = np.hstack([-r, 0.0])
            delta = np.linalg.lstsq(aug, rhs_vec, rcond=None)[0]

            x_pred += delta[:3]
            d += delta[3]

            if np.linalg.norm(delta) < 1e-10:
                break

        x = x_pred

        # --- 新切向量 ---
        eps = 1e-6
        d_plus = d + eps; d = d_plus; x_plus = newton(x.copy())
        d = d - 2*eps; x_minus = newton(x.copy())
        d = d + eps

        tangent = np.hstack([(x_plus - x_minus)/(2*eps), 1.0])
        tangent /= np.linalg.norm(tangent)

        # --- 记录 ---
        X_hist.append(x.copy())
        D_hist.append(d)
        S_hist.append(stability(x))

        arclen += abs(ds)

        if np.linalg.norm(delta) < 5e-11:
            ds = min(ds * 1.5, ds_max)
        elif np.linalg.norm(delta) > 1e-9:
            ds = max(ds / 2, ds_min)

    return np.array(D_hist), np.array(X_hist), np.array(S_hist)

def detect_sn_points(D, X, S):
    """自动检测鞍结点 SN（折点），返回 SN 的 d 和 x."""
    sn_d = []
    sn_x = []

    # 稳定=1, 不稳定=0
    for i in range(len(D) - 1):
        if S[i] != S[i+1]:  # 稳定性发生变化 → SN
            # 线性插值求临界点
            lam_i = np.max(np.linalg.eigvals(jac(X[i])).real)
            lam_j = np.max(np.linalg.eigvals(jac(X[i+1])).real)

            t = lam_i / (lam_i - lam_j)  # 插值比例

            d_sn = D[i] + t * (D[i+1] - D[i])
            x_sn = X[i] + t * (X[i+1] - X[i])

            sn_d.append(d_sn)
            sn_x.append(x_sn)

    return np.array(sn_d), np.array(sn_x)
# ===================================================
# 主程序
# ===================================================
if __name__ == "__main__":

    d_start = 0.0
    x0 = np.array([-0.6, -0.6, -0.6])

    d_pos, X_pos, S_pos = pseudo_arc_continuation(x0, d_start, ds0)
    d_neg, X_neg, S_neg = pseudo_arc_continuation(x0, d_start, -ds0)

    D = np.hstack([d_neg[::-1], d_pos[1:]])
    X = np.vstack([X_neg[::-1], X_pos[1:]])
    S = np.hstack([S_neg[::-1], S_pos[1:]])
    # ---- 自动检测 SN 折点 ----
    sn_d, sn_x = detect_sn_points(D, X, S)
    print("Detected SN points at d =", sn_d)
    # ===================================================
    # 图 1：x1 随 d 的稳定 / 不稳定
    # ===================================================
    plt.figure(figsize=(6, 4))
    for i in range(len(D)-1):
        color = 'black' if S[i] == 1 else 'red'
        plt.plot(D[i:i+2], X[i:i+2, 0], color=color, linewidth=2)
    plt.xlabel("pairwise coupling d")
    plt.ylabel("x1 steady state")
    plt.title("x1: stable (black) / unstable (red)")
    plt.grid(True)
    plt.tight_layout()
    plt.scatter(sn_d, sn_x[:, 0], s=60, c='yellow', edgecolors='black', zorder=5, label='SN')
    plt.legend()
    # ===================================================
    # 图 2：x2, x3 的稳定 / 不稳定
    # ===================================================
    plt.figure(figsize=(6, 4))
    for i in range(len(D)-1):
        color = 'black' if S[i] == 1 else 'red'
        plt.plot(D[i:i+2], X[i:i+2, 1], color=color, linewidth=2)
        plt.plot(D[i:i+2], X[i:i+2, 2], color=color, linewidth=2, linestyle='--')
    plt.xlabel("pairwise coupling d")
    plt.ylabel("x2, x3 steady state")
    plt.title("x2, x3: stable (black) / unstable (red)")
    plt.grid(True)
    plt.tight_layout()
    plt.scatter(sn_d, sn_x[:, 1], s=60, c='yellow', edgecolors='black', zorder=5)
    plt.scatter(sn_d, sn_x[:, 2], s=60, c='yellow', edgecolors='black', zorder=5)
    plt.show()