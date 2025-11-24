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
    d_mat = d*(1-np.eye(3))
    ho = e*np.array([x2*x3, x1*x3, x1*x2])
    return -x**3 + x + c + d_mat @ x + ho

def F(x):
    return rhs(x)

def jac(x):
    x1, x2, x3 = x
    J = np.zeros((3, 3))

    for i in range(3):
        J[i, i] = -3*x[i]**2 + 1

    J += d*(1-np.eye(3))

    J[0, 1] += e*x3;  J[0, 2] += e*x2
    J[1, 0] += e*x3;  J[1, 2] += e*x1
    J[2, 0] += e*x2;  J[2, 1] += e*x1

    return J

def newton(x, tol=1e-10, maxit=20):
    for _ in range(maxit):
        dx = np.linalg.solve(jac(x), -F(x))
        x = x + dx
        if np.linalg.norm(dx) < tol:
            break
    return x

def pseudo_arc_continuation(x0, d0, ds0):
    global d

    x, d = x0.copy(), d0
    ds = ds0

    x = newton(x)
    X_hist, d_hist = [x.copy()], [d]

    eps = 1e-6
    d_plus = d + eps
    d = d_plus;  x_plus = newton(x.copy())
    d = d - eps; x_minus = newton(x.copy())
    d = d_hist[-1]

    tangent = np.hstack([(x_plus - x_minus)/(2*eps), 1.0])
    tangent /= np.linalg.norm(tangent)

    arclen = 0.0
    while arclen < max_arclen:

        # --- 若 d 超出范围，停止 ---
        if d < d_min or d > d_max:
            break

        x_pred = x + ds * tangent[:3]
        d_pred = d + ds * tangent[3]
        d = d_pred

        for _ in range(15):

            # 再次检查 d 越界
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

        eps = 1e-6
        d_plus = d + eps
        d = d_plus; x_plus = newton(x.copy())
        d = d - 2*eps; x_minus = newton(x.copy())
        d = d + eps

        tangent = np.hstack([(x_plus - x_minus)/(2*eps), 1.0])
        tangent /= np.linalg.norm(tangent)

        X_hist.append(x.copy())
        d_hist.append(d)

        arclen += abs(ds)

        if np.linalg.norm(delta) < 5e-11:
            ds = min(ds * 1.5, ds_max)
        elif np.linalg.norm(delta) > 1e-9:
            ds = max(ds / 2, ds_min)

    return np.array(d_hist), np.array(X_hist)

# -------------------- 主程序 --------------------
if __name__ == "__main__":
    d_start = 0.0
    x0 = np.array([-0.6, -0.6, -0.6])

    d_pos, X_pos = pseudo_arc_continuation(x0, d_start, ds0)
    d_neg, X_neg = pseudo_arc_continuation(x0, d_start, -ds0)

    d_vals = np.hstack([d_neg[::-1], d_pos[1:]])
    X_vals = np.vstack([X_neg[::-1], X_pos[1:]])

    plt.figure(figsize=(5, 4))
    plt.plot(d_vals, X_vals[:, 1], 'r', lw=2, label='x2')
    plt.plot(d_vals, X_vals[:, 2], 'b', lw=2, label='x3')
    plt.xlabel('pairwise coupling d')
    plt.ylabel('steady state x_i')
    plt.title(f'3-node bifurcation (e={e}, c1=0.4)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()