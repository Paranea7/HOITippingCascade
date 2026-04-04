import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import fsolve
from scipy.stats import norm, skewnorm


# ==========================================
# 1. 全局 PRL 風格配置 (保持不動)
# ==========================================
def set_prl_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "lines.linewidth": 1.5,
        "figure.dpi": 150
    })


set_prl_style()
colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
SIG_U, SIG_D, H_C = 0.12, 0.05, 0.3849


# ==========================================
# 2. 核心計算引擎 (重構：一切依賴 m 的映射)
# ==========================================

def get_theory_state(m, mu_d, sig_base):
    """核心映射函數：從 m 導出系統所有統計狀態"""
    M = mu_d * m
    # 估算二階矩 q 以閉合 Gamma 的計算 (q = m^2 + Var)
    q = m ** 2 + (SIG_U ** 2 + sig_base ** 2)
    Gamma = np.sqrt(SIG_U ** 2 + SIG_D ** 2 * q + sig_base ** 2)

    # 求解穩態根
    roots = np.roots([1, 0, -1, -M])
    real_roots = np.sort(roots[np.isreal(roots)].real)

    if len(real_roots) == 3:
        xn, xp = real_roots[0], real_roots[-1]
    else:
        xn = xp = real_roots[0]  # 單穩態

    # 計算 Tipping Rate phi
    if M > H_C:
        phi = 1.0
    elif M < -H_C:
        phi = 0.0
    else:
        phi = 0.5 * (1 + erf((M - H_C) / (np.sqrt(2) * Gamma)))

    return phi, xn, xp, Gamma, M


def solve_theory_unified(mu_d, sig_base):
    """迭代求解 m 的自洽點"""

    def consistency_res(m):
        phi, xn, xp, _, _ = get_theory_state(m, mu_d, sig_base)
        target_m = (1 - phi) * xn + phi * xp
        return m - target_m

    # 嘗試從負分支 (-1) 或正分支 (1) 求解，視 mu_d 符號而定
    m_f = fsolve(consistency_res, x0=-1.0 if mu_d < 0.2 else 1.0)[0]
    phi_f, xn_f, xp_f, Gamma_f, M_f = get_theory_state(m_f, mu_d, sig_base)
    return m_f, phi_f, Gamma_f, M_f, xn_f, xp_f


# ==========================================
# 3. 模擬引擎 (保持不動)
# ==========================================
def solve_system_batch(mu_d, sig_base, S=800, dt=0.05, steps=3000, n_batches=10):
    ms, phis = [], []
    gamma_total = np.sqrt(SIG_U ** 2 + SIG_D ** 2 + sig_base ** 2)
    for _ in range(n_batches):
        xi_i = np.random.normal(0, gamma_total, S)
        x = np.full(S, -1.0 if mu_d < 0.2 else 1.0) + np.random.normal(0, 0.02, S)
        for _ in range(steps):
            m = np.mean(x)
            x += (x - x ** 3 + mu_d * m + xi_i) * dt
            x = np.clip(x, -2.5, 2.5)
        ms.append(np.mean(x));
        phis.append(np.sum(x > 0) / S)
    return np.mean(ms), np.std(ms), np.mean(phis), np.std(phis)


def get_combined_pdf_data(mu_d, sig_base, S=2000, n_batches=5):
    all_x = []
    dt = 0.05
    gamma_total = np.sqrt(SIG_U ** 2 + SIG_D ** 2 + sig_base ** 2)
    for _ in range(n_batches):
        xi_i = np.random.normal(0, gamma_total, S)
        x = np.full(S, -1.0 if mu_d < 0.2 else 1.0) + np.random.normal(0, 0.02, S)
        for _ in range(2500):
            m = np.mean(x)
            x += (x - x ** 3 + mu_d * m + xi_i) * dt
        all_x.extend(x)
    return np.array(all_x)


# ==========================================
# 4. 繪圖邏輯 (保持原有結構，僅更新變量映射)
# ==========================================
fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(15, 4.5))
mu_d_range = np.linspace(-0.6, 0.7, 12)
mu_d_theo = np.linspace(-0.6, 0.7, 100)

# --- A. Mean State m ---
for i, sig in enumerate([0.1, 0.25]):
    m_th = [solve_theory_unified(md, sig)[0] for md in mu_d_theo]
    ax_a.plot(mu_d_theo, m_th, '-', color=colors[i], alpha=0.7, label=rf'Theo. $\sigma={sig}$')
    sim_res = [solve_system_batch(md, sig, n_batches=8) for md in mu_d_range]
    m_avg, m_err = [r[0] for r in sim_res], [r[1] for r in sim_res]
    ax_a.errorbar(mu_d_range, m_avg, yerr=m_err, fmt='o', color=colors[i], mfc='none', capsize=3,
                  label=rf'Sim. $\sigma={sig}$')
ax_a.set_title("A. Mean State $m$");
ax_a.set_xlabel(r"$\mu_d$");
ax_a.set_ylabel(r"$m$")
ax_a.legend(frameon=False, fontsize=8)

# --- B. Tipping Rate phi ---
for i, sig in enumerate([0.15, 0.3]):
    phi_th = [solve_theory_unified(md, sig)[1] for md in mu_d_theo]
    ax_b.plot(mu_d_theo, phi_th, '-', color=colors[i + 1], alpha=0.7, label=rf'Theo. $\sigma={sig}$')
    sim_res = [solve_system_batch(md, sig, n_batches=8) for md in mu_d_range]
    p_avg, p_err = [r[2] for r in sim_res], [r[3] for r in sim_res]
    ax_b.errorbar(mu_d_range, p_avg, yerr=p_err, fmt='s', ms=4, color=colors[i + 1], mfc='none', capsize=3,
                  label=rf'Sim. $\sigma={sig}$')
ax_b.set_title(r"B. Tipping Rate $\phi$");
ax_b.set_xlabel(r"$\mu_d$");
ax_b.set_ylabel(r"$\phi$")

# --- C. PDF Evolution (嚴格映射自 m) ---
x_axis = np.linspace(-2.2, 2.2, 1000)
for i, md in enumerate([-0.4, 0.2]):
    combined_x = get_combined_pdf_data(md, 0.0, S=2000, n_batches=15)
    ax_c.hist(combined_x, bins=100, density=True, alpha=0.3, color=colors[i], label=rf'$\mu_d={md}$')

    # 獲取理論計算出的 m_f 及其對應的所有分佈參數
    m_f, phi_f, Gamma_f, M_f, xn_f, xp_f = solve_theory_unified(md, 0.0)


    def improved_pdf(mu_pos, gamma):
        curv = np.abs(3 * mu_pos ** 2 - 1)
        base_sigma = gamma / np.sqrt(2 * curv)
        alpha = -2 if mu_pos > 0 else 0  # 保持你原來的偏態邏輯
        return skewnorm.pdf(x_axis, alpha, loc=mu_pos + (0.05 if mu_pos > 0 else 0), scale=base_sigma * 0.75)


    pdf_theo = (1 - phi_f) * improved_pdf(xn_f, Gamma_f) + phi_f * improved_pdf(xp_f, Gamma_f)
    ax_c.plot(x_axis, pdf_theo, '--', color=colors[i], lw=1.5, label=f'Mapped from $m={m_f:.2f}$')

ax_c.set_title("C. PDF Evolution");
ax_c.set_xlabel("$x_i$");
ax_c.legend(frameon=False, fontsize=8)
plt.tight_layout()
plt.show()
