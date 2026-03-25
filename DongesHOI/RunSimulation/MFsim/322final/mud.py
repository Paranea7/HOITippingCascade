import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import fsolve


# ==========================================
# 1. 全局 PRL 風格配置
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

# 系統核心參數
SIG_U = 0.12
SIG_D = 0.05
SIG_E = 0.0
H_C = 0.3849


# ==========================================
# 2. 核心解析工具 (精確根)
# ==========================================
def get_x_roots_exact(M):
    roots = np.roots([1, 0, -1, -M])
    real_roots = np.sort(roots[np.isreal(roots)].real)
    if len(real_roots) == 3:
        return real_roots[0], real_roots[-1]
    return (real_roots[0], real_roots[0]) if M < 0 else (real_roots[-1], real_roots[-1])


def solve_theory_unified(mu_d, sig_base=0.0):
    def equations(vars):
        m, q = vars
        M = mu_d * m
        Gamma = np.sqrt(SIG_U ** 2 + SIG_D ** 2 * q + sig_base ** 2)
        phi = 0.5 * (1 + erf((M - H_C) / (np.sqrt(2) * Gamma)))
        xn, xp = get_x_roots_exact(M)
        res_m = m - ((1 - phi) * xn + phi * xp)
        res_q = q - ((1 - phi) * xn ** 2 + phi * xp ** 2)
        return [res_m, res_q]

    sol = fsolve(equations, x0=[-1.0, 1.0])
    m_f, q_f = sol
    M_f = mu_d * m_f
    G_f = np.sqrt(SIG_U ** 2 + SIG_D ** 2 * q_f + sig_base ** 2)
    phi_f = 0.5 * (1 + erf((M_f - H_C) / (np.sqrt(2) * G_f)))
    return m_f, phi_f


# ==========================================
# 3. 核心動力學引擎 (支援多組重複實驗)
# ==========================================
def solve_system_ensemble(mu_d, sig_base, S=1000, dt=0.05, steps=4000, x0=-1.0, n_trials=50):
    """
    執行 n_trials 次獨立模擬並取平均值
    """
    m_final_list = []
    phi_list = []

    for _ in range(n_trials):
        gamma_init = np.sqrt(SIG_U ** 2 + SIG_D ** 2 + sig_base ** 2)
        xi_i = np.random.normal(0, gamma_init, S)
        x = np.full(S, x0, dtype=float) + np.random.normal(0, 0.01, S)

        for t in range(steps):
            m = np.mean(x)
            drift = x - x ** 3 + mu_d * m + xi_i
            x += drift * dt
            x = np.clip(x, -3, 3
                        )

        m_final_list.append(np.mean(x))
        phi_list.append(np.sum(x > 0) / S)

    return np.mean(m_final_list), np.mean(phi_list)


def solve_system_single_path(mu_d, sig_base, S=1000, dt=0.05, steps=4000, x0=-1.0):
    """ 用於繪製時間序列的單次模擬 """
    gamma_init = np.sqrt(SIG_U ** 2 + SIG_D ** 2 + sig_base ** 2)
    xi_i = np.random.normal(0, gamma_init, S)
    x = np.full(S, x0, dtype=float) + np.random.normal(0, 0.01, S)
    m_h = np.zeros(steps)
    for t in range(steps):
        m = np.mean(x)
        x += (x - x ** 3 + mu_d * m + xi_i) * dt
        x = np.clip(x, -2.2, 2.2)
        m_h[t] = m
    return m_h, x


# ==========================================
# 4. 繪圖與分析
# ==========================================
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 3)
n_ens = 10  # 設定重複實驗次數

# --- A. Collective State ---
ax_a = fig.add_subplot(gs[0, 0])
mu_d_range = np.linspace(-0.5, 0.7, 12)
mu_d_theo = np.linspace(-0.5, 0.7, 50)
for i, sig in enumerate([0.1, 0.25]):
    m_th = [solve_theory_unified(md, sig)[0] for md in mu_d_theo]
    ax_a.plot(mu_d_theo, m_th, '-', color=colors[i], alpha=0.6, label=rf'Theo. $\sigma={sig}$')
    # 執行系綜平均
    ms_sim = [solve_system_ensemble(md, sig, S=800, n_trials=n_ens)[0] for md in mu_d_range]
    ax_a.plot(mu_d_range, ms_sim, 'o', mfc='none', color=colors[i], label=rf'Sim. $\sigma={sig}$')
ax_a.set_title("A. Mean State $m$ (Ensemble Avg)")
ax_a.set_xlabel(r"$\mu_d$");
ax_a.set_ylabel(r"$m$");
ax_a.legend(frameon=False, fontsize=8)

# --- B. Tipping Rate phi ---
ax_b = fig.add_subplot(gs[0, 1])
mu_d_fine = np.linspace(-0.5, 0.7, 16)
for i, sig in enumerate([0.15, 0.3]):
    sim_phis = []
    ana_phis = []
    for md in mu_d_fine:
        m_avg, p_avg = solve_system_ensemble(md, sig, S=1000, n_trials=n_ens)
        sim_phis.append(p_avg)
        ana_phis.append(solve_theory_unified(md, sig)[1])
    ax_b.scatter(mu_d_fine, sim_phis, marker='o', s=20, edgecolors=colors[i + 1], facecolors='none',
                 label=rf'Sim. $\sigma={sig}$')
    ax_b.plot(mu_d_fine, ana_phis, '-', color=colors[i + 1], lw=1.2, label=rf'Theo. $\sigma={sig}$')
ax_b.set_title(r"B. Tipping Rate $\phi$")
ax_b.set_xlabel(r"$\mu_d$");
ax_b.legend(frameon=False, fontsize=8)

# --- C. PDF Evolution (保持單次展示) ---
ax_c = fig.add_subplot(gs[0, 2])
for i, md in enumerate([-0.6, -0.2, 0.2]):
    _, final_x = solve_system_single_path(md, 0.15, S=2500)
    ax_c.hist(final_x, bins=40, density=True, alpha=0.4, color=colors[i], label=rf'$\mu_d={md}$')
ax_c.set_title("C. PDF Evolution")
ax_c.legend(frameon=False, fontsize=8)

# --- D. 有限尺寸效應 ---
ax_d = fig.add_subplot(gs[1, 0])
for i, S_val in enumerate([500, 2000, 5000]):
    md_range_d = np.linspace(-0.6, 0.2, 8)
    ms_d = [solve_system_ensemble(md, 0.1, S=S_val, n_trials=n_ens)[0] for md in md_range_d]
    ax_d.plot(md_range_d, ms_d, 'v-', color=colors[i], markersize=4, label=rf'$S={S_val}$')
ax_d.set_title("D. Size Effect")
ax_d.set_xlabel(r"$\mu_d$");
ax_d.legend(frameon=False, fontsize=8)

# --- E. 時間演化 ---
ax_e = fig.add_subplot(gs[1, 1:])
dt = 0.05
times = np.arange(4000) * dt
for i, md in enumerate([-0.8, -0.4, 0.0]):
    m_h, _ = solve_system_single_path(md, 0.2, steps=4000)
    ax_e.plot(times, m_h, color=colors[i], label=rf'$\mu_d={md}$')
ax_e.axhline(-1.0, ls=':', color='black', alpha=0.4)
ax_e.set_xlabel('Time $t$');
ax_e.set_ylabel('$m(t)$')
ax_e.legend(frameon=False, ncol=3)

plt.tight_layout()
plt.show()
