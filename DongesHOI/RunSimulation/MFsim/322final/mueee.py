import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import fsolve
import warnings

warnings.filterwarnings("ignore")


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
# 增加色系以區分不同 sigma_e
colors_list = ['#1f77b4', '#d62728', '#2ca02c']

# 核心參數
SIG_U = 0.12
SIG_D = 0.05
H_C = 0.3849


# ==========================================
# 2. 核心解析工具 (加入 sig_e 參數)
# ==========================================
def get_x_roots_exact(M):
    roots = np.roots([1, 0, -1, -M])
    real_roots = np.sort(roots[np.isreal(roots)].real)
    if len(real_roots) == 3:
        return real_roots[0], real_roots[-1]
    return (real_roots[0], real_roots[0]) if M < 0 else (real_roots[-1], real_roots[-1])


def calculate_phi(mu_e, m, q, sig_e):
    M = mu_e * q
    # Gamma 現在動態依賴於傳入的 sig_e
    Gamma = np.sqrt(SIG_U ** 2 + SIG_D ** 2 * q + sig_e ** 2 * q ** 2)
    return 0.5 * (1 + erf((M - H_C) / (np.sqrt(2) * Gamma)))


def solve_theory_m(mu_e, sig_e):
    def eq(v):
        m, q = v
        phi = calculate_phi(mu_e, m, q, sig_e)
        xn, xp = get_x_roots_exact(mu_e * q)
        return [m - ((1 - phi) * xn + phi * xp), q - ((1 - phi) * xn ** 2 + phi * xp ** 2)]

    # 使用上一次解作為 guess 的邏輯在此簡化，通常 [-1, 1] 適合亞穩態追蹤
    sol = fsolve(eq, x0=[-1.0, 1.0])
    return sol[0], sol[1]


# ==========================================
# 3. 動力學引擎 (加入 sig_e 參數)
# ==========================================
def solve_system(mu_e, sig_e, S=1500, dt=0.05, steps=4000, x0=-1.0):
    Gamma_init = np.sqrt(SIG_U ** 2 + SIG_D ** 2 + sig_e ** 2)
    xi = np.random.normal(0, Gamma_init, S)
    x = np.full(S, x0, dtype=float) + np.random.normal(0, 0.01, S)

    m_h = np.zeros(steps)
    for t in range(steps):
        m, q = np.mean(x), np.mean(x ** 2)
        drift = x - x ** 3 + mu_e * q + xi
        x += drift * dt
        x = np.clip(x, -2.5, 2.5)
        m_h[t] = m
    return m_h, x


# ==========================================
# 4. 繪圖與分析
# ==========================================
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 3)

# 設定要對比的 sigma_e 取值
sig_e_samples = [0.05,  0.25]

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])

mu_e_range_sim = np.linspace(0.0, 1.2, 14)
mu_e_range_theo = np.linspace(0.0, 1.2, 80)

for idx, se in enumerate(sig_e_samples):
    # --- A. Collective State 數據 ---
    m_theo = [solve_theory_m(me, se)[0] for me in mu_e_range_theo]
    ms_sim = []
    # --- B. Tipping Rate 數據 ---
    phi_sim = []
    phi_ana = []

    for me in mu_e_range_sim:
        mh, fx = solve_system(me, se)
        ms_sim.append(mh[-500:].mean())
        phi_sim.append(np.sum(fx > 0) / len(fx))

        # 理論 phi
        mt, qt = solve_theory_m(me, se)
        phi_ana.append(calculate_phi(me, mt, qt, se))

    # 繪製 A 圖
    ax_a.plot(mu_e_range_theo, m_theo, '-', color=colors_list[idx], alpha=0.8)
    ax_a.plot(mu_e_range_sim, ms_sim, 'o', mfc='none', color=colors_list[idx], label=rf"$\sigma_e={se}$")

    # 繪製 B 圖
    ax_b.plot(mu_e_range_theo, [calculate_phi(me, *solve_theory_m(me, se), se) for me in mu_e_range_theo],
              '-', color=colors_list[idx], alpha=0.8)
    ax_b.plot(mu_e_range_sim, phi_sim, 's', mfc='none', markersize=4, color=colors_list[idx], label=rf"$\sigma_e={se}$")

# 修飾 A, B 圖
ax_a.set_title(r"A. Mean State $m$ (Multi-$\sigma_e$)")
ax_a.set_xlabel(r"$\mu_e$");
ax_a.set_ylabel("$m$")
ax_a.legend(frameon=False, fontsize=8)

ax_b.set_title(r"B. Tipping Rate $\phi$ (Multi-$\sigma_e$)")
ax_b.set_xlabel(r"$\mu_e$");
ax_b.set_ylabel(r"$\phi$")
ax_b.legend(frameon=False, fontsize=8)

# --- C. PDF 演化 (保持原邏輯，使用中間值 sigma_e) ---
ax_c = fig.add_subplot(gs[0, 2])
for i, me in enumerate([0.3, 0.6, 0.9]):
    _, final_x = solve_system(me, sig_e=0.15)
    ax_c.hist(final_x, bins=40, density=True, alpha=0.4, color=colors_list[i], label=rf'$\mu_e={me}$')
ax_c.set_title(r"C. PDF ($\sigma_e=0.15$)")
ax_c.legend(frameon=False)

# --- D. 有限尺寸效應 (使用中間值 sigma_e) ---
ax_d = fig.add_subplot(gs[1, 0])
for i, S_val in enumerate([500, 2000, 5000]):
    me_range_d = np.linspace(0.4, 1.0, 8)
    ms_d = [solve_system(me, sig_e=0.15, S=S_val)[0][-500:].mean() for me in me_range_d]
    ax_d.plot(me_range_d, ms_d, 'v-', color=colors_list[i], markersize=4, label=rf'$S={S_val}$')
ax_d.set_title(r"D. Size Effect ($\sigma_e=0.15$)")
ax_d.set_xlabel(r"$\mu_e$")
ax_d.legend(frameon=False)

# --- E. 時間演化 ---
ax_e = fig.add_subplot(gs[1, 1:])
dt, steps = 0.02, 6000
times = np.arange(steps) * dt
for i, me in enumerate([0.2, 0.5, 0.8]):
    mh, _ = solve_system(me, sig_e=0.15, steps=steps)
    ax_e.plot(times, mh, color=colors_list[i], label=rf'$\mu_e={me}$')
ax_e.axhline(-1.0, ls=':', color='black', alpha=0.4)
ax_e.set_title(r"E. Temporal Paths ($\sigma_e=0.15$)")
ax_e.set_xlabel('Time $t$');
ax_e.set_ylabel('$m(t)$')
ax_e.legend(frameon=False, ncol=3)

plt.tight_layout()
plt.show()
