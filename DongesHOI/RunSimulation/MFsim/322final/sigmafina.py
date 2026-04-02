import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import fsolve
import warnings
import matplotlib as mpl

warnings.filterwarnings("ignore")


# ==========================================
# 1. PRL 投稿级风格配置
# ==========================================
def set_prl_style():
    mpl.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.linewidth": 0.8,
        "figure.dpi": 150,
    })


set_prl_style()
colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
SIG_U_CONST = 0.128  # 统一校准噪声强度


# ==========================================
# 2. 核心解析引擎：自洽映射模型
# ==========================================
def get_stable_roots(M):
    roots = np.roots([-1, 0, 1, float(M)])
    real_roots = np.sort(roots[np.isreal(roots)].real)
    if len(real_roots) == 3:
        return real_roots[0], real_roots[-1]
    return (real_roots[0], real_roots[0]) if M < 0 else (real_roots[-1], real_roots[-1])


def solve_theory(sig_val, mode='d', mu_u=0.0, mu_d=0.0, mu_e=0.0):
    H_c = 0.3849

    def equations(vars):
        m = float(vars[0])
        sd = sig_val if mode == 'd' else 0.0
        se = sig_val if mode == 'e' else 0.0
        M = mu_u + mu_d * m + mu_e * (m ** 2)
        # 校准点 1: 噪声 Gamma 严格随相干信号 m 缩放
        Gamma = np.sqrt(SIG_U_CONST ** 2 + sd ** 2 * m ** 2 + se ** 2 * m ** 4)
        xn, xp = get_stable_roots(M)
        phi = 0.5 * (1 + erf((M - H_c) / (np.sqrt(2) * Gamma)))
        return [m - ((1 - phi) * xn + phi * xp)]

    # 求解稳态 m
    sol = fsolve(equations, x0=[-0.8])
    m_f = sol[0]

    # 校准点 2: 使用自洽映射计算 Phi (phi = (m - xn) / (xp - xn))
    M_f = mu_u + mu_d * m_f + mu_e * (m_f ** 2)
    xn, xp = get_stable_roots(M_f)

    if abs(xp - xn) > 1e-4:
        phi_f = (m_f - xn) / (xp - xn)
    else:
        phi_f = 1.0 if m_f > 0 else 0.0

    return m_f, np.clip(phi_f, 0, 1)


# ==========================================
# 3. 核心物理引擎：微观动力学模拟
# ==========================================
def run_simulation(mode, sig_val, S=2500, steps=8000, mu_u=0.0, mu_d=0.0, mu_e=0.0):
    # 校准点 3: 模拟与理论使用完全一致的噪声基准
    u_i = np.random.normal(0, SIG_U_CONST, S)
    h_i = np.random.normal(0, sig_val, S)
    x = np.full(S, -1.0)
    dt = 0.02

    for _ in range(steps):
        m_c = np.mean(x)
        if mode == 'd':
            drift = x - x ** 3 + mu_u + (mu_d + h_i) * m_c + mu_e * (m_c ** 2)
        else:
            drift = x - x ** 3 + mu_u + mu_d * m_c + (mu_e + h_i) * (m_c ** 2)
        x += drift * dt

    return np.mean(x), np.sum(x > 0) / S, x


# ==========================================
# 4. 执行绘图 (2x3 Layout)
# ==========================================
fig, axes = plt.subplots(2, 3, figsize=(10, 6.0))
sig_range = np.linspace(0.01, 0.8, 12)
sig_theo = np.linspace(0.01, 0.8, 80)

for row, mode in enumerate(['d', 'e']):
    m_label = r"$\sigma_d$" if mode == 'd' else r"$\sigma_e$"

    m_sims, p_sims, final_xs = [], [], []
    for s in sig_range:
        ms, ps, xs = run_simulation(mode, s)
        m_sims.append(ms);
        p_sims.append(ps);
        final_xs.append(xs)

    m_ths, p_ths = zip(*[solve_theory(s, mode) for s in sig_theo])

    # Col 1: Order Parameter
    ax1 = axes[row, 0]
    ax1.plot(sig_theo, m_ths, '-', color=colors[row * 2], lw=1.5, label='Theory')
    ax1.plot(sig_range, m_sims, 'o', mfc='none', color=colors[row * 2], markersize=5, label='Sim.')
    ax1.set_ylabel(r'Order Parameter $m$')
    ax1.set_xlabel(f'{m_label}')
    ax1.legend(frameon=False, fontsize=8)
    ax1.text(0.05, 0.9, f"({chr(97 + row * 3)})", transform=ax1.transAxes, fontweight='bold')

    # Col 2: Occupancy Phi (Self-consistent Mapping)
    ax2 = axes[row, 1]
    ax2.plot(sig_theo, p_ths, '--', color=colors[row * 2 + 1], lw=1.5, label='Theory')
    ax2.plot(sig_range, p_sims, 's', mfc='none', color=colors[row * 2 + 1], markersize=5, label='Sim.')
    ax2.set_ylabel(r'Tipping Rate $\phi$')
    ax2.set_xlabel(f'{m_label}')
    ax2.legend(frameon=False, fontsize=8)
    ax2.text(0.05, 0.9, f"({chr(98 + row * 3)})", transform=ax2.transAxes, fontweight='bold')

    # Col 3: PDF Evolution
    ax3 = axes[row, 2]
    for i in [1, 5, 10]:
        ax3.hist(final_xs[i], bins=40, density=True, alpha=0.3, label=rf"$\sigma={sig_range[i]:.2f}$")
    ax3.set_xlabel(r'state $x_i$')
    ax3.set_ylabel('Density')
    ax3.legend(frameon=False, fontsize=7)
    ax3.text(0.05, 0.9, f"({chr(99 + row * 3)})", transform=ax3.transAxes, fontweight='bold')

plt.tight_layout()
plt.show()
