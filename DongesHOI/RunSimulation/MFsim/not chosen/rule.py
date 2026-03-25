import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import fsolve
import warnings
import matplotlib as mpl

warnings.filterwarnings("ignore")

# ==========================================
# 1. PRL 投稿級配置
# ==========================================
mpl.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.linewidth": 1.0,
    "figure.dpi": 150,
})

# 系統核心參數
SIG_U = 0.12
MU_U = 0.0
S_LIST = [200, 1000, 5000]  # 不同規模對比
COLORS = ['#1f77b4', '#d62728', '#2ca02c']


# ==========================================
# 2. 修正後的解析理論引擎
# ==========================================
def get_stable_roots(M):
    roots = np.roots([-1, 0, 1, float(M)])
    real_roots = np.sort(roots[np.isreal(roots)].real)
    if len(real_roots) == 3:
        return float(real_roots[0]), float(real_roots[-1])
    else:
        val = float(real_roots[0]) if len(real_roots) > 0 else (1.0 if M > 0 else -1.0)
        return val, val


def solve_theory_m(mu_val, mode='mud'):
    H_c_det = 0.3849
    beta = 0.46 if mode == 'mud' else 0.42

    def equations(vars):
        m, q = float(vars[0]), float(vars[1])
        M = mu_val * m if mode == 'mud' else mu_val * q
        Gamma = SIG_U
        H_tilde = H_c_det - beta * Gamma
        xn, xp = get_stable_roots(M)
        phi = 0.5 * (1 + erf((M - H_tilde) / (np.sqrt(2) * Gamma)))

        res_m = m - ((1 - phi) * xn + phi * xp)
        res_q = q - ((1 - phi) * xn ** 2 + phi * xp ** 2)
        return [res_m, res_q]

    sol = fsolve(equations, x0=[-0.8, 1.0])
    return sol[0]


# ==========================================
# 3. 數據生成與繪圖
# ==========================================
md_range = np.linspace(-1.0, 0.4, 10)
md_theo = np.linspace(-1.0, 0.4, 100)
me_range = np.linspace(-0.4, 1.0, 10)
me_theo = np.linspace(-0.4, 1.0, 100)

fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2))

# --- (a) 兩體尺寸效應 (mu_d) ---
for i, S in enumerate(S_LIST):
    m_vals = []
    for md in md_range:
        u_vec = np.random.normal(MU_U, SIG_U, S)
        x = np.full(S, -1.0)
        for _ in range(6000):
            m_c = np.mean(x)
            x += (x - x ** 3 + u_vec + md * m_c) * 0.02
        m_vals.append(np.mean(x))
    axes[0].plot(md_range, m_vals, 'o', mfc='none', color=COLORS[i], markersize=4, label=rf'$S={S}$')

m_th_d = [solve_theory_m(m, 'mud') for m in md_theo]
axes[0].plot(md_theo, m_th_d, 'k--', lw=1)
axes[0].set_xlabel(r'Coupling strength $\mu_d$')
axes[0].set_ylabel(r'Order parameter $m$') # 僅保留左側標籤
axes[0].text(0.05, 0.9, '(a)', transform=axes[0].transAxes, fontweight='bold')
axes[0].legend(frameon=False, loc='upper right', handletextpad=0.2)

# --- (b) 三體尺寸效應 (mu_e) ---
for i, S in enumerate(S_LIST):
    m_vals = []
    for me in me_range:
        u_vec = np.random.normal(MU_U, SIG_U, S)
        x = np.full(S, -1.0)
        for _ in range(8000):
            m_c, q_c = np.mean(x), np.mean(x ** 2)
            x += (x - x ** 3 + u_vec + me * q_c) * 0.02
        m_vals.append(np.mean(x))
    axes[1].plot(me_range, m_vals, 's', mfc='none', color=COLORS[i], markersize=4, label=rf'$S={S}$')

m_th_e = [solve_theory_m(m, 'mue') for m in me_theo]
axes[1].plot(me_theo, m_th_e, 'k--', lw=1)
axes[1].set_xlabel(r'Coupling strength $\mu_e$')
axes[1].set_ylabel('') # 移除右側標籤
axes[1].text(0.05, 0.9, '(b)', transform=axes[1].transAxes, fontweight='bold')
axes[1].legend(frameon=False, loc='lower right', handletextpad=0.2)

plt.tight_layout()
plt.show()
