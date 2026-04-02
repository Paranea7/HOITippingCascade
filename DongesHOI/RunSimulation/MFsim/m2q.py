import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import fsolve

# ==========================================
# 1. 物理参数与风格配置
# ==========================================
SIG_U, H_C = 0.128, 0.3849
plt.rcParams.update({"font.family": "serif", "font.size": 9, "figure.dpi": 150})


def self_consistent_equations(vars, mu_d, mu_e, sigma_e, mode="m2"):
    m, q = vars
    q = max(q, 1e-6)

    # --- 核心差异：均值场 M 的定义 ---
    if mode == "m2":
        M = mu_d * m + mu_e * (m ** 2)  # 严格理论模式：仅相干部分贡献场
    else:
        M = mu_d * m + mu_e * q  # 混淆模式：将总功率视为驱动力

    Gamma = np.sqrt(SIG_U ** 2 + (sigma_e ** 2 * q ** 2))

    # 翻转概率与分支位置
    phi = 0.5 * (1 + erf((M - H_C) / (np.sqrt(2) * Gamma + 1e-12)))
    xn, xp = -1.0 + M / 2.0, 1.0 + M / 2.0

    return [m - ((1 - phi) * xn + phi * xp), q - ((1 - phi) * xn ** 2 + phi * xp ** 2)]


def generate_phase_data(mode="m2", res=50):
    mu_d_vec = np.linspace(-0.5, 1.0, res)
    mu_e_vec = np.linspace(0.0, 1.0, res)
    grid_phi = np.zeros((res, res))

    for i, me in enumerate(mu_e_vec):
        curr_sol = np.array([-1.0, 1.0])  # 始终从负分支追踪
        for j, md in enumerate(mu_d_vec):
            sol = fsolve(self_consistent_equations, x0=curr_sol, args=(md, me, 0.2, mode))
            # 计算最终 phi
            m_f, q_f = sol
            M_f = md * m_f + me * (m_f ** 2) if mode == "m2" else md * m_f + me * q_f
            G_f = np.sqrt(SIG_U ** 2 + (0.2 ** 2 * q_f ** 2))
            phi_f = 0.5 * (1 + erf((M_f - H_C) / (np.sqrt(2) * G_f + 1e-12)))
            grid_phi[i, j] = phi_f
            curr_sol = sol
    return mu_d_vec, mu_e_vec, grid_phi


# ==========================================
# 2. 执行计算与绘图
# ==========================================
print("Computing Phase Diagrams...")
md, me, phi_m2 = generate_phase_data(mode="m2")
_, _, phi_q = generate_phase_data(mode="q")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

im1 = ax1.imshow(phi_m2, origin='lower', extent=[md[0], md[-1], me[0], me[-1]], cmap='RdBu_r', aspect='auto')
ax1.set_title(r"Mode $M = \mu_d m + \mu_e m^2$ (Theory Correct)")

im2 = ax2.imshow(phi_q, origin='lower', extent=[md[0], md[-1], me[0], me[-1]], cmap='RdBu_r', aspect='auto')
ax2.set_title(r"Mode $M = \mu_d m + \mu_e q$ (Biased)")

for ax in [ax1, ax2]:
    ax.set_xlabel(r"Linear Coupling $\mu_d$")
    ax.set_ylabel(r"Higher-order Coupling $\mu_e$")

plt.colorbar(im1, ax=ax1, label=r"Tipping Rate $\phi$")
plt.colorbar(im2, ax=ax2, label=r"Tipping Rate $\phi$")
plt.tight_layout()
plt.show()
