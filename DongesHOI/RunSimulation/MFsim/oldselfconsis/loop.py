import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# --- PNAS 风格配置 ---
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "axes.labelsize": 9,
    "font.size": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.3,
    "figure.dpi": 300
})


# --- 核心物理逻辑 ---
def solve_cubic_branch(h, current_m):
    coeffs = [1, 0, -1, -h]
    roots = np.roots(coeffs)
    real_roots = roots[np.isclose(roots.imag, 0)].real
    return real_roots[np.argmin(np.abs(real_roots - current_m))]


def self_consistent_step(m_init, H_ext, params, tol=1e-5):
    m = m_init
    q = m ** 2
    mu_d, mu_e = params['mu_d'], params['mu_e']
    sig_d, sig_e, sig_u = params['sig_d'], params['sig_e'], params['sig_u']

    for _ in range(50):
        # 1. 计算当前场强和总噪声
        mu_h = H_ext + mu_d * m +  mu_e * (m ** 2)
        var_h = sig_u ** 2 + (sig_d ** 2) * q + (sig_e ** 2) * (q ** 2)
        sig_h = np.sqrt(max(var_h, 1e-9))

        # 2. 核心优化：标准化积分 (Standardized Integration)
        # 令 z = (h - mu_h) / sig_h, 则 h = mu_h + z * sig_h
        # 这样积分区间永远在 [-8, 8] 之间，且被积函数不再随 sig_h 剧烈缩放
        def standardized_integrand(z):
            h_val = mu_h + z * sig_h
            # 计算标准正态分布概率密度 phi(z)
            phi_z = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z ** 2)
            return solve_cubic_branch(h_val, m) * phi_z

        # 3. 使用自适应积分，增加 limit 并减小 epsabs
        try:
            # 积分区间 [-8, 8] 覆盖了 99.9999% 的高斯权重
            new_m, _ = quad(standardized_integrand, -8, 8, limit=100, epsabs=1e-5)
        except:
            # 如果积分依然失败，使用确定性近似（即 sigma -> 0 的情况）
            new_m = solve_cubic_branch(mu_h, m)

        if abs(new_m - m) < tol: break
        m = new_m
        q = m ** 2
    return m


def get_hysteresis(param_name, param_range, base_params, H_ext=0.1):
    m_f, m_b = [], []
    # 正向
    curr_m = -0.6
    for v in param_range:
        ps = base_params.copy()
        ps[param_name] = v
        curr_m = self_consistent_step(curr_m, H_ext, ps)
        m_f.append(curr_m)
    # 反向
    curr_m = 0.6
    for v in reversed(param_range):
        ps = base_params.copy()
        ps[param_name] = v
        curr_m = self_consistent_step(curr_m, H_ext, ps)
        m_b.append(curr_m)
    return m_f, list(reversed(m_b))


# --- 绘图程序 ---
fig, axes = plt.subplots(2, 2, figsize=(7, 6))
base = {'mu_d': 0.4, 'mu_e': 0.5, 'sig_d': 0.1, 'sig_e': 0.1, 'sig_u': 0.13}

# 子图 A: mu_d (直接耦合)
ax = axes[0, 0]
r = np.linspace(-0.5, 1.5, 30)
mf, mb = get_hysteresis('mu_d', r, base)
ax.plot(r, mf, 'o-', color='#0072B2', ms=3, label='Forward', mfc='w')
ax.plot(r, mb, 's-', color='#D55E00', ms=3, label='Backward', mfc='w')
ax.set_title(r"A. Direct Coupling $\mu_d$", loc='left', fontweight='bold')

# 子图 B: mu_e (高阶耦合)
ax = axes[0, 1]
r = np.linspace(0.0, 1.5, 30)
mf, mb = get_hysteresis('mu_e', r, base)
ax.plot(r, mf, 'o-', color='#0072B2', ms=3, mfc='w')
ax.plot(r, mb, 's-', color='#D55E00', ms=3, mfc='w')
ax.set_title(r"B. Higher-order $\mu_e$", loc='left', fontweight='bold')

# 子图 C: sigma_u (背景噪声)
ax = axes[1, 0]
r = np.linspace(0.01, 0.8, 30)
mf, mb = get_hysteresis('sig_u', r, base)
ax.plot(r, mf, 'o-', color='#0072B2', ms=3, mfc='w')
ax.plot(r, mb, 's-', color='#D55E00', ms=3, mfc='w')
ax.set_title(r"C. Additive Noise $\sigma_u$", loc='left', fontweight='bold')

# 子图 D: sigma_d & sigma_e (乘性噪声综合影响)
# 这里演示同时改变 sig_d 和 sig_e 的逻辑
ax = axes[1, 1]
r = np.linspace(0.01, 1.5, 30)
m_f_d, m_b_d = [], []
curr_m_f, curr_m_b = -1.0, 1.0
for v in r:
    ps = base.copy();
    ps['sig_d'], ps['sig_e'] = v, v
    curr_m_f = self_consistent_step(curr_m_f, 0.0, ps)
    m_f_d.append(curr_m_f)
for v in reversed(r):
    ps = base.copy();
    ps['sig_d'], ps['sig_e'] = v, v
    curr_m_b = self_consistent_step(curr_m_b, 0.0, ps)
    m_b_d.append(curr_m_b)
ax.plot(r, m_f_d, 'o-', color='#0072B2', ms=3, mfc='w')
ax.plot(r, list(reversed(m_b_d)), 's-', color='#D55E00', ms=3, mfc='w')
ax.set_title(r"D. Multiplicative Noise $\sigma_{d,e}$", loc='left', fontweight='bold')

# 统一修饰
for ax in axes.flat:
    ax.set_xlabel("Parameter Value")
    ax.set_ylabel("Order Parameter $m$")
    ax.axhline(0, color='black', lw=0.5, ls='--')

axes[0, 0].legend(frameon=False)
plt.tight_layout()
plt.show()