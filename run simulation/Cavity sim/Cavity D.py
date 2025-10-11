#!/usr/bin/env python3
# single_compare.py
"""
将理论求解（穴位法 SpecificBistableCavitySolver）与数值仿真（RK4）结合，
并绘制/比较结果。

依赖:
    numpy, scipy, matplotlib, scikit-learn

运行:
    python single_compare.py
"""

import time
import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
import matplotlib
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# ---------------- 理论求解器：SpecificBistableCavitySolver（改进版） ----------------
class SpecificBistableCavitySolver:
    """
    固定均值为 ±1.324718 的双稳态系统“穴位法”求解器（改进）
    参数通过 params 字典传入。
    支持可配置的数值保护 denom_eps 与松弛因子 relaxation。
    """

    def __init__(self, params):
        # 模型参数（必需）
        self.μ_c = params['μ_c']
        self.μ_d = params['μ_d']
        self.μ_e = params['μ_e']
        self.σ_c = params['σ_c']
        self.σ_d = params['σ_d']
        self.σ_e = params['σ_e']
        self.ρ_d = params.get('ρ_d', 0.0)
        self.S = params.get('S', np.inf)

        # 固定双稳态峰位置
        self.μ1 = -1.324718
        self.μ2 =  1.324718
        self.μ_sq = 1.754876  # μ1^2 == μ2^2 预计算值

        # 收敛控制参数
        self.max_iter = params.get('max_iter', 1000)
        self.tol = params.get('tol', 1e-6)
        self.relaxation = params.get('relaxation', 0.3)

        # 数值保护
        self.denom_eps = params.get('denom_eps', 1e-8)
        # 初始值通过 initialize_parameters 设置
        self.initialize_parameters()

    def initialize_parameters(self):
        """初始化迭代变量与中间量"""
        self.phi = 0.6
        self.σ1_sq = 0.05
        self.σ2_sq = 0.05
        self.v1 = 1.0
        self.v2 = 1.0

        self.m = (1 - self.phi) * self.μ1 + self.phi * self.μ2
        self.x2_1 = self.μ_sq + self.σ1_sq
        self.x2_2 = self.μ_sq + self.σ2_sq
        self.x2_avg = (1 - self.phi) * self.x2_1 + self.phi * self.x2_2
        self.σ_sq = self.x2_avg - self.m**2

        weighted_x2 = (
            (1 - self.phi)**2 * self.x2_1**2 +
            2 * self.phi * (1 - self.phi) * self.x2_1 * self.x2_2 +
            self.phi**2 * self.x2_2**2
        )
        self.F = self.ρ_d * self.σ_d**2 + 2 * self.σ_e**2 * weighted_x2

        self.D1 = self.σ_c**2 + self.σ_d**2 * self.x2_1 + self.σ_e**2 * self.x2_1**2
        self.D2 = self.σ_c**2 + self.σ_d**2 * self.x2_2 + self.σ_e**2 * self.x2_2**2

        self.history = {'phi': [], 'σ1_sq': [], 'σ2_sq': [], 'v1': [], 'v2': [], 'm': []}

    def calculate_overall_moments(self):
        """更新总体均值与方差等矩"""
        self.m = (1 - self.phi) * self.μ1 + self.phi * self.μ2
        self.x2_1 = self.μ_sq + self.σ1_sq
        self.x2_2 = self.μ_sq + self.σ2_sq
        self.x2_avg = (1 - self.phi) * self.x2_1 + self.phi * self.x2_2
        self.σ_sq = self.x2_avg - self.m**2

    def calculate_feedback_term(self):
        """计算反馈项 F（依赖 x2_1, x2_2, phi）"""
        weighted_x2 = (
            (1 - self.phi)**2 * self.x2_1**2 +
            2 * self.phi * (1 - self.phi) * self.x2_1 * self.x2_2 +
            self.phi**2 * self.x2_2**2
        )
        self.F = self.ρ_d * self.σ_d**2 + 2 * self.σ_e**2 * weighted_x2

    def calculate_noise_strengths(self):
        """计算 D1, D2"""
        self.D1 = self.σ_c**2 + self.σ_d**2 * self.x2_1 + self.σ_e**2 * self.x2_1**2
        self.D2 = self.σ_c**2 + self.σ_d**2 * self.x2_2 + self.σ_e**2 * self.x2_2**2

    def calculate_effective_drive(self):
        """计算有效驱动 μ_eff"""
        self.μ_eff = self.μ_c + self.μ_d * self.m + self.μ_e * self.m**2

    def update_states(self):
        """
        单步更新 σ1_sq, σ2_sq, phi, v1, v2（未应用松弛）
        包含 denom 下限保护，phi 使用 erf 计算
        """
        denom1 = 1.0 - self.v1 * self.F
        denom2 = 1.0 - self.v2 * self.F

        # 下限保护（保留符号以避免不必要的符号翻转）
        denom1 = np.sign(denom1) * max(abs(denom1), self.denom_eps)
        denom2 = np.sign(denom2) * max(abs(denom2), self.denom_eps)

        σ1_sq_new = self.D1 / (2.0 * denom1**2)
        σ2_sq_new = self.D2 / (2.0 * denom2**2)

        # 计算 z 时保护标准差分母
        denom_std = np.sqrt(max(1e-18, 2.0 * self.D2 / (denom2**2)))
        z = (self.μ_eff / denom2) / denom_std
        phi_new = 0.5 * (1.0 + erf(z))

        v1_new = 1.0 / denom1
        v2_new = 1.0 / denom2

        return σ1_sq_new, σ2_sq_new, phi_new, v1_new, v2_new

    def solve_iterative(self, verbose=True):
        """迭代求解自洽方程组，返回结果字典"""
        self.initialize_parameters()
        for i in range(self.max_iter):
            # 记录历史
            self.history['phi'].append(self.phi)
            self.history['σ1_sq'].append(self.σ1_sq)
            self.history['σ2_sq'].append(self.σ2_sq)
            self.history['v1'].append(self.v1)
            self.history['v2'].append(self.v2)
            self.history['m'].append(self.m)

            # 更新中间量
            self.calculate_overall_moments()
            self.calculate_feedback_term()
            self.calculate_noise_strengths()
            self.calculate_effective_drive()

            σ1_sq_new, σ2_sq_new, phi_new, v1_new, v2_new = self.update_states()

            # 松弛更新
            self.σ1_sq = self.relaxation * σ1_sq_new + (1.0 - self.relaxation) * self.σ1_sq
            self.σ2_sq = self.relaxation * σ2_sq_new + (1.0 - self.relaxation) * self.σ2_sq
            self.phi   = self.relaxation * phi_new   + (1.0 - self.relaxation) * self.phi
            self.v1    = self.relaxation * v1_new    + (1.0 - self.relaxation) * self.v1
            self.v2    = self.relaxation * v2_new    + (1.0 - self.relaxation) * self.v2

            # 数值保护：方差非负、phi 限制在 [0,1]
            self.σ1_sq = max(self.σ1_sq, 1e-12)
            self.σ2_sq = max(self.σ2_sq, 1e-12)
            self.phi = np.clip(self.phi, 0.0, 1.0)

            if verbose and (i % 50 == 0):
                print(f"[Theory] iter {i:4d}: phi={self.phi:.6f}, σ1²={self.σ1_sq:.3e}, σ2²={self.σ2_sq:.3e}")

            if self.check_convergence():
                if verbose:
                    print(f"[Theory] Converged at iter {i}")
                break
        else:
            if verbose:
                print("[Theory] Warning: reached max_iter without convergence")

        self.calculate_overall_moments()
        return self.get_results()

    def check_convergence(self):
        """判断收敛：phi, σ1_sq, σ2_sq 的变化均小于 tol"""
        if len(self.history['phi']) < 2:
            return False
        dphi = abs(self.history['phi'][-1] - self.history['phi'][-2])
        dσ1 = abs(self.history['σ1_sq'][-1] - self.history['σ1_sq'][-2])
        dσ2 = abs(self.history['σ2_sq'][-1] - self.history['σ2_sq'][-2])
        return (dphi < self.tol and dσ1 < self.tol and dσ2 < self.tol)

    def get_results(self):
        """返回结果字典"""
        return {
            'm': self.m,
            'σ_sq': self.σ_sq,
            'phi': self.phi,
            'μ1': self.μ1,
            'μ2': self.μ2,
            'σ1_sq': self.σ1_sq,
            'σ2_sq': self.σ2_sq,
            'v1': self.v1,
            'v2': self.v2,
            'F': self.F,
            'D1': self.D1,
            'D2': self.D2,
            'history': self.history
        }

    # 可视化函数
    def plot_convergence(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes[0,0].plot(self.history['phi']); axes[0,0].set_title('phi 迭代')
        axes[0,1].plot(self.history['σ1_sq'], label='σ1²'); axes[0,1].plot(self.history['σ2_sq'], label='σ2²'); axes[0,1].legend(); axes[0,1].set_title('方差迭代')
        axes[1,0].plot(self.history['v1'], label='v1'); axes[1,0].plot(self.history['v2'], label='v2'); axes[1,0].legend(); axes[1,0].set_title('响应参数')
        axes[1,1].plot(self.history['m']); axes[1,1].set_title('总体均值 m')
        for ax in axes.flatten():
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_bimodal_distribution(self, x_range=(-3,3), n_points=1000):
        x = np.linspace(x_range[0], x_range[1], n_points)
        g1 = (1.0 - self.phi) * np.exp(-(x - self.μ1)**2 / (2.0 * self.σ1_sq)) / np.sqrt(2*np.pi*self.σ1_sq)
        g2 = self.phi * np.exp(-(x - self.μ2)**2 / (2.0 * self.σ2_sq)) / np.sqrt(2*np.pi*self.σ2_sq)
        total = g1 + g2
        plt.figure(figsize=(9,5))
        plt.plot(x, g1, 'b--', label=f'状态1 μ1={self.μ1}, σ1²={self.σ1_sq:.4f}')
        plt.plot(x, g2, 'r--', label=f'状态2 μ2={self.μ2}, σ2²={self.σ2_sq:.4f}')
        plt.plot(x, total, 'k-', lw=2, label=f'总体 φ={self.phi:.3f}')
        plt.axvline(self.μ1, color='b', linestyle=':'); plt.axvline(self.μ2, color='r', linestyle=':')
        plt.xlabel('x'); plt.ylabel('pdf'); plt.legend(); plt.grid(True, alpha=0.3)
        plt.title('理论双峰分布（穴位法）')
        plt.show()

# ---------------- 数值仿真（第一个程序中的 RK4 部分） ----------------
def generate_parameters_numeric(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, rng=None):
    """
    生成数值仿真所需的随机矩阵/向量。
    返回 c_i, d_ji, E （d_ji 为传入 dynamics 的矩阵）
    """
    if rng is None:
        rng = np.random.default_rng()
    c_i = rng.normal(loc=mu_c, scale=sigma_c, size=s)
    D = rng.normal(loc=mu_d, scale=sigma_d, size=(s, s))
    np.fill_diagonal(D, 0.0)
    d_ji = D.T.copy()
    E = rng.normal(loc=mu_e, scale=sigma_e, size=(s, s))
    E = (E + E.T) / 2.0
    return c_i, d_ji, E

def dynamics_derivative(x, c_i, d_ji, E):
    """示例动力学导数：dx/dt = -x + c + (d_ji.T @ x)/s + E.T @ x"""
    s = x.size
    linear_term = (d_ji.T @ x) / float(s)
    quad_term = E.T @ x
    dx = -x + c_i + linear_term + quad_term
    return dx

def rk4_step(x, dt, c_i, d_ji, E):
    k1 = dynamics_derivative(x, c_i, d_ji, E)
    k2 = dynamics_derivative(x + 0.5*dt*k1, c_i, d_ji, E)
    k3 = dynamics_derivative(x + 0.5*dt*k2, c_i, d_ji, E)
    k4 = dynamics_derivative(x + dt*k3, c_i, d_ji, E)
    return x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

def dynamics_simulation(s, c_i, d_ji, E, x_init, t_steps, dt=0.01, record_every=10):
    x = x_init.copy().astype(float)
    traj = []
    for t in range(t_steps):
        x = rk4_step(x, dt, c_i, d_ji, E)
        if (t % record_every) == 0:
            traj.append(x.copy())
    traj = np.array(traj)
    return x, traj

def estimate_phi_threshold(x_final, threshold=0.0):
    """简单阈值估计 phi（右峰占比）"""
    right = np.sum(x_final > threshold)
    return float(right) / x_final.size

def estimate_phi_gmm(x_final):
    """使用 GMM（2 成分）来估计两个成分的权重并返回靠近 μ2 (正峰) 的权重"""
    samples = np.asarray(x_final).reshape(-1,1)
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
    gmm.fit(samples)
    means = gmm.means_.flatten()
    weights = gmm.weights_.flatten()
    # 找到哪个分量 mean 更靠近正峰 μ2
    idx_pos = np.argmax(means)
    return float(weights[idx_pos]), gmm

def plot_final_state_distribution(x_final, bins=100, x_range=(-3,3)):
    plt.figure(figsize=(8,5))
    plt.hist(x_final, bins=bins, range=x_range, density=True, alpha=0.6, color='C0', label='数值直方图')
    samples = x_final.reshape(-1,1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.12).fit(samples)
    xs = np.linspace(x_range[0], x_range[1], 500).reshape(-1,1)
    dens = np.exp(kde.score_samples(xs))
    plt.plot(xs[:,0], dens, 'k-', lw=2, label='KDE')
    plt.xlabel('x'); plt.ylabel('pdf'); plt.title('数值仿真最终分布'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.show()

def plot_theory_vs_numeric(solver, x_samples, method='kde', x_range=(-3,3)):
    xs = np.linspace(x_range[0], x_range[1], 500)
    g1 = (1.0 - solver.phi) * np.exp(-(xs - solver.μ1)**2 / (2.0*solver.σ1_sq)) / np.sqrt(2*np.pi*solver.σ1_sq)
    g2 = solver.phi * np.exp(-(xs - solver.μ2)**2 / (2.0*solver.σ2_sq)) / np.sqrt(2*np.pi*solver.σ2_sq)
    theory = g1 + g2

    plt.figure(figsize=(8,5))
    plt.plot(xs, theory, 'r-', lw=2, label='理论 双高斯')
    samples = np.asarray(x_samples).reshape(-1,1)
    if method == 'kde':
        kde = KernelDensity(kernel='gaussian', bandwidth=0.12).fit(samples)
        dens = np.exp(kde.score_samples(xs.reshape(-1,1)))
        plt.plot(xs, dens, 'k--', lw=2, label='数值 KDE')
    else:
        gmm = fit_gmm(samples, n_components=2)
        dens = np.exp(gmm.score_samples(xs.reshape(-1,1)))
        plt.plot(xs, dens, 'k--', lw=2, label='数值 GMM')
    plt.axvline(solver.μ1, color='b', linestyle=':', label='理论 μ1')
    plt.axvline(solver.μ2, color='r', linestyle=':', label='理论 μ2')
    plt.xlabel('x'); plt.ylabel('pdf'); plt.legend(); plt.grid(True, alpha=0.3); plt.title('理论 vs 数值')
    plt.show()

def fit_gmm(samples, n_components=2):
    samples = np.asarray(samples).reshape(-1,1)
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
    gmm.fit(samples)
    return gmm

# ---------------- 主流程 ----------------
def run_all_and_compare(
    # 理论参数
    mu_c=0.0, mu_d=0.2, mu_e=0.2, sigma_c=0.4, sigma_d=0.3, sigma_e=0.1, rho_d=1.0,
    # 数值仿真参数
    s=2000, t_steps=3000, dt=0.01, x0_val=0.6,
    # 其他
    use_gmm_for_estimate=False, quick_mode=False
):
    """
    运行理论求解与数值仿真并比较。
    quick_mode=True 时使用较小规模（便于调试）。
    """
    if quick_mode:
        s = min(s, 500)
        t_steps = min(t_steps, 1000)
        print("[Mode] quick_mode ON: reduced s and t_steps for faster run")

    params = {
        'μ_c': mu_c, 'μ_d': mu_d, 'μ_e': mu_e,
        'σ_c': sigma_c, 'σ_d': sigma_d, 'σ_e': sigma_e,
        'ρ_d': rho_d, 'S': s,
        'max_iter': 2000, 'tol': 1e-6, 'relaxation': 0.3, 'denom_eps': 1e-8
    }

    # 1) 理论求解
    solver = SpecificBistableCavitySolver(params)
    t0 = time.time()
    res = solver.solve_iterative(verbose=True)
    t1 = time.time()
    print(f"[Theory] solved in {t1-t0:.2f} s")
    print(f"[Theory] phi={res['phi']:.6f}, σ1²={res['σ1_sq']:.6e}, σ2²={res['σ2_sq']:.6e}")

    solver.plot_convergence()
    solver.plot_bimodal_distribution()

    # 2) 数值仿真生成参数并运行 RK4
    rng = np.random.default_rng(42)
    c_i, d_ji, E = generate_parameters_numeric(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, rng=rng)
    x_init = np.full(s, x0_val, dtype=float)

    print("[Numeric] start RK4 simulation...")
    t0 = time.time()
    x_final, traj = dynamics_simulation(s, c_i, d_ji, E, x_init, t_steps, dt=dt, record_every=10)
    t1 = time.time()
    print(f"[Numeric] done in {t1-t0:.2f} s")

    # 3) 后处理与比较
    plot_final_state_distribution(x_final)
    if use_gmm_for_estimate:
        w_pos, gmm = estimate_phi_gmm(x_final)
        print(f"[Numeric] GMM estimated weight for positive-mean component = {w_pos:.4f}")
    else:
        phi_num = estimate_phi_threshold(x_final, threshold=0.0)
        print(f"[Numeric] threshold estimated phi = {phi_num:.4f}")

    plot_theory_vs_numeric(solver, x_final, method='kde')
    # 两栏对比
    xs = np.linspace(-3,3,500)
    g1 = (1.0 - solver.phi) * np.exp(-(xs - solver.μ1)**2 / (2.0*solver.σ1_sq)) / np.sqrt(2*np.pi*solver.σ1_sq)
    g2 = solver.phi * np.exp(-(xs - solver.μ2)**2 / (2.0*solver.σ2_sq)) / np.sqrt(2*np.pi*solver.σ2_sq)
    theory = g1 + g2

    fig, axes = plt.subplots(1,2, figsize=(14,5))
    axes[0].hist(x_final, bins=100, range=(-3,3), density=True, alpha=0.6, color='C0')
    axes[0].plot(xs, theory, 'r-', lw=2); axes[0].set_title('直方图与理论'); axes[0].grid(True, alpha=0.3)
    samples = x_final.reshape(-1,1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.12).fit(samples)
    dens = np.exp(kde.score_samples(xs.reshape(-1,1)))
    axes[1].plot(xs, dens, 'k--', lw=2); axes[1].plot(xs, theory, 'r-', lw=2); axes[1].set_title('KDE 与理论'); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return solver, x_final, traj

# ---------------- 如果直接运行脚本 ----------------
if __name__ == "__main__":
    # 你可以在这里调整参数或开启 quick_mode 以加快测试
    solver, x_final, traj = run_all_and_compare(
        mu_c=0.0, mu_d=0.2, mu_e=0.2,
        sigma_c=0.4, sigma_d=0.3, sigma_e=0.1, rho_d=1.0,
        s=2000, t_steps=3000, dt=0.01, x0_val=0.6,
        use_gmm_for_estimate=False,
        quick_mode=False  # 调试时可设为 True
    )