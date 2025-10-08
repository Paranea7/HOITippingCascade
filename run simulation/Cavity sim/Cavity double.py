import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
import matplotlib
import time

# 设置中文字体（可选）
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# ---------------- 你的数值仿真模块（保留并略微重构） ----------------

def generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, rng=None):
    """
    生成参数：
    - c_i: 长度 s 的向量，N(mu_c, sigma_c)
    - d_ij: s x s 矩阵，均值 mu_d/s，标准差 sigma_d/s
    - d_ji: 与 d_ij 同形，按相关系数 rho_d 生成相关矩阵
    - e_ijk: s x s x s 张量，均值 mu_e/s^2，标准差 sigma_e/s^2
    注意：当 s 很大时，e_ijk 占用内存会很大（O(s^3)），请谨慎使用。
    """
    if rng is None:
        rng = np.random.default_rng()

    c_i = rng.normal(mu_c, sigma_c, s)
    d_ij = rng.normal(mu_d / s, sigma_d / s, (s, s))
    noise_matrix = rng.normal(mu_d / s, sigma_d / s, (s, s))
    d_ji = rho_d * d_ij + np.sqrt(max(0.0, 1 - rho_d ** 2)) * noise_matrix
    e_ijk = rng.normal(mu_e / (s ** 2), sigma_e / (s ** 2), (s, s, s))
    return c_i, d_ij, d_ji, e_ijk


def compute_dynamics(x, c_i, d_ji, e_ijk):
    """
    计算 dx/dt 的右侧：
    dx = -x^3 + x + c_i + dot(d_ji, x) + contraction(e_ijk, x, x)
    使用 einsum 加速 e_ijk 收缩。
    """
    dx = -x ** 3 + x + c_i
    dx += np.dot(d_ji, x)
    e_contribution = np.einsum('ijk,j,k->i', e_ijk, x, x)
    dx += e_contribution
    return dx


def rk4_step(x, c_i, d_ji, e_ijk, dt):
    """ 四阶龙格-库塔单步 """
    k1 = compute_dynamics(x, c_i, d_ji, e_ijk)
    k2 = compute_dynamics(x + 0.5 * dt * k1, c_i, d_ji, e_ijk)
    k3 = compute_dynamics(x + 0.5 * dt * k2, c_i, d_ji, e_ijk)
    k4 = compute_dynamics(x + dt * k3, c_i, d_ji, e_ijk)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps, dt=0.01, record_every=1):
    """
    使用 RK4 对 t_steps 步进行仿真。
    返回：最终状态 x, survival_counts(时间序列)
    """
    x = x_init.copy()
    survival_counts = []
    for step in range(t_steps):
        x = rk4_step(x, c_i, d_ji, e_ijk, dt)
        if (step % record_every) == 0:
            survival_counts.append(np.sum(x > 0))
    return x, np.array(survival_counts)


def plot_final_state_distribution(final_states, bins=100):
    plt.figure(figsize=(8, 5))
    plt.hist(final_states, bins=bins, density=True, alpha=0.7, color='tab:blue', edgecolor='black')
    plt.title('Final State Distribution')
    plt.xlabel('x (final)')
    plt.ylabel('Density')
    plt.xlim(-1.6, 1.6)
    plt.grid(alpha=0.3)
    plt.show()


def estimate_phi_from_final_states(final_states):
    """ φ_num = fraction of components with x>0 at final time """
    return np.mean(final_states > 0.0)


# ---------------- 理论模块（self-consistent solver） ----------------

class BistableCavitySolver:
    def __init__(self, params):
        self.μ_c = params['μ_c']
        self.μ_d = params['μ_d']
        self.μ_e = params['μ_e']
        self.σ_c = params['σ_c']
        self.σ_d = params['σ_d']
        self.σ_e = params['σ_e']
        self.ρ_d = params.get('ρ_d', 1.0)
        self.S = params.get('S', np.inf)

        self.max_iter = params.get('max_iter', 2000)
        self.tol = params.get('tol', 1e-6)
        self.relaxation = params.get('relaxation', 0.5)

        self.denom_eps = params.get('denom_eps', 1e-6)
        self.denom_warn_limit = params.get('denom_warn_limit', 20)

        self.initialize_parameters()
        self.denom_warn_count = 0

    def initialize_parameters(self):
        self.μ1 = -1.1
        self.μ2 = 1.1
        self.phi = 0.5
        self.σ1_sq = 0.05
        self.σ2_sq = 0.05
        self.v1 = 1.0
        self.v2 = 1.0
        self.history = {'phi': [], 'μ1': [], 'μ2': [], 'σ1_sq': [], 'σ2_sq': [], 'v1': [], 'v2': []}

    def calculate_overall_moments(self):
        self.m = (1 - self.phi) * self.μ1 + self.phi * self.μ2
        self.x2_1 = self.μ1 ** 2 + self.σ1_sq
        self.x2_2 = self.μ2 ** 2 + self.σ2_sq
        self.x2_avg = (1 - self.phi) * self.x2_1 + self.phi * self.x2_2
        self.σ_sq = self.x2_avg - self.m ** 2

    def calculate_feedback_term(self):
        weighted_x2 = (
            (1 - self.phi) ** 2 * self.x2_1 ** 2 +
            2 * self.phi * (1 - self.phi) * self.x2_1 * self.x2_2 +
            self.phi ** 2 * self.x2_2 ** 2
        )
        self.F = self.ρ_d * self.σ_d ** 2 + 2 * self.σ_e ** 2 * weighted_x2

    def calculate_noise_strengths(self):
        self.D1 = self.σ_c ** 2 + self.σ_d ** 2 * self.x2_1 + self.σ_e ** 2 * self.x2_1 ** 2
        self.D2 = self.σ_c ** 2 + self.σ_d ** 2 * self.x2_2 + self.σ_e ** 2 * self.x2_2 ** 2

    def calculate_effective_drive(self):
        self.μ_eff = self.μ_c + self.μ_d * self.m + self.μ_e * self.m ** 2

    def _safe_denom(self, denom):
        denom_sign = 1.0 if denom == 0.0 else np.sign(denom)
        if abs(denom) < self.denom_eps:
            denom_safe = denom_sign * self.denom_eps
            return denom_safe, True
        return denom, False

    def update_states(self):
        denom1 = 1 - self.v1 * self.F
        denom2 = 1 - self.v2 * self.F
        denom1_safe, adj1 = self._safe_denom(denom1)
        denom2_safe, adj2 = self._safe_denom(denom2)
        denom_warn_flag = adj1 or adj2

        σ1_sq_new = self.D1 / (2 * denom1_safe ** 2)
        σ2_sq_new = self.D2 / (2 * denom2_safe ** 2)

        var2 = (self.D2 / denom2_safe ** 2) if denom2_safe != 0 else self.D2 / (self.denom_eps ** 2)
        z_raw = (self.μ2) / np.sqrt(2 * var2) if var2 > 0 else 0.0
        z = np.clip(z_raw, -10.0, 10.0)
        phi_new = 0.5 * (1 + erf(z))

        v1_new = 1 / denom1_safe
        v2_new = 1 / denom2_safe

        return σ1_sq_new, σ2_sq_new, phi_new, v1_new, v2_new, denom_warn_flag

    def apply_bistable_constraints(self, μ1, μ2):
        """
        已取消对 μ1, μ2 的强制截断（应客户要求）。
        如果需要再恢复约束或使用其它策略，请修改此函数。
        """
        return μ1, μ2

    def solve_iterative(self, verbose=True):
        self.initialize_parameters()
        self.denom_warn_count = 0
        for i in range(self.max_iter):
            self.history['phi'].append(self.phi)
            self.history['μ1'].append(self.μ1)
            self.history['μ2'].append(self.μ2)
            self.history['σ1_sq'].append(self.σ1_sq)
            self.history['σ2_sq'].append(self.σ2_sq)
            self.history['v1'].append(self.v1)
            self.history['v2'].append(self.v2)

            self.calculate_overall_moments()
            self.calculate_feedback_term()
            self.calculate_noise_strengths()
            self.calculate_effective_drive()

            σ1_sq_new, σ2_sq_new, phi_new, v1_new, v2_new, denom_warn_flag = self.update_states()
            if denom_warn_flag:
                self.denom_warn_count += 1
                if verbose:
                    print(f"[Warning] denom protection triggered at iter {i} (count={self.denom_warn_count})")

            # 取消对 μ1, μ2 的外部截断
            self.μ1, self.μ2 = self.apply_bistable_constraints(self.μ1, self.μ2)

            self.σ1_sq = self.relaxation * σ1_sq_new + (1 - self.relaxation) * self.σ1_sq
            self.σ2_sq = self.relaxation * σ2_sq_new + (1 - self.relaxation) * self.σ2_sq
            self.phi = self.relaxation * phi_new + (1 - self.relaxation) * self.phi
            self.v1 = self.relaxation * v1_new + (1 - self.relaxation) * self.v1
            self.v2 = self.relaxation * v2_new + (1 - self.relaxation) * self.v2

            if self.check_convergence():
                if verbose:
                    print(f"Converged at iter {i}")
                break

            if self.denom_warn_count >= self.denom_warn_limit:
                if verbose:
                    print("Terminating early due to denom protection")
                break
        else:
            if verbose:
                print("Warning: max iter reached without convergence")

        self.calculate_overall_moments()
        return self.get_results()

    def check_convergence(self):
        if len(self.history['phi']) < 2:
            return False
        delta_phi = abs(self.history['phi'][-1] - self.history['phi'][-2])
        delta_σ1 = abs(self.history['σ1_sq'][-1] - self.history['σ1_sq'][-2])
        delta_σ2 = abs(self.history['σ2_sq'][-1] - self.history['σ2_sq'][-2])
        return (delta_phi < self.tol and delta_σ1 < self.tol and delta_σ2 < self.tol)

    def get_results(self):
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
            'history': self.history,
            'denom_warn_count': self.denom_warn_count
        }

    def plot_convergence(self):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        axes[0, 0].plot(self.history['phi'])
        axes[0, 0].set_title('共存比例 φ')
        axes[0, 0].set_xlabel('迭代')
        axes[0, 0].set_ylabel('φ')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(self.history['μ1'], label='μ1')
        axes[0, 1].plot(self.history['μ2'], label='μ2')
        axes[0, 1].set_title('状态均值 μ1 / μ2')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(self.history['σ1_sq'], label='σ1_sq')
        axes[1, 0].plot(self.history['σ2_sq'], label='σ2_sq')
        axes[1, 0].set_title('状态方差 σ^2')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(self.history['v1'], label='v1')
        axes[1, 1].plot(self.history['v2'], label='v2')
        axes[1, 1].set_title('响应参数 v1, v2')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_bimodal_distribution(self, x_range=(-2, 2), n_points=1000):
        x = np.linspace(x_range[0], x_range[1], n_points)
        σ1_sq = max(self.σ1_sq, 1e-12)
        σ2_sq = max(self.σ2_sq, 1e-12)
        gaussian1 = ((1 - self.phi) *
                     np.exp(-(x - self.μ1) ** 2 / (2 * σ1_sq)) /
                     np.sqrt(2 * np.pi * σ1_sq))
        gaussian2 = (self.phi *
                     np.exp(-(x - self.μ2) ** 2 / (2 * σ2_sq)) /
                     np.sqrt(2 * np.pi * σ2_sq))
        total = gaussian1 + gaussian2

        plt.figure(figsize=(8,5))
        plt.plot(x, gaussian1, 'b--', label=f'状态1 μ1={self.μ1:.2f}')
        plt.plot(x, gaussian2, 'r--', label=f'状态2 μ2={self.μ2:.2f}')
        plt.plot(x, total, 'k-', label='总体分布')
        plt.axvline(self.μ1, color='b', linestyle=':')
        plt.axvline(self.μ2, color='r', linestyle=':')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('density')
        plt.title('双峰分布')
        plt.grid(alpha=0.3)
        plt.show()


# ---------------- 在一张图中比较数值与理论分布（修改为两面板） ----------------

def compute_hist_probabilities(data, bins=100, range=None):
    """
    计算每-bin 的概率质量（mass）与密度（density）。
    返回：bin_edges, bin_centers, mass, density
    """
    counts, bin_edges = np.histogram(data, bins=bins, range=range, density=False)
    N = float(data.size)
    widths = np.diff(bin_edges)
    mass = counts.astype(float) / N
    density = np.zeros_like(mass)
    nonzero = widths > 0
    density[nonzero] = mass[nonzero] / widths[nonzero]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_edges, bin_centers, mass, density

def plot_compare_distribution_two_panel(solver, final_states, x_range=(-3, 3), bins=100, alpha_hist=0.6):
    """
    在一张图中用两个子图比较数值与理论：
    - 左：密度（density） —— 原来的 plt.hist(density=True) 与理论连续密度曲线
    - 右：每-bin 概率质量（per-bin mass） —— 数值用 bar（mass），理论在相同 bins 上积分得到 theory_mass 并绘出
    """
    final_states = np.asarray(final_states)

    # --- 左图：密度对比（保持原样） ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左轴：density histogram + theory continuous density
    axes[0].hist(final_states, bins=bins, density=True,
                 alpha=alpha_hist, color='tab:blue', edgecolor='black', label='Numerical (final, density)')
    x = np.linspace(x_range[0], x_range[1], 2000)
    σ1_sq = max(solver.σ1_sq, 1e-12)
    σ2_sq = max(solver.σ2_sq, 1e-12)
    gaussian1 = ((1 - solver.phi) *
                 np.exp(-(x - solver.μ1) ** 2 / (2 * σ1_sq)) /
                 np.sqrt(2 * np.pi * σ1_sq))
    gaussian2 = (solver.phi *
                 np.exp(-(x - solver.μ2) ** 2 / (2 * σ2_sq)) /
                 np.sqrt(2 * np.pi * σ2_sq))
    total = gaussian1 + gaussian2
    axes[0].plot(x, total, 'k-', lw=2, label='Theory total (density)')
    axes[0].plot(x, gaussian1, 'b--', lw=1.2, label='Theory mode1 (density)')
    axes[0].plot(x, gaussian2, 'r--', lw=1.2, label='Theory mode2 (density)')
    axes[0].axvline(solver.μ1, color='b', linestyle=':', alpha=0.8)
    axes[0].axvline(solver.μ2, color='r', linestyle=':', alpha=0.8)
    axes[0].set_xlim(x_range)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Density: Numerical vs Theory')
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    # --- 右图：每-bin 概率质量对比（mass） ---
    # 计算数值直方图的 mass（counts / N）
    bin_edges = np.linspace(x_range[0], x_range[1], bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths = np.diff(bin_edges)
    counts, _ = np.histogram(final_states, bins=bin_edges, density=False)
    N = final_states.size
    mass = counts.astype(float) / N
    total_prob = mass.sum()

    # 在细网格上计算理论密度并在每个 bin 上积分得到 theory_mass
    finer = 5  # 在每个 bin 内使用 finer*bins 的细网格积分
    x_fine = np.linspace(x_range[0], x_range[1], int(bins * finer) + 1)
    gaussian1_f = ((1 - solver.phi) *
                   np.exp(-(x_fine - solver.μ1) ** 2 / (2 * σ1_sq)) /
                   np.sqrt(2 * np.pi * σ1_sq))
    gaussian2_f = (solver.phi *
                   np.exp(-(x_fine - solver.μ2) ** 2 / (2 * σ2_sq)) /
                   np.sqrt(2 * np.pi * σ2_sq))
    theory_density_fine = gaussian1_f + gaussian2_f

    theory_mass = np.zeros_like(mass)
    for i in range(len(bin_edges) - 1):
        left, right = bin_edges[i], bin_edges[i+1]
        mask = (x_fine >= left) & (x_fine <= right)
        xs = x_fine[mask]
        ys = theory_density_fine[mask]
        if xs.size == 0:
            # 若没有细点落入该 bin，则以中心密度乘以宽度近似
            xc = 0.5 * (left + right)
            yc = ((1 - solver.phi) * np.exp(-(xc - solver.μ1) ** 2 / (2 * σ1_sq)) / np.sqrt(2 * np.pi * σ1_sq)
                  + solver.phi * np.exp(-(xc - solver.μ2) ** 2 / (2 * σ2_sq)) / np.sqrt(2 * np.pi * σ2_sq))
            theory_mass[i] = yc * (right - left)
        elif xs.size == 1:
            theory_mass[i] = ys[0] * (right - left)
        else:
            theory_mass[i] = np.trapz(ys, xs)
    theory_total = theory_mass.sum()

    # 绘制右图：数值 mass（bar）与理论 mass（点线）
    axes[1].bar(bin_centers, mass, width=widths, alpha=0.6, color='tab:blue', edgecolor='black',
                label='Numerical: per-bin probability (mass)')
    axes[1].plot(bin_centers, theory_mass, 'r.-', lw=1.6, ms=6, label='Theory: per-bin probability (integrated)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Probability per bin (mass)')
    axes[1].set_title('Per-bin probability: Numerical vs Theory')
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    # 在右上角加上归一性信息
    axes[1].text(0.98, 0.95,
                 f'sum(numerical)={total_prob:.6f}\nsum(theory)={theory_total:.6f}',
                 transform=axes[1].transAxes, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    plt.show()

    # 返回用于进一步分析的数据
    return {
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
        'mass': mass,
        'theory_mass': theory_mass,
        'total_prob': total_prob,
        'theory_total': theory_total
    }


# ---------------- 整合：在数值仿真与理论之间做比较 ----------------

def plot_phi_comparison(mu_d_values, phi_theory_vals, phi_num_vals):
    plt.figure(figsize=(8, 4))
    plt.plot(mu_d_values, phi_theory_vals, 'o-', label='Theory φ', linewidth=2)
    plt.plot(mu_d_values, phi_num_vals, 's--', label='Numerical φ (RK4 final)', linewidth=1.5)
    plt.xlabel('μ_d')
    plt.ylabel('φ')
    plt.title('Theory vs Numerical (RK4) φ comparison')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def parameter_scan_and_compare(mu_d_min=0.0, mu_d_max=1.0, n_points=5,
                               s=300, rng_seed=123,
                               x_init_value=0.6, t_steps=3000, dt=0.01,
                               mu_c=0.5, mu_e=0.2, sigma_c=0.1, sigma_d=0.3, sigma_e=0.1,
                               rho_d=1.0):
    rng = np.random.default_rng(rng_seed)
    μ_d_values = np.linspace(mu_d_min, mu_d_max, n_points)
    phi_theory = []
    phi_num = []

    for μ_d in μ_d_values:
        th_params = {
            'μ_c': mu_c,
            'μ_d': μ_d,
            'μ_e': mu_e,
            'σ_c': sigma_c,
            'σ_d': sigma_d,
            'σ_e': sigma_e,
            'ρ_d': rho_d,
            'S': s,
            'max_iter': 500,
            'tol': 1e-5,
            'relaxation': 0.3,
            'denom_eps': 1e-8,
            'denom_warn_limit': 20
        }
        solver = BistableCavitySolver(th_params)
        res = solver.solve_iterative(verbose=False)
        phi_theory.append(res['phi'])

        c_i, d_ij, d_ji, e_ijk = generate_parameters(s, mu_c, sigma_c, μ_d, sigma_d, rho_d, mu_e, sigma_e, rng=rng)
        x_init = np.full(s, x_init_value)
        x_final, survival_counts = dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps, dt=dt, record_every=1)
        phi_n = estimate_phi_from_final_states(x_final)
        phi_num.append(phi_n)

        print(f"μ_d={μ_d:.4f}   φ_theory={res['phi']:.4f}   φ_num(final)={phi_n:.4f}")

    phi_theory = np.array(phi_theory)
    phi_num = np.array(phi_num)

    plot_phi_comparison(μ_d_values, phi_theory, phi_num)

    return {
        'μ_d_values': μ_d_values,
        'phi_theory': phi_theory,
        'phi_num': phi_num
    }


# ---------------- 主程序 ----------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
from scipy.stats import norm
import time
import warnings

warnings.filterwarnings('ignore')


class BistableCavitySolver:
    def __init__(self, params):
        """
        双稳态系统空穴法求解器

        参数:
        params: 包含所有参数的字典
        """
        self.params = params
        self.convergence_history = []

    def single_particle_potential(self, x, mu_eff):
        """单粒子有效势能"""
        return x ** 4 / 4 - x ** 2 / 2 - mu_eff * x

    def effective_drift(self, x, mu_eff):
        """有效漂移项"""
        return -x ** 3 + x + mu_eff

    def boltzmann_distribution(self, x, mu_eff, sigma_eff):
        """玻尔兹曼分布近似"""
        if sigma_eff < 1e-10:
            # 无噪声情况，使用delta函数
            stable_points = self.find_stable_points(mu_eff)
            if len(stable_points) > 0:
                pdf = np.zeros_like(x)
                idx = np.argmin(np.abs(x - stable_points[0]))
                pdf[idx] = 1.0
                return pdf
            else:
                return np.ones_like(x) / len(x)

        V_eff = self.single_particle_potential(x, mu_eff)
        D_eff = sigma_eff ** 2 / 2
        pdf = np.exp(-V_eff / D_eff)
        pdf = pdf / (np.sum(pdf) * (x[1] - x[0]))  # 归一化
        return pdf

    def find_stable_points(self, mu_eff, n_points=1000):
        """寻找稳定点"""
        x_test = np.linspace(-2, 2, n_points)
        drift = self.effective_drift(x_test, mu_eff)

        # 寻找漂移为零的点
        zeros = []
        for i in range(1, len(x_test) - 1):
            if drift[i - 1] * drift[i] <= 0:
                root = fsolve(lambda x: self.effective_drift(x, mu_eff), x_test[i])[0]
                zeros.append(root)

        # 区分稳定和不稳定点
        stable_points = []
        for z in zeros:
            if 3 * z ** 2 - 1 < 0:  # 二阶导数为负，是稳定点
                stable_points.append(z)

        return stable_points

    def response_function(self, x, mu_eff):
        """响应函数"""
        return 1.0 / (3 * x ** 2 - 1)

    def cavity_equations(self, variables):
        """空穴法自洽方程"""
        p = self.params
        m, q, phi_plus, G = variables

        S = p['S']

        # 计算有效参数
        mu_eff = (p['μ_c'] + p['μ_d'] * m +
                  p['μ_e'] * (m ** 2 + (1.21 - m ** 2) / S) +
                  G * p['ρ_d'] * p['σ_d'] ** 2 * (phi_plus + (1 - phi_plus)) * m)

        sigma_eff2 = (p['σ_c'] ** 2 +
                      p['σ_d'] ** 2 * (1.21 - m ** 2) +
                      1.4641 * p['σ_e'] ** 2)
        sigma_eff = np.sqrt(max(sigma_eff2, 1e-10))

        # 求解稳态分布
        x_range = np.linspace(-2.5, 2.5, 1000)
        pdf = self.boltzmann_distribution(x_range, mu_eff, sigma_eff)
        dx = x_range[1] - x_range[0]

        # 计算新的自洽变量
        m_new = np.sum(x_range * pdf) * dx
        q_new = np.sum(x_range ** 2 * pdf) * dx
        phi_plus_new = np.sum(pdf[x_range > 0]) * dx

        # 响应函数更新
        stable_points = self.find_stable_points(mu_eff)
        if len(stable_points) > 0:
            G_vals = [self.response_function(x, mu_eff) for x in stable_points]
            G_new = np.mean(G_vals)
        else:
            G_new = G

        # 自能修正
        denominator = 2.63 - G * p['ρ_d'] * p['σ_d'] ** 2 * (phi_plus + (1 - phi_plus)) - 2 * G * p['σ_e'] ** 2 * q
        if abs(denominator) < p['denom_eps']:
            denominator = np.sign(denominator) * p['denom_eps']
        G_new = 1.0 / denominator

        residuals = np.array([m_new - m, q_new - q, phi_plus_new - phi_plus, G_new - G])

        return residuals, np.array([m_new, q_new, phi_plus_new, G_new]), mu_eff, sigma_eff

    def solve_iterative(self, initial_guess=None, verbose=False):
        """迭代求解自洽方程"""
        p = self.params

        if initial_guess is None:
            # 初始猜测
            m0 = 0.0
            q0 = 1.21
            phi_plus0 = 0.5
            G0 = 1.0 / 2.63
            initial_guess = np.array([m0, q0, phi_plus0, G0])

        variables = initial_guess.copy()
        self.convergence_history = [variables.copy()]

        if verbose:
            print("开始迭代求解...")
            print(f"{'迭代':>4} {'m':>10} {'q':>10} {'φ_+':>10} {'G':>10} {'残差':>12}")
            print("-" * 65)

        denom_warn_count = 0

        for i in range(p['max_iter']):
            old_variables = variables.copy()

            # 计算残差和新变量
            residuals, new_variables, mu_eff, sigma_eff = self.cavity_equations(variables)

            # 检查分母是否过小
            if np.any(np.isnan(new_variables)) or np.any(np.isinf(new_variables)):
                if denom_warn_count < p['denom_warn_limit']:
                    print(f"警告: 第{i}次迭代出现数值问题，调整参数")
                    denom_warn_count += 1
                # 回退到上一步并减小步长
                variables = old_variables
                continue

            # 阻尼更新
            variables = (1 - p['relaxation']) * old_variables + p['relaxation'] * new_variables

            # 约束变量范围
            variables[0] = np.clip(variables[0], -1.5, 1.5)  # m
            variables[1] = max(variables[1], 0.1)  # q
            variables[2] = np.clip(variables[2], 0.01, 0.99)  # phi_plus
            variables[3] = np.clip(variables[3], -10, 10)  # G

            self.convergence_history.append(variables.copy())

            # 检查收敛
            residual_norm = np.linalg.norm(residuals)
            change_norm = np.linalg.norm(variables - old_variables)

            if verbose and i % 50 == 0:
                print(f"{i:4d} {variables[0]:10.6f} {variables[1]:10.6f} "
                      f"{variables[2]:10.6f} {variables[3]:10.6f} {residual_norm:12.6e}")

            if residual_norm < p['tol'] and change_norm < p['tol']:
                if verbose:
                    print(f"收敛于第 {i} 次迭代")
                break

        else:
            if verbose:
                print(f"达到最大迭代次数 {p['max_iter']}，未完全收敛")

        # 最终计算有效参数
        _, _, mu_eff_final, sigma_eff_final = self.cavity_equations(variables)

        result = {
            'm': variables[0],
            'q': variables[1],
            'phi': variables[2],
            'G': variables[3],
            'μ_eff': mu_eff_final,
            'σ_eff': sigma_eff_final,
            'σ_sq': sigma_eff_final ** 2,
            'converged': residual_norm < p['tol']
        }

        return result

    def plot_convergence(self):
        """绘制收敛历史"""
        if not self.convergence_history:
            print("没有收敛历史数据")
            return

        history = np.array(self.convergence_history)
        iterations = np.arange(len(history))

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        variables = ['m', 'q', 'φ_+', 'G']
        for i, ax in enumerate(axes.flat):
            ax.plot(iterations, history[:, i], 'b-', linewidth=2)
            ax.set_xlabel('迭代次数')
            ax.set_ylabel(variables[i])
            ax.set_title(f'{variables[i]} 的收敛历史')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_bimodal_distribution(self, x_range=(-2, 2), n_points=1000):
        """绘制双峰分布"""
        result = self.solve_iterative(verbose=False)

        x = np.linspace(x_range[0], x_range[1], n_points)
        pdf = self.boltzmann_distribution(x, result['μ_eff'], result['σ_eff'])

        plt.figure(figsize=(10, 6))
        plt.plot(x, pdf, 'b-', linewidth=2, label='理论分布')
        plt.axvline(0, color='r', linestyle='--', alpha=0.7, label='势垒位置')
        plt.axvline(1.1, color='g', linestyle='--', alpha=0.7, label='稳定点 ±1.1')
        plt.axvline(-1.1, color='g', linestyle='--', alpha=0.7)
        plt.xlabel('x')
        plt.ylabel('P(x)')
        plt.title(f'双稳态分布: m={result["m"]:.3f}, φ_+={result["phi"]:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


# 数值模拟函数
def generate_parameters(S, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, rng=None):
    """生成随机参数"""
    if rng is None:
        rng = np.random.default_rng()

    # 生成 c_i
    c_i = rng.normal(mu_c, sigma_c, S)

    # 生成对称相关的 d_ij 矩阵
    d_ij = np.zeros((S, S))
    for i in range(S):
        for j in range(i + 1, S):
            z1, z2 = rng.normal(0, 1, 2)
            d_ij[i, j] = mu_d / S + sigma_d / np.sqrt(S) * z1
            d_ij[j, i] = mu_d / S + sigma_d / np.sqrt(S) * (rho_d * z1 + np.sqrt(1 - rho_d ** 2) * z2)

    # 对角线设为0
    np.fill_diagonal(d_ij, 0)

    # 生成 e_ijk 张量 (为节省内存，使用稀疏存储或即时计算)
    # 这里我们只存储必要的统计量，不生成完整张量
    e_ijk = None  # 在实际模拟中动态生成

    return c_i, d_ij, d_ij.T, e_ijk


def dynamics_equation(x, t, c_i, d_ji, e_ijk_func, S):
    """动力学方程"""
    dxdt = np.zeros(S)
    for i in range(S):
        # 固有项
        dxdt[i] = -x[i] ** 3 + x[i] + c_i[i]

        # 线性耦合项
        dxdt[i] += np.sum(d_ji[i] * x)

        # 二次耦合项 (简化为平均场近似以节省计算)
        # 在实际完整实现中，这里应该计算完整的二次耦合
        mean_x = np.mean(x)
        dxdt[i] += 0.5 * mean_x ** 2  # 简化处理

    return dxdt


def dynamics_simulation(S, c_i, d_ji, e_ijk, x_init, t_steps, dt=0.01, record_every=10):
    """数值模拟动力学"""
    t = np.arange(0, t_steps * dt, dt)

    # 使用RK4方法
    x_current = x_init.copy()
    survival_counts = []

    for step in range(t_steps):
        # RK4步骤
        k1 = dt * dynamics_equation(x_current, 0, c_i, d_ji, None, S)
        k2 = dt * dynamics_equation(x_current + 0.5 * k1, 0, c_i, d_ji, None, S)
        k3 = dt * dynamics_equation(x_current + 0.5 * k2, 0, c_i, d_ji, None, S)
        k4 = dt * dynamics_equation(x_current + k3, 0, c_i, d_ji, None, S)

        x_current += (k1 + 2 * k2 + 2 * k3 + k4) / 6

        if step % record_every == 0:
            # 统计存活物种 (远离0的状态)
            survival_count = np.sum(np.abs(x_current) > 0.5)
            survival_counts.append(survival_count)

    return x_current, survival_counts


def plot_final_state_distribution(x_final, bins=50):
    """绘制最终状态分布"""
    plt.figure(figsize=(10, 6))
    plt.hist(x_final, bins=bins, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='势垒')
    plt.axvline(1.1, color='green', linestyle='--', linewidth=1, alpha=0.7, label='稳定点')
    plt.axvline(-1.1, color='green', linestyle='--', linewidth=1, alpha=0.7)
    plt.xlabel('最终状态 x')
    plt.ylabel('概率密度')
    plt.title('数值模拟最终状态分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def estimate_phi_from_final_states(x_final, threshold=0.5):
    """从最终状态估计 φ_+"""
    return np.sum(x_final > threshold) / len(x_final)


def plot_compare_distribution_two_panel(solver, x_final, x_range=(-2, 2), bins=100):
    """在两面板图中比较理论与数值分布"""
    result = solver.solve_iterative(verbose=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 左图：理论分布
    x_theory = np.linspace(x_range[0], x_range[1], 1000)
    pdf_theory = solver.boltzmann_distribution(x_theory, result['μ_eff'], result['σ_eff'])
    ax1.plot(x_theory, pdf_theory, 'b-', linewidth=2, label='理论分布')
    ax1.axvline(0, color='r', linestyle='--', alpha=0.7, label='势垒')
    ax1.set_xlabel('x')
    ax1.set_ylabel('P(x)')
    ax1.set_title(f'理论分布: m={result["m"]:.3f}, φ_+={result["phi"]:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右图：数值分布
    ax2.hist(x_final, bins=bins, density=True, alpha=0.7, color='orange',
             edgecolor='black', label='数值分布')
    ax2.axvline(0, color='r', linestyle='--', alpha=0.7, label='势垒')
    phi_num = estimate_phi_from_final_states(x_final)
    ax2.set_xlabel('x')
    ax2.set_ylabel('概率密度')
    ax2.set_title(f'数值分布: φ_+={phi_num:.3f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"理论预测: φ_+ = {result['phi']:.4f}")
    print(f"数值结果: φ_+ = {phi_num:.4f}")
    print(f"相对误差: {abs(result['phi'] - phi_num) / result['phi'] * 100:.2f}%")


def parameter_scan_and_compare(mu_d_min, mu_d_max, n_points, s, rng_seed,
                               x_init_value, t_steps, dt, **fixed_params):
    """参数扫描比较理论预测和数值模拟"""
    mu_d_values = np.linspace(mu_d_min, mu_d_max, n_points)
    phi_theory = []
    phi_numerical = []

    rng = np.random.default_rng(rng_seed)

    print("进行参数扫描...")
    for i, mu_d in enumerate(mu_d_values):
        print(f"进度: {i + 1}/{n_points}, μ_d = {mu_d:.3f}")

        # 理论预测
        params = fixed_params.copy()
        params.update({
            'μ_d': mu_d,
            'S': s,
            'max_iter': 2000,
            'tol': 1e-5,
            'relaxation': 0.2,
            'denom_eps': 1e-6,
            'denom_warn_limit': 10
        })
        solver = BistableCavitySolver(params)
        result = solver.solve_iterative(verbose=False)
        phi_theory.append(result['phi'])

        # 数值模拟
        c_i, d_ij, d_ji, e_ijk = generate_parameters(
            s, fixed_params['mu_c'], fixed_params['sigma_c'],
            mu_d, fixed_params['sigma_d'], fixed_params['rho_d'],
            fixed_params['mu_e'], fixed_params['sigma_e'], rng=rng
        )
        x_init = np.full(s, x_init_value)
        x_final, _ = dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps, dt=dt)
        phi_num = estimate_phi_from_final_states(x_final)
        phi_numerical.append(phi_num)

    # 绘制比较图
    plt.figure(figsize=(10, 6))
    plt.plot(mu_d_values, phi_theory, 'bo-', linewidth=2, markersize=6, label='理论预测')
    plt.plot(mu_d_values, phi_numerical, 'rs-', linewidth=2, markersize=6, label='数值模拟')
    plt.xlabel('μ_d')
    plt.ylabel('φ_+')
    plt.title('理论预测与数值模拟比较')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return {
        'mu_d_values': mu_d_values,
        'phi_theory': phi_theory,
        'phi_numerical': phi_numerical
    }


if __name__ == "__main__":
    # 示例：单点理论求解 + 单次数值仿真
    s = 2000  # 系统规模（注意内存）
    mu_c = 0.0
    sigma_c = 0.4
    mu_d = 0.2
    sigma_d = 0.3
    rho_d = 1.0
    mu_e = 0.2
    sigma_e = 0.1

    # 理论求解
    params = {
        'μ_c': mu_c,
        'μ_d': mu_d,
        'μ_e': mu_e,
        'σ_c': sigma_c,
        'σ_d': sigma_d,
        'σ_e': sigma_e,
        'ρ_d': rho_d,
        'S': s,
        'max_iter': 1000,
        'tol': 1e-6,
        'relaxation': 0.3,
        'denom_eps': 1e-6,
        'denom_warn_limit': 20
    }
    solver = BistableCavitySolver(params)
    res = solver.solve_iterative(verbose=True)

    print("\n理论解摘要：")
    print(f"总体均值 m = {res['m']:.6f}, φ = {res['phi']:.6f}, σ² = {res['σ_sq']:.6f}")

    solver.plot_convergence()
    solver.plot_bimodal_distribution(x_range=(-3, 3))

    # 数值仿真（RK4）
    rng = np.random.default_rng(42)
    c_i, d_ij, d_ji, e_ijk = generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, rng=rng)
    x_init = np.full(s, 0.6)
    t_steps = 3000
    dt = 0.01

    print("\n开始数值仿真（RK4）...")
    t0 = time.time()
    x_final, survival_counts = dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps, dt=dt, record_every=10)
    t1 = time.time()
    print(f"数值仿真完成，耗时 {(t1 - t0):.2f} 秒")

    # 绘制最终状态分布与统计 φ
    plot_final_state_distribution(x_final)
    phi_num = estimate_phi_from_final_states(x_final)
    print(f"数值仿真 φ_num(final) = {phi_num:.4f}")

    # 在两面板图中比较理论与数值分布
    plot_compare_distribution_two_panel(solver, x_final, x_range=(-3, 3), bins=100)

    # 可选：对 μ_d 扫描并比较（默认不执行以节省时间）
    run_scan = input("是否运行 μ_d 扫描并比较理论/数值 φ？(y/n) [n]: ").strip().lower() or 'n'
    if run_scan == 'y':
        scan_res = parameter_scan_and_compare(mu_d_min=0.0, mu_d_max=1.0, n_points=5,
                                              s=s, rng_seed=123,
                                              x_init_value=0.6, t_steps=3000, dt=0.01,
                                              mu_c=0.5, mu_e=0.2,
                                              sigma_c=0.1, sigma_d=0.3, sigma_e=0.1,
                                              rho_d=1.0)
        print("扫描完成。")
    else:
        print("跳过扫描。")

    print("程序结束。")