import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
import matplotlib

# 设置全局字体为黑体（用于中文显示）
matplotlib.rcParams['font.family'] = 'SimHei'
# 解决负号显示问题（避免中文字体下负号显示为方块）
matplotlib.rcParams['axes.unicode_minus'] = False

class BistableCavitySolver:
    """
    双稳态系统空穴法求解器（μ1, μ2 固定为 -1 和 +1；含 denom 保护）
    """

    def __init__(self, params):
        """
        初始化求解器并读取参数字典
        重要说明：本版本将 μ1 和 μ2 固定为 -1 与 +1（不可被迭代更新）。
        """
        self.μ_c = params['μ_c']
        self.μ_d = params['μ_d']
        self.μ_e = params['μ_e']
        self.σ_c = params['σ_c']
        self.σ_d = params['σ_d']
        self.σ_e = params['σ_e']
        self.ρ_d = params.get('ρ_d', 1.0)
        self.S = params.get('S', np.inf)

        # 迭代控制参数
        self.max_iter = params.get('max_iter', 2000)
        self.tol = params.get('tol', 1e-6)
        self.relaxation = params.get('relaxation', 0.5)

        # denom 保护相关参数
        self.denom_eps = params.get('denom_eps', 1e-6)
        self.denom_warn_limit = params.get('denom_warn_limit', 20)

        # 初始化内部变量与历史记录
        self.initialize_parameters()

        # 用于记录 denom 保护触发次数
        self.denom_warn_count = 0

    def initialize_parameters(self):
        """
        初始化状态参数
        关键修改：将 μ1 和 μ2 固定为 -1 与 +1
        """
        # 强制两个状态均值为常数（不可被迭代更新）
        self.μ1 = -1.0
        self.μ2 = 1.0

        # 初始共存比例
        self.phi = 0.5

        # 方差初值（仍可迭代）
        self.σ1_sq = 0.05
        self.σ2_sq = 0.05

        # 响应参数可迭代（初值）
        self.v1 = 1.0
        self.v2 = 1.0

        # 记录历史
        self.history = {
            'phi': [],
            'μ1': [],  # 虽为常数也记录以便绘图
            'μ2': [],
            'σ1_sq': [],
            'σ2_sq': [],
            'v1': [],
            'v2': []
        }

    def calculate_overall_moments(self):
        """
        计算总体矩：使用固定的 μ1, μ2
        """
        self.m = (1 - self.phi) * self.μ1 + self.phi * self.μ2
        self.x2_1 = self.μ1 ** 2 + self.σ1_sq
        self.x2_2 = self.μ2 ** 2 + self.σ2_sq
        self.x2_avg = (1 - self.phi) * self.x2_1 + self.phi * self.x2_2
        self.σ_sq = self.x2_avg - self.m ** 2

    def calculate_feedback_term(self):
        """
        计算反馈项 F
        """
        weighted_x2 = (
            (1 - self.phi) ** 2 * self.x2_1 ** 2 +
            2 * self.phi * (1 - self.phi) * self.x2_1 * self.x2_2 +
            self.phi ** 2 * self.x2_2 ** 2
        )
        self.F = (
            self.ρ_d * self.σ_d ** 2 +
            2 * self.σ_e ** 2 * weighted_x2
        )

    def calculate_noise_strengths(self):
        """
        计算各状态的噪声强度 D1, D2
        """
        self.D1 = (
            self.σ_c ** 2 +
            self.σ_d ** 2 * self.x2_1 +
            self.σ_e ** 2 * self.x2_1 ** 2
        )
        self.D2 = (
            self.σ_c ** 2 +
            self.σ_d ** 2 * self.x2_2 +
            self.σ_e ** 2 * self.x2_2 ** 2
        )

    def calculate_effective_drive(self):
        """
        计算有效驱动力 μ_eff
        """
        self.μ_eff = self.μ_c + self.μ_d * self.m + self.μ_e * self.m ** 2

    def _safe_denom(self, denom):
        """
        安全处理 denom：避免分母为 0 或极小值
        """
        denom_sign = 1.0 if denom == 0.0 else np.sign(denom)
        if abs(denom) < self.denom_eps:
            denom_safe = denom_sign * self.denom_eps
            return denom_safe, True
        return denom, False

    def update_states(self):
        """
        更新状态（注意 μ1, μ2 固定）：
        返回更新的 σ1_sq, σ2_sq, phi, v1, v2 以及 denom 保护标志
        """
        denom1 = 1 - self.v1 * self.F
        denom2 = 1 - self.v2 * self.F

        denom1_safe, adj1 = self._safe_denom(denom1)
        denom2_safe, adj2 = self._safe_denom(denom2)

        denom_warn_flag = adj1 or adj2

        # μ1, μ2 保持固定，不再计算 μ_new
        # 计算方差（使用安全 denom）
        σ1_sq_new = self.D1 / (2 * denom1_safe ** 2)
        σ2_sq_new = self.D2 / (2 * denom2_safe ** 2)

        # 计算 phi_new（使用 state2 的分布右侧概率）
        var2 = (self.D2 / denom2_safe ** 2) if denom2_safe != 0 else self.D2 / (self.denom_eps ** 2)
        z_raw = (self.μ2) / np.sqrt(2 * var2) if var2 > 0 else 0.0
        z = np.clip(z_raw, -10.0, 10.0)
        phi_new = 0.5 * (1 + erf(z))

        v1_new = 1 / denom1_safe
        v2_new = 1 / denom2_safe

        return σ1_sq_new, σ2_sq_new, phi_new, v1_new, v2_new, denom_warn_flag

    def apply_bistable_constraints(self, μ1, μ2):
        """
        保留约束函数（此处 μ1, μ2 固定，因此可仅做检查）
        """
        μ1_constrained = min(μ1, -0.1)
        μ2_constrained = max(μ2, 0.1)
        μ1_constrained = max(μ1_constrained, -2.0)
        μ2_constrained = min(μ2_constrained, 2.0)
        return μ1_constrained, μ2_constrained

    def solve_iterative(self, verbose=True):
        """
        迭代求解自洽方程组：
        - μ1, μ2 固定为 -1 和 +1
        - 仅更新方差、phi、v1、v2
        """
        self.initialize_parameters()
        self.denom_warn_count = 0

        for i in range(self.max_iter):
            # 记录历史（包括固定的 μ1, μ2）
            self.history['phi'].append(self.phi)
            self.history['μ1'].append(self.μ1)
            self.history['μ2'].append(self.μ2)
            self.history['σ1_sq'].append(self.σ1_sq)
            self.history['σ2_sq'].append(self.σ2_sq)
            self.history['v1'].append(self.v1)
            self.history['v2'].append(self.v2)

            # 计算中间量（注意 μ1, μ2 固定）
            self.calculate_overall_moments()
            self.calculate_feedback_term()
            self.calculate_noise_strengths()
            self.calculate_effective_drive()

            # 更新（μ1, μ2 不变）
            σ1_sq_new, σ2_sq_new, phi_new, v1_new, v2_new, denom_warn_flag = self.update_states()

            if denom_warn_flag:
                self.denom_warn_count += 1
                if verbose:
                    print(f"Warning: denom protection triggered at iteration {i} (count={self.denom_warn_count})")

            # μ1, μ2 固定 —— 仍可做一次检查，保持原样
            self.μ1, self.μ2 = self.apply_bistable_constraints(self.μ1, self.μ2)

            # 松弛更新仅作用于可变项
            self.σ1_sq = self.relaxation * σ1_sq_new + (1 - self.relaxation) * self.σ1_sq
            self.σ2_sq = self.relaxation * σ2_sq_new + (1 - self.relaxation) * self.σ2_sq
            self.phi = self.relaxation * phi_new + (1 - self.relaxation) * self.phi
            self.v1 = self.relaxation * v1_new + (1 - self.relaxation) * self.v1
            self.v2 = self.relaxation * v2_new + (1 - self.relaxation) * self.v2

            # 检查收敛（同样基于 φ、σ1_sq、σ2_sq 或者保留原判断）
            if self.check_convergence():
                if verbose:
                    print(f"收敛于第 {i} 次迭代")
                break

            if self.denom_warn_count >= self.denom_warn_limit:
                if verbose:
                    print(f"Terminating early: denom protection triggered {self.denom_warn_count} times (>= {self.denom_warn_limit})")
                break
        else:
            if verbose:
                print("警告: 达到最大迭代次数但未收敛")

        # 最终计算总体矩
        self.calculate_overall_moments()

        return self.get_results()

    def check_convergence(self):
        """
        检查收敛：这里我们改为检测 phi、σ1_sq、σ2_sq 的变化
        """
        if len(self.history['phi']) < 2:
            return False

        delta_phi = abs(self.history['phi'][-1] - self.history['phi'][-2])
        delta_σ1 = abs(self.history['σ1_sq'][-1] - self.history['σ1_sq'][-2])
        delta_σ2 = abs(self.history['σ2_sq'][-1] - self.history['σ2_sq'][-2])

        return (delta_phi < self.tol and delta_σ1 < self.tol and delta_σ2 < self.tol)

    def get_results(self):
        """
        返回最终结果
        """
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
        """
        绘制收敛过程
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].plot(self.history['phi'])
        axes[0, 0].set_xlabel('迭代次数')
        axes[0, 0].set_ylabel('共存比例 φ')
        axes[0, 0].set_title('共存比例收敛过程')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(self.history['μ1'], label='状态1均值 μ₁')
        axes[0, 1].plot(self.history['μ2'], label='状态2均值 μ₂')
        axes[0, 1].set_xlabel('迭代次数')
        axes[0, 1].set_ylabel('状态均值')
        axes[0, 1].set_title('状态均值（固定）')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(self.history['σ1_sq'], label='状态1方差 σ₁²')
        axes[1, 0].plot(self.history['σ2_sq'], label='状态2方差 σ₂²')
        axes[1, 0].set_xlabel('迭代次数')
        axes[1, 0].set_ylabel('状态方差')
        axes[1, 0].set_title('状态方差收敛过程')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(self.history['v1'], label='状态1响应 v₁')
        axes[1, 1].plot(self.history['v2'], label='状态2响应 v₂')
        axes[1, 1].set_xlabel('迭代次数')
        axes[1, 1].set_ylabel('响应参数')
        axes[1, 1].set_title('响应参数收敛过程')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_bimodal_distribution(self, x_range=(-2, 2), n_points=1000):
        """
        绘制双峰分布（μ1, μ2 为固定值）
        """
        x = np.linspace(x_range[0], x_range[1], n_points)

        gaussian1 = ((1 - self.phi) *
                     np.exp(-(x - self.μ1) ** 2 / (2 * self.σ1_sq)) /
                     np.sqrt(2 * np.pi * self.σ1_sq))
        gaussian2 = (self.phi *
                     np.exp(-(x - self.μ2) ** 2 / (2 * self.σ2_sq)) /
                     np.sqrt(2 * np.pi * self.σ2_sq))
        total_dist = gaussian1 + gaussian2

        plt.figure(figsize=(10, 6))
        plt.plot(x, gaussian1, 'b--', alpha=0.7,
                 label=f'状态1 (μ₁={self.μ1:.3f}), σ₁²={self.σ1_sq:.3f}')
        plt.plot(x, gaussian2, 'r--', alpha=0.7,
                 label=f'状态2 (μ₂={self.μ2:.3f}), σ₂²={self.σ2_sq:.3f}')
        plt.plot(x, total_dist, 'k-', linewidth=2,
                 label=f'总体分布, φ={self.phi:.3f}')

        plt.axvline(x=self.μ1, color='b', linestyle=':', alpha=0.5, label=f'稳定状态 x={self.μ1}')
        plt.axvline(x=self.μ2, color='r', linestyle=':', alpha=0.5, label=f'稳定状态 x={self.μ2}')
        plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5, label='不稳定状态 x=0')

        plt.xlabel('x')
        plt.ylabel('概率密度')
        plt.title('双稳态系统的概率分布（μ1, μ2 固定）')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


# 参数扫描示例
def parameter_scan(mu_d_min=0.0, mu_d_max=2.0, n_points=21, verbose=False):
    """
    参数扫描：观察系统行为随线性耦合强度 μ_d 的变化
    返回：扫描点数组以及结果字典
    """
    μ_d_values = np.linspace(mu_d_min, mu_d_max, n_points)
    phi_values = []
    μ1_values = []
    μ2_values = []
    denom_warn_counts = []
    m_values = []
    σ_sq_values = []

    for μ_d in μ_d_values:
        params = {
            'μ_c': 0.5,
            'μ_d': μ_d,
            'μ_e': 0.2,
            'σ_c': 0.1,
            'σ_d': 0.3,
            'σ_e': 0.1,
            'ρ_d': 0.0,
            'S': 100,
            'max_iter': 500,
            'tol': 1e-5,
            'relaxation': 0.3,
            'denom_eps': 1e-6,
            'denom_warn_limit': 20
        }

        solver = BistableCavitySolver(params)
        results = solver.solve_iterative(verbose=verbose)
        phi_values.append(results['phi'])
        μ1_values.append(results['μ1'])
        μ2_values.append(results['μ2'])
        denom_warn_counts.append(results['denom_warn_count'])
        m_values.append(results['m'])
        σ_sq_values.append(results['σ_sq'])

    # 绘制结果
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 共存比例随 μ_d 的变化
    axes[0].plot(μ_d_values, phi_values, 'o-', linewidth=2, markersize=5)
    axes[0].set_xlabel('线性耦合强度 μ_d')
    axes[0].set_ylabel('共存比例 φ')
    axes[0].set_title('共存比例随 μ_d 的变化')
    axes[0].grid(True, alpha=0.3)

    # 两个状态均值随 μ_d 的变化（固定值）
    axes[1].plot(μ_d_values, μ1_values, 's-', linewidth=2, markersize=5, label='状态1均值 μ₁')
    axes[1].plot(μ_d_values, μ2_values, '^-', linewidth=2, markersize=5, label='状态2均值 μ₂')
    axes[1].set_xlabel('线性耦合强度 μ_d')
    axes[1].set_ylabel('状态均值')
    axes[1].set_title('状态均值随 μ_d 的变化（固定）')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 总体均值与方差随 μ_d 的变化
    axes[2].plot(μ_d_values, m_values, 'o-', label='总体均值 m')
    axes[2].plot(μ_d_values, σ_sq_values, 's-', label='总体方差 σ²')
    axes[2].set_xlabel('线性耦合强度 μ_d')
    axes[2].set_ylabel('总体矩')
    axes[2].set_title('总体均值与方差随 μ_d 的变化')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        'μ_d_values': μ_d_values,
        'phi_values': np.array(phi_values),
        'μ1_values': np.array(μ1_values),
        'μ2_values': np.array(μ2_values),
        'denom_warn_counts': np.array(denom_warn_counts),
        'm_values': np.array(m_values),
        'σ_sq_values': np.array(σ_sq_values)
    }


# 测试主程序
if __name__ == "__main__":
    # 默认单次求解参数
    params = {
        'μ_c': 0.0,
        'μ_d': 0.2,
        'μ_e': 0.2,
        'σ_c': 0.4,
        'σ_d': 0.3,
        'σ_e': 0.1,
        'ρ_d': 1.0,
        'S': 100,
        'max_iter': 1000,
        'tol': 1e-6,
        'relaxation': 0.3,
        'denom_eps': 1e-6,
        'denom_warn_limit': 20
    }

    solver = BistableCavitySolver(params)
    results = solver.solve_iterative(verbose=True)

    print("\n双稳态系统空穴法求解结果（μ1=-1, μ2=+1 固定）:")
    print(f"总体均值 m = {results['m']:.6f}")
    print(f"总体方差 σ² = {results['σ_sq']:.6f}")
    print(f"共存比例 φ = {results['phi']:.6f}")
    print(f"状态1: μ₁ = {results['μ1']:.6f}, σ₁² = {results['σ1_sq']:.6f}")
    print(f"状态2: μ₂ = {results['μ2']:.6f}, σ₂² = {results['σ2_sq']:.6f}")
    print(f"响应参数: v₁ = {results['v1']:.6f}, v₂ = {results['v2']:.6f}")
    print(f"反馈项 F = {results['F']:.6f}")
    print(f"噪声强度: D₁ = {results['D1']:.6f}, D₂ = {results['D2']:.6f}")
    print(f"denom 保护触发次数: {results['denom_warn_count']}")

    # 绘制单次求解的收敛过程与双峰分布
    solver.plot_convergence()
    solver.plot_bimodal_distribution(x_range=(-3, 3))

    # 询问用户是否执行参数扫描
    ans = input("是否执行 μ_d 参数扫描并绘图？(y/n) [y]: ").strip().lower() or 'y'
    if ans == 'y':
        scan_results = parameter_scan(mu_d_min=0.0, mu_d_max=2.0, n_points=25, verbose=False)
        # 可选地打印一些 summary
        print("\n参数扫描完成。示例输出：")
        for i, mu_d in enumerate(scan_results['μ_d_values']):
            print(f"μ_d={mu_d:.3f}, φ={scan_results['phi_values'][i]:.4f}, denom_warn={scan_results['denom_warn_counts'][i]}")

    print("程序结束。")