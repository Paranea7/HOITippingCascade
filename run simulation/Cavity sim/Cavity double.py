import numpy as np
from scipy.special import erf
from scipy.integrate import quad
import matplotlib.pyplot as plt


class RevisedBimodalCavityMethod:
    """
    修正的双峰分布空穴法计算类
    """

    def __init__(self, params):
        """
        初始化参数

        Parameters:
        params: dict, 包含模型参数的字典
        """
        self.μ_c = params['μ_c']
        self.μ_d = params['μ_d']
        self.μ_e = params['μ_e']
        self.σ_c = params['σ_c']
        self.σ_d = params['σ_d']
        self.σ_e = params['σ_e']
        self.ρ_d = params.get('ρ_d', 0.0)
        self.S = params.get('S', np.inf)

    def initialize_parameters(self):
        """
        初始化状态参数 - 允许非对称分布
        """
        # 初始猜测值
        self.phi = 0.5  # 共存比例
        self.v = 1.0  # 响应参数

        # 两个状态的参数 - 不再固定为 -1 和 1
        self.μ1 = -0.8  # 状态1均值 (非共存) - 初始猜测
        self.μ2 = 0.9  # 状态2均值 (共存) - 初始猜测
        self.σ1_sq = 0.1  # 状态1方差
        self.σ2_sq = 0.1  # 状态2方差

    def calculate_moments(self):
        """
        计算总体矩
        """
        # 总体均值
        self.m = (1 - self.phi) * self.μ1 + self.phi * self.μ2

        # 总体二阶矩
        self.x2_avg = (1 - self.phi) * (self.μ1 ** 2 + self.σ1_sq) + self.phi * (self.μ2 ** 2 + self.σ2_sq)

        # 总体方差
        self.σ_sq = self.x2_avg - self.m ** 2

    def calculate_noise_variance(self):
        """
        计算噪声方差
        """
        self.D = (self.σ_c ** 2 +
                  self.σ_d ** 2 * self.x2_avg +
                  self.σ_e ** 2 * self.x2_avg ** 2)

    def calculate_denominator(self):
        """
        计算分母项 - 考虑完整的高阶相互作用
        """
        # 注意：这里包含了因子2，考虑了两种高阶相互作用项
        self.denominator = (1 - self.v *
                            (self.ρ_d * self.σ_d ** 2 * self.phi +
                             2 * self.σ_e ** 2 * self.phi ** 2 * (self.μ2 ** 2 + self.σ2_sq)))

    def calculate_effective_drive(self):
        """
        计算有效驱动力
        """
        self.μ_eff = self.μ_c + self.μ_d * self.m + self.μ_e * self.m ** 2

    def update_states(self):
        """
        更新状态参数 - 考虑非对称分布
        """
        # 更新两个状态的均值
        μ1_new = self.μ_eff / self.denominator
        μ2_new = self.μ_eff / self.denominator

        # 更新两个状态的方差
        σ1_sq_new = self.D / (2 * self.denominator ** 2)
        σ2_sq_new = self.D / (2 * self.denominator ** 2)

        # 更新共存比例 - 使用阈值确定状态
        # 这里我们使用一个阈值来区分两个状态
        # 在实际应用中，这个阈值可能需要根据系统特性调整
        threshold = (μ1_new + μ2_new) / 2  # 使用两个状态均值的中间值作为阈值

        def integrand(x):
            return (np.exp(-(x - self.μ_eff / self.denominator) ** 2 /
                           (2 * self.D / self.denominator ** 2)) /
                    np.sqrt(2 * np.pi * self.D / self.denominator ** 2))

        # 计算状态2的概率（x > threshold的区域）
        phi_new, _ = quad(integrand, threshold, np.inf)

        # 更新响应参数
        v_new = 1 / self.denominator

        return μ1_new, μ2_new, σ1_sq_new, σ2_sq_new, phi_new, v_new, threshold

    def solve(self, max_iter=1000, tol=1e-6, verbose=True):
        """
        求解自洽方程组

        Parameters:
        max_iter: int, 最大迭代次数
        tol: float, 收敛容差
        verbose: bool, 是否打印迭代信息

        Returns:
        dict: 包含求解结果的字典
        """
        self.initialize_parameters()

        for i in range(max_iter):
            # 计算矩和噪声
            self.calculate_moments()
            self.calculate_noise_variance()
            self.calculate_denominator()
            self.calculate_effective_drive()

            # 更新状态参数
            μ1_new, μ2_new, σ1_sq_new, σ2_sq_new, phi_new, v_new, threshold = self.update_states()

            # 检查收敛
            converged = (abs(phi_new - self.phi) < tol and
                         abs(v_new - self.v) < tol and
                         abs(μ1_new - self.μ1) < tol and
                         abs(μ2_new - self.μ2) < tol and
                         abs(σ1_sq_new - self.σ1_sq) < tol and
                         abs(σ2_sq_new - self.σ2_sq) < tol)

            # 更新参数
            self.μ1, self.μ2, self.σ1_sq, self.σ2_sq, self.phi, self.v = (
                μ1_new, μ2_new, σ1_sq_new, σ2_sq_new, phi_new, v_new)

            if verbose and i % 100 == 0:
                print(f"Iteration {i}: φ = {self.phi:.4f}, m = {self.m:.4f}, μ₁ = {self.μ1:.4f}, μ₂ = {self.μ2:.4f}")

            if converged:
                if verbose:
                    print(f"收敛于第 {i} 次迭代")
                    print(f"最终状态: φ = {self.phi:.4f}, μ₁ = {self.μ1:.4f}, μ₂ = {self.μ2:.4f}")
                break
        else:
            if verbose:
                print("警告: 达到最大迭代次数但未收敛")

        # 计算最终矩
        self.calculate_moments()

        return {
            'm': self.m,
            'σ_sq': self.σ_sq,
            'phi': self.phi,
            'v': self.v,
            'μ1': self.μ1,
            'μ2': self.μ2,
            'σ1_sq': self.σ1_sq,
            'σ2_sq': self.σ2_sq,
            'D': self.D,
            'threshold': (self.μ1 + self.μ2) / 2
        }

    def plot_bimodal_distribution(self, x_range=(-3, 3), n_points=1000):
        """
        绘制双峰分布

        Parameters:
        x_range: tuple, x轴范围
        n_points: int, 点数
        """
        x = np.linspace(x_range[0], x_range[1], n_points)

        # 计算两个高斯分布
        gaussian1 = ((1 - self.phi) *
                     np.exp(-(x - self.μ1) ** 2 / (2 * self.σ1_sq)) /
                     np.sqrt(2 * np.pi * self.σ1_sq))
        gaussian2 = (self.phi *
                     np.exp(-(x - self.μ2) ** 2 / (2 * self.σ2_sq)) /
                     np.sqrt(2 * np.pi * self.σ2_sq))

        # 总体分布
        total_dist = gaussian1 + gaussian2

        plt.figure(figsize=(10, 6))
        plt.plot(x, gaussian1, 'b--', alpha=0.7, label=f'状态1 (非共存), μ₁={self.μ1:.2f}, σ₁²={self.σ1_sq:.2f}')
        plt.plot(x, gaussian2, 'r--', alpha=0.7, label=f'状态2 (共存), μ₂={self.μ2:.2f}, σ₂²={self.σ2_sq:.2f}')
        plt.plot(x, total_dist, 'k-', linewidth=2, label=f'总体分布, φ={self.phi:.2f}')

        # 标记阈值
        threshold = (self.μ1 + self.μ2) / 2
        plt.axvline(x=threshold, color='g', linestyle=':', label=f'阈值 = {threshold:.2f}')

        plt.xlabel('x')
        plt.ylabel('概率密度')
        plt.title('非对称双峰分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


# 测试代码
if __name__ == "__main__":
    # 参数设置
    params = {
        'μ_c': 0.5,  # 外部输入均值
        'μ_d': 1.0,  # 线性耦合均值
        'μ_e': 0.2,  # 二次耦合均值
        'σ_c': 0.1,  # 外部输入涨落
        'σ_d': 0.3,  # 线性耦合涨落
        'σ_e': 0.1,  # 二次耦合涨落
        'ρ_d': 0.0,  # 线性耦合相关性
        'S': 100  # 物种数量
    }

    # 创建求解器实例
    solver = RevisedBimodalCavityMethod(params)

    # 求解自洽方程
    results = solver.solve(verbose=True)

    # 打印结果
    print("\n修正的双峰分布空穴法结果:")
    print(f"总体均值 m = {results['m']:.4f}")
    print(f"总体方差 σ² = {results['σ_sq']:.4f}")
    print(f"共存比例 φ = {results['phi']:.4f}")
    print(f"响应参数 v = {results['v']:.4f}")
    print(f"状态1: μ₁ = {results['μ1']:.4f}, σ₁² = {results['σ1_sq']:.4f}")
    print(f"状态2: μ₂ = {results['μ2']:.4f}, σ₂² = {results['σ2_sq']:.4f}")
    print(f"噪声强度 D = {results['D']:.4f}")
    print(f"状态阈值 = {results['threshold']:.4f}")

    # 绘制双峰分布
    solver.plot_bimodal_distribution()


# 参数扫描示例 - 观察系统行为变化
def parameter_scan():
    """
    参数扫描示例：观察系统行为随参数变化
    """
    μ_d_values = np.linspace(0.5, 2.0, 20)
    phi_values = []
    μ1_values = []
    μ2_values = []

    for μ_d in μ_d_values:
        params = {
            'μ_c': 0.5, 'μ_d': μ_d, 'μ_e': 0.2,
            'σ_c': 0.1, 'σ_d': 0.3, 'σ_e': 0.1,
            'ρ_d': 0.0, 'S': 100
        }

        solver = RevisedBimodalCavityMethod(params)
        results = solver.solve(verbose=False)
        phi_values.append(results['phi'])
        μ1_values.append(results['μ1'])
        μ2_values.append(results['μ2'])

    # 绘制结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 共存比例随μ_d的变化
    ax1.plot(μ_d_values, phi_values, 'o-', linewidth=2)
    ax1.set_xlabel('线性耦合强度 μ_d')
    ax1.set_ylabel('共存比例 φ')
    ax1.set_title('共存比例随线性耦合强度的变化')
    ax1.grid(True, alpha=0.3)

    # 两个状态均值随μ_d的变化
    ax2.plot(μ_d_values, μ1_values, 's-', linewidth=2, label='状态1均值 μ₁')
    ax2.plot(μ_d_values, μ2_values, '^-', linewidth=2, label='状态2均值 μ₂')
    ax2.set_xlabel('线性耦合强度 μ_d')
    ax2.set_ylabel('状态均值')
    ax2.set_title('状态均值随线性耦合强度的变化')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# 运行参数扫描
parameter_scan()