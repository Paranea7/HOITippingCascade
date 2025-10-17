import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import quad
import warnings

warnings.filterwarnings('ignore')


class TwoStateRSB:
    def __init__(self, beta=1.0, S=100):
        # 温度参数
        self.beta = beta

        # 系统大小
        self.S = S

        # 模型参数（均值项）
        self.c_bar = 0.0
        self.d_bar = 0.1
        self.e_bar = -0.05

        # 模型参数（方差项）
        self.sigma_c = 0.5
        self.sigma_d = 0.3
        self.sigma_e = 0.2

        # RSB 参数
        self.q_d = 0.0  # 自重叠
        self.q_1 = 0.0  # 状态内重叠
        self.q_0 = 0.0  # 状态间重叠
        self.m_1 = 0.0  # 状态1的平均值
        self.m_2 = 0.0  # 状态2的平均值
        self.w = 0.5  # 状态1的权重

    def effective_potential(self, x):
        """有效势函数"""
        return 0.25 * x ** 4 - 0.5 * x ** 2

    def gaussian_integral(self, func, mu=0, sigma=1, n_points=1000):
        """高斯积分"""
        x_min = mu - 5 * sigma
        x_max = mu + 5 * sigma
        x_vals = np.linspace(x_min, x_max, n_points)
        dx = (x_max - x_min) / (n_points - 1)

        integrand = func(x_vals) * np.exp(-0.5 * ((x_vals - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        return np.sum(integrand) * dx

    def effective_hamiltonian(self, x, z0, z1):
        """有效哈密顿量"""
        # 均值项贡献
        mean_contribution = self.c_bar + self.S * self.d_bar * self.w + self.S ** 2 * self.e_bar * self.w ** 2

        # 方差项贡献
        sigma_0 = np.sqrt(self.sigma_c ** 2 + self.sigma_d ** 2 * self.q_0 + self.sigma_e ** 2 * self.q_0 ** 2)
        sigma_1 = np.sqrt(
            self.sigma_d ** 2 * (self.q_1 - self.q_0) + self.sigma_e ** 2 * (self.q_1 ** 2 - self.q_0 ** 2))

        random_field = sigma_0 * z0 + sigma_1 * z1

        return self.effective_potential(x) - x * (mean_contribution + random_field)

    def compute_magnetization(self, z0_vals, z1_vals):
        """计算磁化强度"""
        numerator_1 = 0.0
        denominator_1 = 0.0
        numerator_2 = 0.0
        denominator_2 = 0.0

        for z0 in z0_vals:
            for z1 in z1_vals:
                # 状态1的有效哈密顿量
                H1 = lambda x: self.effective_hamiltonian(x, z0, z1)

                # 计算配分函数和磁化强度
                Z1, _ = quad(lambda x: np.exp(-self.beta * H1(x)), -5, 5)
                m1, _ = quad(lambda x: x * np.exp(-self.beta * H1(x)), -5, 5)
                m1 /= Z1

                numerator_1 += m1 * np.exp(-0.5 * z0 ** 2 - 0.5 * z1 ** 2)
                denominator_1 += np.exp(-0.5 * z0 ** 2 - 0.5 * z1 ** 2)

                # 状态2（使用不同的随机场实现）
                H2 = lambda x: self.effective_hamiltonian(x, -z0, z1)  # 对称性假设
                Z2, _ = quad(lambda x: np.exp(-self.beta * H2(x)), -5, 5)
                m2, _ = quad(lambda x: x * np.exp(-self.beta * H2(x)), -5, 5)
                m2 /= Z2

                numerator_2 += m2 * np.exp(-0.5 * z0 ** 2 - 0.5 * z1 ** 2)
                denominator_2 += np.exp(-0.5 * z0 ** 2 - 0.5 * z1 ** 2)

        m1_avg = numerator_1 / denominator_1
        m2_avg = numerator_2 / denominator_2

        return m1_avg, m2_avg

    def compute_free_energy_difference(self, z0_vals, z1_vals):
        """计算自由能差"""
        f1 = 0.0
        f2 = 0.0

        for z0 in z0_vals:
            for z1 in z1_vals:
                # 状态1的自由能
                H1 = lambda x: self.effective_hamiltonian(x, z0, z1)
                Z1, _ = quad(lambda x: np.exp(-self.beta * H1(x)), -5, 5)
                f1_component = -np.log(Z1) / self.beta
                f1 += f1_component * np.exp(-0.5 * z0 ** 2 - 0.5 * z1 ** 2)

                # 状态2的自由能
                H2 = lambda x: self.effective_hamiltonian(x, -z0, z1)
                Z2, _ = quad(lambda x: np.exp(-self.beta * H2(x)), -5, 5)
                f2_component = -np.log(Z2) / self.beta
                f2 += f2_component * np.exp(-0.5 * z0 ** 2 - 0.5 * z1 ** 2)

        # 归一化
        normalization = len(z0_vals) * len(z1_vals)
        f1 /= normalization
        f2 /= normalization

        return f1, f2

    def update_order_parameters(self, z0_vals, z1_vals):
        """更新序参数"""
        # 计算磁化强度
        m1, m2 = self.compute_magnetization(z0_vals, z1_vals)

        # 计算自由能差
        f1, f2 = self.compute_free_energy_difference(z0_vals, z1_vals)

        # 更新权重
        delta_f = f2 - f1
        self.w = 1.0 / (1.0 + np.exp(-self.beta * delta_f))

        # 更新重叠参数
        self.q_d = self.w * m1 ** 2 + (1 - self.w) * m2 ** 2
        self.q_1 = self.w ** 2 * m1 ** 2 + (1 - self.w) ** 2 * m2 ** 2 + 2 * self.w * (1 - self.w) * m1 * m2
        self.q_0 = self.w * (1 - self.w) * (m1 ** 2 + m2 ** 2) + (self.w ** 2 + (1 - self.w) ** 2) * m1 * m2

        # 更新状态平均值
        self.m_1 = m1
        self.m_2 = m2

        return m1, m2, self.w

    def solve_self_consistent(self, n_iter=50, tolerance=1e-6, n_z=5):
        """自洽求解"""
        print("开始自洽求解...")
        print(f"{'迭代':<4} {'m1':<10} {'m2':<10} {'权重':<10} {'q_d':<10} {'q_1':<10} {'q_0':<10}")
        print("-" * 80)

        # 高斯随机场采样点
        z0_vals = np.linspace(-2, 2, n_z)
        z1_vals = np.linspace(-2, 2, n_z)

        for i in range(n_iter):
            m1_old, m2_old, w_old = self.m_1, self.m_2, self.w

            m1, m2, w = self.update_order_parameters(z0_vals, z1_vals)

            # 检查收敛
            delta_m1 = abs(m1 - m1_old)
            delta_m2 = abs(m2 - m2_old)
            delta_w = abs(w - w_old)

            print(
                f"{i + 1:<4} {m1:<10.6f} {m2:<10.6f} {w:<10.6f} {self.q_d:<10.6f} {self.q_1:<10.6f} {self.q_0:<10.6f}")

            if delta_m1 < tolerance and delta_m2 < tolerance and delta_w < tolerance:
                print(f"\n收敛于第 {i + 1} 次迭代")
                break
        else:
            print(f"\n在 {n_iter} 次迭代后未完全收敛")

        return self.m_1, self.m_2, self.w

    def plot_potential_landscape(self):
        """绘制势能景观"""
        x = np.linspace(-2, 2, 1000)

        # 计算有效势
        V_eff = 0.25 * x ** 4 - 0.5 * x ** 2 - self.c_bar * x - 0.5 * self.S * self.d_bar * x ** 2 - (
                    1 / 3) * self.S ** 2 * self.e_bar * x ** 3

        plt.figure(figsize=(10, 6))
        plt.plot(x, V_eff, 'b-', linewidth=2, label='有效势')

        # 标记稳定状态
        if hasattr(self, 'm_1') and hasattr(self, 'm_2'):
            plt.axvline(self.m_1, color='red', linestyle='--', alpha=0.7, label=f'状态1: m={self.m_1:.3f}')
            plt.axvline(self.m_2, color='green', linestyle='--', alpha=0.7, label=f'状态2: m={self.m_2:.3f}')

        plt.xlabel('磁化强度 m')
        plt.ylabel('有效势 V(m)')
        plt.title('双稳系统的势能景观')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def analyze_phase_transition(self, param_range, param_name='d_bar'):
        """分析相变行为"""
        m1_values = []
        m2_values = []
        w_values = []

        original_value = getattr(self, param_name)

        for param_val in param_range:
            setattr(self, param_name, param_val)

            # 重置序参数
            self.q_d = 0.0
            self.q_1 = 0.0
            self.q_0 = 0.0
            self.m_1 = 0.0
            self.m_2 = 0.0
            self.w = 0.5

            # 求解自洽方程
            z0_vals = np.linspace(-2, 2, 5)
            z1_vals = np.linspace(-2, 2, 5)

            # 快速求解（减少迭代次数）
            for _ in range(10):
                self.update_order_parameters(z0_vals, z1_vals)

            m1_values.append(self.m_1)
            m2_values.append(self.m_2)
            w_values.append(self.w)

        # 恢复原始参数值
        setattr(self, param_name, original_value)

        # 绘制相图
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(param_range, m1_values, 'r-', label='状态1 (m1)')
        plt.plot(param_range, m2_values, 'g-', label='状态2 (m2)')
        plt.xlabel(param_name)
        plt.ylabel('磁化强度')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 2)
        plt.plot(param_range, w_values, 'b-', label='状态1权重')
        plt.xlabel(param_name)
        plt.ylabel('权重 w')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 3)
        plt.plot(param_range, np.array(m1_values) - np.array(m2_values), 'purple', label='|m1 - m2|')
        plt.xlabel(param_name)
        plt.ylabel('状态差异')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle(f'相变行为 vs {param_name}', y=1.02)
        plt.show()


# 使用示例
if __name__ == "__main__":
    # 创建两状态RSB分析器
    rsb = TwoStateRSB(beta=2.0, S=100)

    # 设置参数（示例）
    rsb.c_bar = 0.0  # 外部场的均值
    rsb.d_bar = 0.15  # 线性耦合的均值
    rsb.e_bar = -0.08  # 非线性耦合的均值

    rsb.sigma_c = 0.6  # 外部场的涨落
    rsb.sigma_d = 0.4  # 线性耦合的涨落
    rsb.sigma_e = 0.3  # 非线性耦合的涨落

    # 自洽求解
    m1, m2, w = rsb.solve_self_consistent(n_iter=2000, n_z=5)

    print(f"\n最终结果:")
    print(f"状态1: m = {m1:.6f}")
    print(f"状态2: m = {m2:.6f}")
    print(f"状态1权重: w = {w:.6f}")
    print(f"重叠参数: q_d = {rsb.q_d:.6f}, q_1 = {rsb.q_1:.6f}, q_0 = {rsb.q_0:.6f}")

    # 绘制势能景观
    rsb.plot_potential_landscape()

    # 分析相变行为
    d_bar_range = np.linspace(0.05, 0.25, 20)
    rsb.analyze_phase_transition(d_bar_range, 'd_bar')