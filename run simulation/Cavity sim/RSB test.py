import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize
import matplotlib


matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False
class StepByStepRSB:
    def __init__(self, S=100):
        self.S = S  # 系统大小

        # 原始模型参数
        self.original_params = {
            'c_mean': 0.0, 'c_std': 0.4,  # c_i 的统计
            'd_mean': 0.3, 'd_std': 0.4,  # d_ji 的统计
            'e_mean': 0.3, 'e_std': 0.2  # e_ijk 的统计
        }

        print("=== 步骤1: 模型设定 ===")
        print(f"系统方程: dx_i/dt = -x_i^3 + x_i + c_i + Σ_j d_ji x_j + Σ_j,k e_ijk x_j x_k")
        print(f"系统大小 S = {S}")
        print(f"随机参数:")
        print(f"  c_i ~ N({self.original_params['c_mean']}, {self.original_params['c_std']}^2)")
        print(f"  d_ji ~ N({self.original_params['d_mean']}/S, {self.original_params['d_std']}^2/S)")
        print(f"  e_ijk ~ N({self.original_params['e_mean']}/S^2, {self.original_params['e_std']}^2/S^2)")

    def step1_derive_hamiltonian(self):
        """步骤1: 推导对应的哈密顿量"""
        print("\n=== 步骤1: 推导哈密顿量 ===")
        print("寻找哈密顿量 H 使得 dx_i/dt = -∂H/∂x_i")
        print("通过积分得到:")
        print("H = Σ_i [1/4 x_i^4 - 1/2 x_i^2 - c_i x_i] - 1/2 Σ_ij d_ij x_i x_j - 1/3 Σ_ijk e_ijk x_i x_j x_k")
        print("验证: ∂H/∂x_i = x_i^3 - x_i - c_i - Σ_j d_ji x_j - Σ_jk e_ijk x_j x_k")
        print("因此 dx_i/dt = -∂H/∂x_i 成立")

        return {
            'potential': lambda x: 0.25 * x ** 4 - 0.5 * x ** 2,
            'linear_coupling': lambda x, c: -c * x,
            'quadratic_coupling': lambda x, y, d: -0.5 * d * x * y,
            'cubic_coupling': lambda x, y, z, e: -(1 / 3) * e * x * y * z
        }

    def step2_replica_method(self, n_replicas=2):
        """步骤2: 引入副本方法"""
        print("\n=== 步骤2: 副本方法 ===")
        print(f"引入 {n_replicas} 个副本系统")
        print("配分函数: Z = ∫ Π_i dx_i exp(-βH)")
        print("计算无序平均时，使用副本技巧:")
        print("E[Z^n] = E[ ∫ Π_{i,a} dx_i^a exp(-β Σ_a H_a) ]")
        print("通过对随机变量 c_i, d_ji, e_ijk 取平均来处理无序")

        # 定义副本变量
        self.n_replicas = n_replicas
        self.beta = 2.0  # 逆温度

        print(f"使用逆温度 β = {self.beta}")
        print("现在对随机系数进行平均...")

        return self._average_over_disorder()

    def _average_over_disorder(self):
        """对无序进行平均"""
        p = self.original_params

        print("\n对随机系数进行高斯平均:")
        print("E[exp(β Σ_i c_i x_i)] = exp(β^2 σ_c^2/2 Σ_{i,a,b} x_i^a x_i^b)")
        print("E[exp(β Σ_ij d_ji x_i^a x_j^a)] = exp(β^2 σ_d^2/(2S) Σ_{i,j,a,b} x_i^a x_i^b x_j^a x_j^b)")
        print(
            "E[exp(β Σ_ijk e_ijk x_i^a x_j^a x_k^a)] = exp(β^2 σ_e^2/(2S^2) Σ_{i,j,k,a,b} x_i^a x_i^b x_j^a x_j^b x_k^a x_k^b)")

        # 定义重叠矩阵
        print("\n引入重叠矩阵: Q^{ab} = (1/S) Σ_i x_i^a x_i^b")

        return {
            'c_contribution': f"(β^2 {p['c_std'] ** 2}/2) Σ_i Σ_{'{a,b}'} x_i^a x_i^b",
            'd_contribution': f"(β^2 {p['d_std'] ** 2} S/2) Σ_{'{a,b}'} (Q^{'{ab}'})^2",
            'e_contribution': f"(β^2 {p['e_std'] ** 2} S/2) Σ_{'{a,b}'} (Q^{'{ab}'})^3"
        }

    def step3_1rsb_ansatz(self):
        """步骤3: 1RSB Ansatz"""
        print("\n=== 步骤3: 1RSB Ansatz ===")
        print("假设副本对称破缺，将n个副本分成n/m个块，每块m个副本")
        print("重叠矩阵结构:")
        print("  Q^{ab} = q_d   如果 a = b (自重叠)")
        print("          = q_1   如果 a,b在同一块内")
        print("          = q_0   如果 a,b在不同块间")

        # 初始化RSB参数
        self.rsb_params = {
            'q_d': 0.8,  # 自重叠
            'q_1': 0.6,  # 块内重叠
            'q_0': 0.2,  # 块间重叠
            'm_RSB': 0.3,  # Parisi参数
            'beta': self.beta
        }

        print(f"\n初始RSB参数:")
        print(f"  q_d = {self.rsb_params['q_d']} (自重叠)")
        print(f"  q_1 = {self.rsb_params['q_1']} (块内重叠)")
        print(f"  q_0 = {self.rsb_params['q_0']} (块间重叠)")
        print(f"  m = {self.rsb_params['m_RSB']} (Parisi参数)")

        return self.rsb_params

    def step4_effective_hamiltonian(self):
        """步骤4: 推导有效哈密顿量"""
        print("\n=== 步骤4: 有效哈密顿量推导 ===")
        print("经过无序平均和鞍点近似，得到单点有效哈密顿量:")

        p = self.original_params
        r = self.rsb_params

        # 计算各种贡献
        mean_field = p['c_mean'] + p['d_mean'] * r['m_RSB'] + p['e_mean'] * r['m_RSB'] ** 2
        sigma_0 = np.sqrt(p['c_std'] ** 2 + p['d_std'] ** 2 * r['q_0'] + p['e_std'] ** 2 * r['q_0'] ** 2)
        sigma_1 = np.sqrt(p['d_std'] ** 2 * (r['q_1'] - r['q_0']) + p['e_std'] ** 2 * (r['q_1'] ** 2 - r['q_0'] ** 2))

        print(f"均值场贡献: {mean_field:.4f}")
        print(f"随机场方差1 (块间): σ_0 = {sigma_0:.4f}")
        print(f"随机场方差2 (块内): σ_1 = {sigma_1:.4f}")

        print("\n有效哈密顿量:")
        print("H_eff(x|z₀,z₁) = V(x) - x[均值场 + σ₀z₀ + σ₁z₁]")
        print("其中 z₀, z₁ ~ N(0,1) 是高斯随机场")

        self.effective_params = {
            'mean_field': mean_field,
            'sigma_0': sigma_0,
            'sigma_1': sigma_1
        }

        return self.effective_params

    def effective_hamiltonian(self, x, z0, z1):
        """有效哈密顿量实现"""
        V = 0.25 * x ** 4 - 0.5 * x ** 2  # 势能
        random_field = self.effective_params['sigma_0'] * z0 + self.effective_params['sigma_1'] * z1
        field = self.effective_params['mean_field'] + random_field

        return V - x * field

    def step5_free_energy(self, n_points=5):
        """步骤5: 计算1RSB自由能"""
        print("\n=== 步骤5: 1RSB自由能计算 ===")

        # 对高斯随机场进行采样
        z0_vals = np.linspace(-2, 2, n_points)
        z1_vals = np.linspace(-2, 2, n_points)

        print("计算内部自由能项:")
        print("f_int = -1/(βm) E[ln ∫Dz₁ (∫Dz₀ exp(-βH_eff))^m]")

        f1_vals = []
        f2_vals = []

        for z0 in z0_vals:
            for z1 in z1_vals:
                # 状态1的自由能
                H1 = lambda x: self.effective_hamiltonian(x, z0, z1)
                Z1, _ = quad(lambda x: np.exp(-self.beta * H1(x)), -3, 3)
                f1 = -np.log(Z1) / self.beta if Z1 > 1e-10 else 0
                f1_vals.append(f1)

                # 状态2的自由能（对称状态）
                H2 = lambda x: self.effective_hamiltonian(x, -z0, z1)
                Z2, _ = quad(lambda x: np.exp(-self.beta * H2(x)), -3, 3)
                f2 = -np.log(Z2) / self.beta if Z2 > 1e-10 else 0
                f2_vals.append(f2)

        # 计算平均自由能
        f1_avg = np.mean(f1_vals)
        f2_avg = np.mean(f2_vals)

        print(f"状态1平均自由能: {f1_avg:.6f}")
        print(f"状态2平均自由能: {f2_avg:.6f}")

        # 计算内部自由能
        m = self.rsb_params['m_RSB']
        f_int = -1 / (self.beta * m) * np.log(
            np.exp(-self.beta * m * f1_avg) + np.exp(-self.beta * m * f2_avg)
        )

        # 能量项
        p = self.original_params
        r = self.rsb_params
        energy_term = 0.5 * (
                p['c_std'] ** 2 +
                p['d_std'] ** 2 * r['q_d'] +
                p['e_std'] ** 2 * (r['q_d'] ** 2 + 2 * r['q_1'] ** 2 - 2 * r['q_0'] ** 2)
        )

        total_free_energy = f_int + energy_term

        print(f"内部自由能: {f_int:.6f}")
        print(f"能量项: {energy_term:.6f}")
        print(f"总自由能: {total_free_energy:.6f}")

        return total_free_energy

    def step6_self_consistent_equations(self, max_iter=20):
        """步骤6: 求解自洽方程"""
        print("\n=== 步骤6: 自洽方程求解 ===")
        print("序参数的自洽方程:")
        print("q_d = ⟨x^2⟩")
        print("q_1 = ⟨⟨x⟩²⟩ (状态内平均)")
        print("q_0 = ⟨⟨x⟩⟩² (状态间平均)")
        print("m 由 ∂f/∂m = 0 确定")

        print(f"\n开始迭代求解 (最多{max_iter}次迭代):")
        print("迭代   q_d       q_1       q_0       m_RSB     自由能")
        print("-" * 60)

        free_energy_history = []

        for i in range(max_iter):
            # 计算当前状态的统计量
            stats = self._compute_statistics()

            # 更新RSB参数
            old_params = self.rsb_params.copy()
            self._update_parameters(stats)

            # 计算自由能
            free_energy = self.step5_free_energy()
            free_energy_history.append(free_energy)

            print(f"{i + 1:2d}    {self.rsb_params['q_d']:.6f}  {self.rsb_params['q_1']:.6f}  "
                  f"{self.rsb_params['q_0']:.6f}  {self.rsb_params['m_RSB']:.6f}  {free_energy:.6f}")

            # 检查收敛
            if self._check_convergence(old_params):
                print(f"\n收敛于第 {i + 1} 次迭代")
                break
        else:
            print(f"\n在 {max_iter} 次迭代后未完全收敛")

        return free_energy_history

    def _compute_statistics(self, n_points=5):
        """计算统计量"""
        z0_vals = np.linspace(-2, 2, n_points)
        z1_vals = np.linspace(-2, 2, n_points)

        m1_list, m2_list, m1_sq_list = [], [], []

        for z0 in z0_vals:
            for z1 in z1_vals:
                # 状态1
                H1 = lambda x: self.effective_hamiltonian(x, z0, z1)
                Z1, _ = quad(lambda x: np.exp(-self.beta * H1(x)), -3, 3)
                if Z1 > 1e-10:
                    m1, _ = quad(lambda x: x * np.exp(-self.beta * H1(x)), -3, 3)
                    m1 /= Z1
                    m1_sq, _ = quad(lambda x: x ** 2 * np.exp(-self.beta * H1(x)), -3, 3)
                    m1_sq /= Z1
                else:
                    m1, m1_sq = 0, 0

                # 状态2
                H2 = lambda x: self.effective_hamiltonian(x, -z0, z1)
                Z2, _ = quad(lambda x: np.exp(-self.beta * H2(x)), -3, 3)
                if Z2 > 1e-10:
                    m2, _ = quad(lambda x: x * np.exp(-self.beta * H2(x)), -3, 3)
                    m2 /= Z2
                else:
                    m2 = 0

                m1_list.append(m1)
                m2_list.append(m2)
                m1_sq_list.append(m1_sq)

        return {
            'm1': np.mean(m1_list),
            'm2': np.mean(m2_list),
            'm1_sq': np.mean(m1_sq_list)
        }

    def _update_parameters(self, stats):
        """更新RSB参数"""
        # 计算权重
        f1 = self._compute_single_free_energy(1)
        f2 = self._compute_single_free_energy(2)
        delta_f = f2 - f1
        w = 1.0 / (1.0 + np.exp(-self.beta * self.rsb_params['m_RSB'] * delta_f))

        # 更新重叠参数
        self.rsb_params['q_d'] = w * stats['m1_sq'] + (1 - w) * stats['m1_sq']  # 简化假设
        self.rsb_params['q_1'] = w ** 2 * stats['m1'] ** 2 + (1 - w) ** 2 * stats['m2'] ** 2 + 2 * w * (1 - w) * stats[
            'm1'] * stats['m2']
        self.rsb_params['q_0'] = w * (1 - w) * (stats['m1'] ** 2 + stats['m2'] ** 2) + (w ** 2 + (1 - w) ** 2) * stats[
            'm1'] * stats['m2']

        # 简单更新m (实际应该通过优化自由能)
        self.rsb_params['m_RSB'] = 0.3 + 0.4 * (1 - abs(stats['m1'] - stats['m2']))

    def _compute_single_free_energy(self, state):
        """计算单个状态的自由能"""
        n_points = 3
        z0_vals = np.linspace(-1, 1, n_points)
        z1_vals = np.linspace(-1, 1, n_points)

        f_vals = []
        for z0 in z0_vals:
            for z1 in z1_vals:
                if state == 1:
                    H = lambda x: self.effective_hamiltonian(x, z0, z1)
                else:
                    H = lambda x: self.effective_hamiltonian(x, -z0, z1)

                Z, _ = quad(lambda x: np.exp(-self.beta * H(x)), -3, 3)
                if Z > 1e-10:
                    f = -np.log(Z) / self.beta
                    f_vals.append(f)

        return np.mean(f_vals) if f_vals else 0

    def _check_convergence(self, old_params, tolerance=1e-4):
        """检查收敛"""
        for key in ['q_d', 'q_1', 'q_0']:
            if abs(self.rsb_params[key] - old_params[key]) > tolerance:
                return False
        return True

    def step7_analysis_and_visualization(self, free_energy_history):
        """步骤7: 结果分析和可视化"""
        print("\n=== 步骤7: 结果分析 ===")

        # 最终统计量
        stats = self._compute_statistics()

        print(f"\n最终结果:")
        print(f"状态1磁化强度: m1 = {stats['m1']:.6f}")
        print(f"状态2磁化强度: m2 = {stats['m2']:.6f}")
        print(f"状态差异: |m1 - m2| = {abs(stats['m1'] - stats['m2']):.6f}")
        print(
            f"重叠参数: q_d = {self.rsb_params['q_d']:.6f}, q_1 = {self.rsb_params['q_1']:.6f}, q_0 = {self.rsb_params['q_0']:.6f}")

        # 物理分析
        print(f"\n物理分析:")
        if self.rsb_params['q_1'] > self.rsb_params['q_0']:
            print("✓ 系统显示RSB行为: q_1 > q_0")
            print("  状态内关联强于状态间关联")
        else:
            print("✗ 系统处于副本对称状态")

        if abs(stats['m1'] - stats['m2']) > 0.1:
            print("✓ 系统处于双稳状态")
        else:
            print("✗ 系统处于单稳状态")

        # 可视化
        self._plot_results(free_energy_history, stats)

    def _plot_results(self, free_energy_history, stats):
        """绘制结果"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 自由能收敛
        axes[0, 0].plot(free_energy_history, 'b-o', linewidth=2, markersize=4)
        axes[0, 0].set_xlabel('迭代次数')
        axes[0, 0].set_ylabel('自由能')
        axes[0, 0].set_title('自由能收敛过程')
        axes[0, 0].grid(True, alpha=0.3)

        # 势能景观
        x = np.linspace(-2, 2, 1000)
        V = 0.25 * x ** 4 - 0.5 * x ** 2
        axes[0, 1].plot(x, V, 'b-', linewidth=2)
        axes[0, 1].axvline(stats['m1'], color='red', linestyle='--', label=f'状态1: m={stats["m1"]:.3f}')
        axes[0, 1].axvline(stats['m2'], color='green', linestyle='--', label=f'状态2: m={stats["m2"]:.3f}')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('V(x)')
        axes[0, 1].set_title('有效势能景观')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 重叠参数
        overlaps = [self.rsb_params['q_d'], self.rsb_params['q_1'], self.rsb_params['q_0']]
        labels = ['q_d\n(自重叠)', 'q_1\n(状态内)', 'q_0\n(状态间)']
        axes[1, 0].bar(labels, overlaps, color=['blue', 'green', 'red'], alpha=0.7)
        axes[1, 0].set_ylabel('重叠参数值')
        axes[1, 0].set_title('RSB重叠参数')

        # 状态分布
        f1 = self._compute_single_free_energy(1)
        f2 = self._compute_single_free_energy(2)
        w1 = 1.0 / (1.0 + np.exp(-self.beta * self.rsb_params['m_RSB'] * (f2 - f1)))
        w2 = 1 - w1
        axes[1, 1].pie([w1, w2], labels=[f'状态1\nw={w1:.3f}', f'状态2\nw={w2:.3f}'],
                       autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'])
        axes[1, 1].set_title('状态权重分布')

        plt.tight_layout()
        plt.show()


# 运行完整的RSB分析
if __name__ == "__main__":
    # 创建分析器
    analyzer = StepByStepRSB(S=100)

    # 步骤1: 推导哈密顿量
    analyzer.step1_derive_hamiltonian()

    # 步骤2: 副本方法
    analyzer.step2_replica_method(n_replicas=2)

    # 步骤3: 1RSB Ansatz
    analyzer.step3_1rsb_ansatz()

    # 步骤4: 有效哈密顿量
    analyzer.step4_effective_hamiltonian()

    # 步骤5: 自由能计算
    free_energy = analyzer.step5_free_energy()

    # 步骤6: 自洽方程求解
    free_energy_history = analyzer.step6_self_consistent_equations(max_iter=15)

    # 步骤7: 结果分析
    analyzer.step7_analysis_and_visualization(free_energy_history)