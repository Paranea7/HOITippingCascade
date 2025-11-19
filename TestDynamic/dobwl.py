import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
import networkx as nx
from tqdm import tqdm
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class BistableCoupledSystem:
    """
    双稳态耦合系统的模拟程序
    考虑多子系统、双稳态特性和平均场近似
    """

    def __init__(self, S1, S2, phi, coupling_strength, c1_type1, c2_type2,
                 intra_coupling=0.1, inter_coupling=0.05, noise_level=0.01):
        """
        初始化双稳态耦合系统

        Parameters:
        S1: int, 类型1子系统数量
        S2: int, 类型2子系统数量
        phi: float, 类型1系统比例
        coupling_strength: float, 总体耦合强度
        c1_type1: float, 类型1系统控制参数
        c2_type2: float, 类型2系统控制参数
        intra_coupling: float, 类型内耦合相对强度
        inter_coupling: float, 类型间耦合相对强度
        noise_level: float, 噪声水平
        """
        self.S1 = S1
        self.S2 = S2
        self.S = S1 + S2
        self.phi = phi
        self.coupling_strength = coupling_strength
        self.c1_type1 = c1_type1
        self.c2_type2 = c2_type2
        self.intra_coupling = intra_coupling
        self.inter_coupling = inter_coupling
        self.noise_level = noise_level

        # 初始化状态
        self.x = np.zeros(self.S)

        # 分配类型
        self.types = np.concatenate([np.ones(S1), np.zeros(S2)]).astype(int)

        # 初始化耦合矩阵
        self._initialize_coupling_matrix()

        # 计算无耦合系统的稳定状态
        self._calculate_uncoupled_states()

    def _initialize_coupling_matrix(self):
        """初始化耦合矩阵，考虑规模效应"""
        self.D = np.zeros((self.S, self.S))

        for i in range(self.S):
            for j in range(self.S):
                if i != j:  # 排除自耦合
                    if self.types[i] == self.types[j]:
                        # 类型内耦合
                        self.D[i, j] = self.coupling_strength * self.intra_coupling / self.S
                    else:
                        # 类型间耦合
                        self.D[i, j] = self.coupling_strength * self.inter_coupling / self.S

    def _calculate_uncoupled_states(self):
        """计算无耦合系统的稳定状态"""
        # 类型1系统的稳定状态
        p = -1
        q = -self.c1_type1
        delta = (q / 2) ** 2 + (p / 3) ** 3

        if delta > 0:
            # 一个实根
            self.x1_stable = [np.cbrt(-q / 2 + np.sqrt(delta)) + np.cbrt(-q / 2 - np.sqrt(delta))]
        elif delta == 0:
            # 三个实根，两个相等
            self.x1_stable = [2 * np.cbrt(-q / 2), -np.cbrt(-q / 2)]
        else:
            # 三个不同实根
            r = np.sqrt(-(p / 3) ** 3)
            theta = np.arccos(-q / (2 * r))
            self.x1_stable = [2 * np.cbrt(r) * np.cos(theta / 3),
                              2 * np.cbrt(r) * np.cos((theta + 2 * np.pi) / 3),
                              2 * np.cbrt(r) * np.cos((theta + 4 * np.pi) / 3)]

        # 类型2系统的稳定状态 (c2=0)
        self.x2_stable = [-1, 0, 1]

        # 识别稳定状态（通过局部稳定性分析）
        self.x1_stable_stable = [x for x in self.x1_stable if -3 * x ** 2 + 1 < 0]
        self.x2_stable_stable = [x for x in self.x2_stable if -3 * x ** 2 + 1 < 0]

        print(f"类型1稳定状态: {[f'{x:.3f}' for x in self.x1_stable_stable]}")
        print(f"类型2稳定状态: {[f'{x:.3f}' for x in self.x2_stable_stable]}")

    def dynamics(self, x, t):
        """系统动力学方程"""
        dxdt = np.zeros_like(x)

        for i in range(self.S):
            # 固有动力学
            intrinsic = -x[i] ** 3 + x[i] + (self.c1_type1 if self.types[i] == 1 else self.c2_type2)

            # 耦合项
            coupling = 0
            for j in range(self.S):
                if i != j:
                    coupling += self.D[i, j] * x[j]

            # 噪声项
            noise = self.noise_level * np.random.normal(0, 1)

            dxdt[i] = intrinsic + coupling + noise

        return dxdt

    def mean_field_dynamics(self, m, t):
        """平均场动力学方程"""
        # 有效耦合强度
        beta_eff = self.coupling_strength * (
                self.phi * (self.S1 - 1) / self.S * self.intra_coupling +
                self.phi * self.S2 / self.S * self.inter_coupling +
                (1 - self.phi) * self.S1 / self.S * self.inter_coupling +
                (1 - self.phi) * (self.S2 - 1) / self.S * self.intra_coupling
        )

        # 平均控制参数
        c_avg = self.phi * self.c1_type1 + (1 - self.phi) * self.c2_type2

        # 平均场动力学
        dmdt = -m ** 3 + m + c_avg + beta_eff * m

        return dmdt

    def bimodal_mean_field_dynamics(self, state_vars, t):
        """
        双峰平均场动力学

        state_vars: [m1_plus, m1_minus, p1_plus, m2_plus, m2_minus, p2_plus]
        """
        m1_plus, m1_minus, p1_plus, m2_plus, m2_minus, p2_plus = state_vars

        # 计算全局平均场
        p1_minus = 1 - p1_plus
        p2_minus = 1 - p2_plus
        m_global = self.phi * (p1_plus * m1_plus + p1_minus * m1_minus) + \
                   (1 - self.phi) * (p2_plus * m2_plus + p2_minus * m2_minus)

        # 类型1子系统的平均场动力学
        dm1_plus_dt = -m1_plus ** 3 + m1_plus + self.c1_type1 + \
                      self.coupling_strength * m_global
        dm1_minus_dt = -m1_minus ** 3 + m1_minus + self.c1_type1 + \
                       self.coupling_strength * m_global

        # 类型2子系统的平均场动力学
        dm2_plus_dt = -m2_plus ** 3 + m2_plus + self.c2_type2 + \
                      self.coupling_strength * m_global
        dm2_minus_dt = -m2_minus ** 3 + m2_minus + self.c2_type2 + \
                       self.coupling_strength * m_global

        # 状态转移动力学（简化模型）
        # 转移率与势垒高度和耦合强度相关
        barrier_height = 0.1  # 势垒高度参数
        transition_rate = 0.01 * np.exp(-barrier_height + self.coupling_strength * m_global)

        dp1_plus_dt = transition_rate * (1 - p1_plus) - transition_rate * p1_plus
        dp2_plus_dt = transition_rate * (1 - p2_plus) - transition_rate * p2_plus

        return [dm1_plus_dt, dm1_minus_dt, dp1_plus_dt,
                dm2_plus_dt, dm2_minus_dt, dp2_plus_dt]

    def simulate(self, T, dt, initial_condition='random'):
        """模拟系统演化"""
        t = np.arange(0, T, dt)

        # 设置初始条件
        if initial_condition == 'random':
            self.x = np.random.uniform(-2, 2, self.S)
        elif initial_condition == 'mixed':
            # 混合初始条件，模拟双稳态
            for i in range(self.S):
                if self.types[i] == 1:
                    self.x[i] = np.random.choice(self.x1_stable_stable)
                else:
                    self.x[i] = np.random.choice(self.x2_stable_stable)

        # 模拟完整系统
        x_sol = odeint(self.dynamics, self.x, t)

        # 计算平均场
        mf_trajectory = np.array([np.mean(x_sol[i]) for i in range(len(t))])

        # 模拟传统平均场
        mf_initial = np.mean(self.x)
        mf_sol = odeint(self.mean_field_dynamics, mf_initial, t)

        # 模拟双峰平均场
        bimodal_initial = [
            max(self.x1_stable_stable), min(self.x1_stable_stable), 0.5,
            max(self.x2_stable_stable), min(self.x2_stable_stable), 0.5
        ]
        bimodal_sol = odeint(self.bimodal_mean_field_dynamics, bimodal_initial, t)

        # 计算双峰平均场的全局平均
        bimodal_global = []
        for i in range(len(t)):
            m1_plus, m1_minus, p1_plus, m2_plus, m2_minus, p2_plus = bimodal_sol[i]
            p1_minus = 1 - p1_plus
            p2_minus = 1 - p2_plus
            m_global = self.phi * (p1_plus * m1_plus + p1_minus * m1_minus) + \
                       (1 - self.phi) * (p2_plus * m2_plus + p2_minus * m2_minus)
            bimodal_global.append(m_global)

        return {
            'time': t,
            'full_system': x_sol,
            'mean_field': mf_sol.flatten(),
            'bimodal_mean_field': np.array(bimodal_global),
            'bimodal_details': bimodal_sol,
            'mf_trajectory': mf_trajectory
        }

    def analyze_bistability(self, results):
        """分析双稳态特性"""
        # 计算最终状态分布
        final_states = results['full_system'][-1]

        # 分类状态
        type1_states = final_states[:self.S1]
        type2_states = final_states[self.S1:]

        # 识别双稳态
        type1_bistable = len(np.unique(np.round(type1_states, 1))) > 1
        type2_bistable = len(np.unique(np.round(type2_states, 1))) > 1

        print(f"类型1系统双稳态: {type1_bistable}")
        print(f"类型2系统双稳态: {type2_bistable}")

        # 计算状态占比
        if len(self.x1_stable_stable) > 1:
            threshold = (max(self.x1_stable_stable) + min(self.x1_stable_stable)) / 2
            type1_high = np.sum(type1_states > threshold) / self.S1
            print(f"类型1高状态占比: {type1_high:.3f}")

        if len(self.x2_stable_stable) > 1:
            threshold = (max(self.x2_stable_stable) + min(self.x2_stable_stable)) / 2
            type2_high = np.sum(type2_states > threshold) / self.S2
            print(f"类型2高状态占比: {type2_high:.3f}")

        return {
            'type1_bistable': type1_bistable,
            'type2_bistable': type2_bistable,
            'type1_high_ratio': type1_high if 'type1_high' in locals() else None,
            'type2_high_ratio': type2_high if 'type2_high' in locals() else None
        }

    def plot_results(self, results, analysis):
        """绘制结果"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. 完整系统演化
        for i in range(min(10, self.S)):  # 只显示前10个子系统
            axes[0, 0].plot(results['time'], results['full_system'][:, i], alpha=0.7)
        axes[0, 0].set_title('完整系统演化')
        axes[0, 0].set_xlabel('时间')
        axes[0, 0].set_ylabel('状态')

        # 2. 平均场比较
        axes[0, 1].plot(results['time'], results['mf_trajectory'], label='完整系统平均', linewidth=2)
        axes[0, 1].plot(results['time'], results['mean_field'], '--', label='传统平均场', linewidth=2)
        axes[0, 1].plot(results['time'], results['bimodal_mean_field'], '--', label='双峰平均场', linewidth=2)
        axes[0, 1].set_title('平均场比较')
        axes[0, 1].set_xlabel('时间')
        axes[0, 1].set_ylabel('平均场')
        axes[0, 1].legend()

        # 3. 最终状态分布
        final_states = results['full_system'][-1]
        axes[0, 2].hist(final_states[:self.S1], alpha=0.7, label='类型1', bins=20)
        axes[0, 2].hist(final_states[self.S1:], alpha=0.7, label='类型2', bins=20)
        axes[0, 2].set_title('最终状态分布')
        axes[0, 2].set_xlabel('状态值')
        axes[0, 2].set_ylabel('频数')
        axes[0, 2].legend()

        # 4. 双峰平均场细节
        axes[1, 0].plot(results['time'], results['bimodal_details'][:, 0], label='m1+')
        axes[1, 0].plot(results['time'], results['bimodal_details'][:, 1], label='m1-')
        axes[1, 0].set_title('类型1双峰状态')
        axes[1, 0].set_xlabel('时间')
        axes[1, 0].set_ylabel('状态值')
        axes[1, 0].legend()

        # 5. 状态占比演化
        axes[1, 1].plot(results['time'], results['bimodal_details'][:, 2], label='p1+')
        axes[1, 1].plot(results['time'], results['bimodal_details'][:, 5], label='p2+')
        axes[1, 1].set_title('高状态占比演化')
        axes[1, 1].set_xlabel('时间')
        axes[1, 1].set_ylabel('占比')
        axes[1, 1].legend()

        # 6. 相图分析
        # 计算势函数
        m_range = np.linspace(-2, 2, 100)
        beta_eff = self.coupling_strength * (
                self.phi * (self.S1 - 1) / self.S * self.intra_coupling +
                self.phi * self.S2 / self.S * self.inter_coupling +
                (1 - self.phi) * self.S1 / self.S * self.inter_coupling +
                (1 - self.phi) * (self.S2 - 1) / self.S * self.intra_coupling
        )
        c_avg = self.phi * self.c1_type1 + (1 - self.phi) * self.c2_type2

        potential = 0.25 * m_range ** 4 - 0.5 * (1 + beta_eff) * m_range ** 2 - c_avg * m_range
        axes[1, 2].plot(m_range, potential)
        axes[1, 2].set_title(f'有效势函数 (β={beta_eff:.3f})')
        axes[1, 2].set_xlabel('平均场 m')
        axes[1, 2].set_ylabel('势函数 V(m)')

        plt.tight_layout()
        plt.show()


def parameter_sweep_analysis():
    """参数扫描分析"""
    phi_values = np.linspace(0.1, 0.9, 5)
    coupling_values = np.linspace(-0.5, 0.5, 10)

    results = []

    for phi in tqdm(phi_values, desc="扫描phi"):
        for coupling in coupling_values:
            # 创建系统
            system = BistableCoupledSystem(
                S1=int(50 * phi), S2=int(50 * (1 - phi)),
                phi=phi, coupling_strength=coupling,
                c1_type1=2 * np.sqrt(3) / 9, c2_type2=0
            )

            # 模拟
            sim_results = system.simulate(T=100, dt=0.1, initial_condition='mixed')

            # 分析
            analysis = system.analyze_bistability(sim_results)

            results.append({
                'phi': phi,
                'coupling': coupling,
                'final_m': np.mean(sim_results['full_system'][-1]),
                'bistable': analysis['type1_bistable'] or analysis['type2_bistable'],
                'type1_high_ratio': analysis['type1_high_ratio'],
                'type2_high_ratio': analysis['type2_high_ratio']
            })

    return results


# 示例使用
if __name__ == "__main__":
    print("=== 双稳态耦合系统模拟 ===")

    # 创建系统实例
    system = BistableCoupledSystem(
        S1=30, S2=20, phi=0.6,
        coupling_strength=0.2,
        c1_type1=2 * np.sqrt(3) / 9, c2_type2=0,
        noise_level=0.02
    )

    # 运行模拟
    print("运行模拟...")
    results = system.simulate(T=50, dt=0.1, initial_condition='mixed')

    # 分析结果
    print("分析双稳态特性...")
    analysis = system.analyze_bistability(results)

    # 绘制结果
    print("绘制结果...")
    system.plot_results(results, analysis)

    # 参数扫描分析
    print("进行参数扫描分析...")
    sweep_results = parameter_sweep_analysis()

    # 绘制参数扫描结果
    phi_vals = [r['phi'] for r in sweep_results]
    coupling_vals = [r['coupling'] for r in sweep_results]
    final_m_vals = [r['final_m'] for r in sweep_results]
    bistable_vals = [r['bistable'] for r in sweep_results]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    scatter = plt.scatter(phi_vals, coupling_vals, c=final_m_vals, cmap='viridis')
    plt.colorbar(scatter, label='最终平均场')
    plt.xlabel('φ (类型1比例)')
    plt.ylabel('耦合强度')
    plt.title('最终状态相图')

    plt.subplot(1, 3, 2)
    plt.scatter(phi_vals, coupling_vals, c=bistable_vals, cmap='coolwarm')
    plt.xlabel('φ (类型1比例)')
    plt.ylabel('耦合强度')
    plt.title('双稳态区域')

    plt.subplot(1, 3, 3)
    # 提取有效的type1_high_ratio值
    valid_ratios = [(p, c, r) for p, c, r in zip(phi_vals, coupling_vals,
                                                 [r['type1_high_ratio'] for r in sweep_results])
                    if r is not None]
    if valid_ratios:
        p_vals, c_vals, r_vals = zip(*valid_ratios)
        scatter = plt.scatter(p_vals, c_vals, c=r_vals, cmap='plasma')
        plt.colorbar(scatter, label='高状态占比')
        plt.xlabel('φ (类型1比例)')
        plt.ylabel('耦合强度')
        plt.title('类型1高状态占比')

    plt.tight_layout()
    plt.show()

    print("模拟完成！")