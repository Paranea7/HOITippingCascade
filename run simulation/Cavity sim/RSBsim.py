#!/usr/bin/env python3
"""
改进版 MultiSystemRSB 程序（与先前设计一致）：

主要改动点（摘要）：
- 统一耦合缩放：d ~ N(mu_d/s, sigma_d/s), e ~ N(mu_e/s^2, sigma_e/s^2)（与最初设计一致）。
- 在子进程中使用确定性的 RNG（np.random.RandomState(seed)）以保证可复现。
- 增加选项 generate_c_in_worker（默认 False）。若为 False，则在主进程生成 c 并传入子进程；若为 True，则子进程内部生成 c。
- 增加 sparse_e_prob 选项（默认 0.0），在大 S 时可用来稀疏化三体张量以节省内存。
- 修正并行收集顺序、绘图小错误、并提供参数扫描与示例运行。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

class MultiSystemRSB:
    def __init__(self, n_systems=100, S=50,
                 params=None, rsb_params=None,
                 generate_c_in_worker=False, phi_0=0.0, fixed_c_value=None,
                 sparse_e_prob=0.0, seed_base=None):
        """
        参数说明（新增/修改）：
        - generate_c_in_worker: 若 True，子进程内部生成 c_i；若 False（默认），在主进程生成并传入 worker。
        - phi_0: 当 generate_c_in_worker=True 时生效，控制非零 c_i 的数量（比例或绝对数，参照原脚本语义）。
        - fixed_c_value: 若不为 None，非零 c_i 元素设为该值，否则使用 params['c_mean']。
        - sparse_e_prob: 在生成 e_ijk 时，每个三体项以该概率被保留（其余设为0），可节省内存。0.0 表示不稀疏。
        - seed_base: 可选基准随机种子，用于生成子进程种子（若 None 则使用随机 base）。
        """
        self.n_systems = n_systems
        self.S = S

        # 参数字典（若外部提供则使用）
        if params is None:
            params = {
                'c_mean': 0.0, 'c_std': 0.5,
                'd_mean': 0.1, 'd_std': 0.3,
                'e_mean': -0.05, 'e_std': 0.2
            }
        if rsb_params is None:
            rsb_params = {
                'q_d': 0.8, 'q_1': 0.6, 'q_0': 0.2,
                'm_RSB': 0.5, 'beta': 2.0
            }

        self.params = params
        self.rsb_params = rsb_params

        # 行为控制参数
        self.generate_c_in_worker = generate_c_in_worker
        self.phi_0 = phi_0
        self.fixed_c_value = fixed_c_value
        self.sparse_e_prob = float(sparse_e_prob)
        self.seed_base = seed_base

        # 存储
        self.systems_data = []
        self.final_states = []

    # ----------------- 力场 / 能量函数（主进程可用） -----------------
    def effective_hamiltonian(self, x, z0, z1):
        p = self.params
        r = self.rsb_params

        mean_field = p['c_mean'] + p['d_mean'] * r['m_RSB'] + p['e_mean'] * r['m_RSB'] ** 2
        sigma_0 = np.sqrt(p['c_std'] ** 2 + p['d_std'] ** 2 * r['q_0'] + p['e_std'] ** 2 * r['q_0'] ** 2)
        sigma_1 = np.sqrt(p['d_std'] ** 2 * (r['q_1'] - r['q_0']) + p['e_std'] ** 2 * (r['q_1'] ** 2 - r['q_0'] ** 2))

        random_field = sigma_0 * z0 + sigma_1 * z1
        V = 0.25 * x ** 4 - 0.5 * x ** 2

        return V - x * (mean_field + random_field)

    # ----------------- 单系统静态方法（可在子进程中运行） -----------------
    @staticmethod
    def _generate_c_for_worker(S, phi_0, fixed_c_value, params, rng):
        """根据 phi_0 语义生成 c_i 向量（在 worker 内部调用）"""
        if phi_0 is None:
            count = 0
        else:
            phi_val = float(phi_0)
            if phi_val <= 0.0:
                count = 0
            elif 0.0 < phi_val <= 1.0:
                count = int(round(phi_val * S))
            else:
                count = int(round(phi_val))
                if count > S:
                    count = S
        c = np.zeros(S, dtype=float)
        if count > 0:
            idx = rng.choice(S, size=count, replace=False)
            c[idx] = float(fixed_c_value) if fixed_c_value is not None else float(params.get('c_mean', 0.0))
        return c

    @staticmethod
    def simulate_single_system_static(system_id, S, params, generate_c_in_worker,
                                      phi_0, fixed_c_value, sparse_e_prob,
                                      n_steps=1000, dt=0.01, seed=None, c_from_main=None):
        """
        子进程运行单系统模拟（确定性 RNG）。
        - seed: 必须传入以保证每个子进程生成不同而可复现的随机数流
        - c_from_main: 如果主进程已生成 c，则传入；否则为 None，子进程按 generate_c_in_worker 生成
        返回包含 trajectory、final_state、c、d、e、id 的字典。
        """
        rng = np.random.RandomState(seed)

        # 初始值 x_init：默认统一常数（保留修改点）
        x = np.full(S, -0.6, dtype=float)  # 使用你之前脚本中 -0.6 的初始值（MOD）

        # c 生成：如果主进程已经传入则使用，否则按 phi_0 在子进程内生成
        if c_from_main is not None:
            c = np.array(c_from_main, dtype=float)
        elif generate_c_in_worker:
            c = MultiSystemRSB._generate_c_for_worker(S, phi_0, fixed_c_value, params, rng)
        else:
            # 如果既没有主进程传入也不在 worker 中生成，则使用零向量或参数均值
            c = rng.normal(params.get('c_mean', 0.0), params.get('c_std', 0.0), S)

        # 耦合矩阵与张量 —— 统一缩放为：d ~ N(mu_d/s, sigma_d/s), e ~ N(mu_e/s^2, sigma_e/s^2)
        mu_d = params.get('d_mean', 0.0)
        sigma_d = params.get('d_std', 0.0)
        mu_e = params.get('e_mean', 0.0)
        sigma_e = params.get('e_std', 0.0)

        d = rng.normal(loc=mu_d / S, scale=sigma_d / S, size=(S, S))
        np.fill_diagonal(d, 0.0)

        # d_ji 在这里我们直接使用 d 的转置或一个相关矩阵，如需相关性可在 params 中添加 rho_d
        rho_d = params.get('rho_d', 1.0)
        noise = rng.normal(loc=mu_d / S, scale=sigma_d / S, size=(S, S))
        d_ji = rho_d * d + np.sqrt(max(0.0, 1.0 - rho_d ** 2)) * noise
        np.fill_diagonal(d_ji, 0.0)

        # 生成 e：可选择稀疏化以节省内存
        if sparse_e_prob <= 0.0:
            e = rng.normal(loc=mu_e / (S ** 2), scale=sigma_e / (S ** 2), size=(S, S, S))
        else:
            # 逐元素按概率保留，未保留的位置为0
            e = np.zeros((S, S, S), dtype=float)
            mask = rng.uniform(0.0, 1.0, size=(S, S, S)) < sparse_e_prob
            # 只在 mask True 的位置采样
            nnz = np.count_nonzero(mask)
            if nnz > 0:
                e_vals = rng.normal(loc=mu_e / (S ** 2), scale=sigma_e / (S ** 2), size=nnz)
                e[mask] = e_vals

        # 强制 e[:, j, k] 在 j,k 上对称：e[i, j, k] == e[i, k, j]
        # 仅对 j<k 做对称化以避免不必要重复
        for i in range(S):
            for j in range(S):
                for k in range(j + 1, S):
                    val = 0.5 * (e[i, j, k] + e[i, k, j])
                    e[i, j, k] = val
                    e[i, k, j] = val
            e[i, i, i] = 0.0

        # 时间演化（显式 Euler 或 RK4 可选；这里使用显式 Euler 步进，跟你先前实现一致）
        trajectory = np.zeros((n_steps, S), dtype=float)
        trajectory[0] = x.copy()

        for t in range(1, n_steps):
            # 计算线性耦合 d_ji @ x（注意 shape）
            linear_coupling = d_ji.dot(x)
            # 非线性耦合：coupling_nonlinear[i] = sum_{j,k} e[j,k,i] * x[j] * x[k]
            # 使用外积与 tensordot 快速求和
            X = np.outer(x, x)  # (S,S)
            coupling_nonlinear = np.tensordot(X, e, axes=([0, 1], [0, 1]))  # (S,)
            dxdt = -x ** 3 + x + c + linear_coupling + coupling_nonlinear
            x = x + dt * dxdt
            trajectory[t] = x.copy()

        return {
            'id': system_id,
            'trajectory': trajectory,
            'final_state': x.copy(),
            'c': c,
            'd': d,
            'd_ji': d_ji,
            'e': e
        }

    # ----------------- 并行管理与模拟入口 -----------------
    def simulate_all_systems(self, n_steps=1000, dt=0.01, max_workers=None, seeds=None):
        """并行模拟所有系统；若 generate_c_in_worker=False，则在主进程生成 c 并传给 worker。"""
        print("=== 开始并行模拟所有系统 ===")
        self.systems_data = []
        self.final_states = []

        if max_workers is None:
            max_workers = max(1, (os.cpu_count() or 1) - 1)

        # 准备子进程用的 seeds（确定性）
        if seeds is None:
            if self.seed_base is None:
                base = np.random.randint(0, 2 ** 31 - 1)
            else:
                base = int(self.seed_base)
            seeds = [int(base + i + 1) for i in range(self.n_systems)]
        else:
            if len(seeds) < self.n_systems:
                raise ValueError("seeds 列表长度必须 >= n_systems")

        # 如果主进程负责生成 c，则创建 c_list 并传给每个 worker
        c_list = None
        if not self.generate_c_in_worker:
            c_list = []
            rng_main = np.random.RandomState(seeds[0] + 12345)  # 主进程生成 c 的 RNG（确定性偏移）
            for i in range(self.n_systems):
                c_vec = self._generate_c_for_worker(self.S, self.phi_0, self.fixed_c_value, self.params, rng_main)
                c_list.append(c_vec)

        futures = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i in range(self.n_systems):
                seed_i = seeds[i]
                c_for_job = c_list[i] if c_list is not None else None
                futures.append(
                    executor.submit(
                        MultiSystemRSB.simulate_single_system_static,
                        i, self.S, self.params, self.generate_c_in_worker,
                        self.phi_0, self.fixed_c_value, self.sparse_e_prob,
                        n_steps, dt, seed_i, c_for_job
                    )
                )

            for fut in as_completed(futures):
                res = fut.result()
                self.systems_data.append(res)
                self.final_states.append(res['final_state'])

        # 按 id 排序以恢复原有顺序
        self.systems_data.sort(key=lambda x: x['id'])
        self.final_states = np.array([s['final_state'] for s in self.systems_data])

        print("所有系统模拟完成（并行）")
        return self.systems_data

    # ----------------- RSB 分析相关 -----------------
    def analyze_rsb_distribution(self):
        print("\n=== RSB分布分析 ===")
        magnetizations = np.mean(self.final_states, axis=1)
        overlap_distribution = self._compute_overlap_distribution()
        rsb_estimates = self._estimate_rsb_parameters(magnetizations, overlap_distribution)
        return magnetizations, overlap_distribution, rsb_estimates

    def _compute_overlap_distribution(self, n_pairs=1000):
        print("计算重叠分布...")
        overlaps = []
        for _ in range(n_pairs):
            i, j = np.random.choice(self.n_systems, 2, replace=False)
            q = np.dot(self.final_states[i], self.final_states[j]) / self.S
            overlaps.append(q)
        return np.array(overlaps)

    def _estimate_rsb_parameters(self, magnetizations, overlaps):
        print("估计RSB参数...")
        q_d_estimate = np.mean([np.dot(state, state) / self.S for state in self.final_states])

        hist, bin_edges = np.histogram(overlaps, bins=50, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        try:
            popt, _ = curve_fit(self._double_gaussian, bin_centers, hist,
                                p0=[0.3, 0.6, 0.1, 0.2, 0.3, 0.1], maxfev=20000)
            q_1_estimate, q_0_estimate = popt[1], popt[4]
        except Exception:
            q_1_estimate = np.percentile(overlaps, 75)
            q_0_estimate = np.percentile(overlaps, 25)

        positive_m = np.sum(magnetizations > 0) / self.n_systems
        negative_m = np.sum(magnetizations < 0) / self.n_systems
        m_RSB_estimate = min(positive_m, negative_m) * 2

        return {
            'q_d': q_d_estimate,
            'q_1': q_1_estimate,
            'q_0': q_0_estimate,
            'm_RSB': m_RSB_estimate,
            'weight_positive': positive_m,
            'weight_negative': negative_m
        }

    @staticmethod
    def _double_gaussian(x, a1, mu1, sigma1, a2, mu2, sigma2):
        return (a1 * np.exp(-(x - mu1) ** 2 / (2 * sigma1 ** 2)) +
                a2 * np.exp(-(x - mu2) ** 2 / (2 * sigma2 ** 2)))

    def compare_with_rsb_prediction(self, rsb_estimates):
        print("\n=== 与RSB预测比较 ===")
        return self._rsb_predicted_distribution(rsb_estimates)

    def _rsb_predicted_distribution(self, rsb_estimates):
        n_points = 1000
        x = np.linspace(-2, 2, n_points)
        m1 = rsb_estimates.get('m1_estimated', 0.8)
        m2 = rsb_estimates.get('m2_estimated', -0.8)
        w1 = rsb_estimates['weight_positive']
        w2 = rsb_estimates['weight_negative']
        sigma = 0.2
        pdf = (w1 * np.exp(-(x - m1) ** 2 / (2 * sigma ** 2)) +
               w2 * np.exp(-(x - m2) ** 2 / (2 * sigma ** 2)))
        pdf = pdf / np.sum(pdf)
        return x, pdf

    # ----------------- 可视化 -----------------
    def plot_comprehensive_results(self, magnetizations, overlaps, rsb_estimates):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        axes[0, 0].hist(magnetizations, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_xlabel('平均磁化强度')
        axes[0, 0].set_ylabel('概率密度')
        axes[0, 0].set_title(f'{self.n_systems}个系统的磁化强度分布\n均值: {np.mean(magnetizations):.3f}')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].hist(overlaps, bins=30, density=True, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].axvline(rsb_estimates['q_1'], color='red', linestyle='--', label=f'q_1 = {rsb_estimates["q_1"]:.3f}')
        axes[0, 1].axvline(rsb_estimates['q_0'], color='orange', linestyle='--', label=f'q_0 = {rsb_estimates["q_0"]:.3f}')
        axes[0, 1].set_xlabel('重叠 q')
        axes[0, 1].set_ylabel('P(q)')
        axes[0, 1].set_title('重叠分布 P(q)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        weights = [rsb_estimates['weight_positive'], rsb_estimates['weight_negative']]
        labels = [f'正磁化: {weights[0]:.3f}', f'负磁化: {weights[1]:.3f}']
        axes[0, 2].pie(weights, labels=labels, autopct='%1.1f%%', colors=['lightcoral', 'lightblue'])
        axes[0, 2].set_title('状态权重分布')

        if len(self.systems_data) > 0:
            system_idx = 0
            trajectory = self.systems_data[system_idx]['trajectory']
            time = np.arange(trajectory.shape[0])
            for i in range(min(10, self.S)):
                axes[1, 0].plot(time, trajectory[:, i], alpha=0.7, linewidth=1)
            axes[1, 0].set_xlabel('时间步')
            axes[1, 0].set_ylabel('x_i(t)')
            axes[1, 0].set_title('典型系统的时间演化')
            axes[1, 0].grid(True, alpha=0.3)

        if len(self.final_states) > 1:
            corr_matrix = np.corrcoef(self.final_states[:20])
            im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            axes[1, 1].set_xlabel('系统索引')
            axes[1, 1].set_ylabel('系统索引')
            axes[1, 1].set_title('系统间相关性矩阵')
            plt.colorbar(im, ax=axes[1, 1])

        rsb_keys = ['q_d', 'q_1', 'q_0', 'm_RSB']
        rsb_values = [rsb_estimates[k] for k in rsb_keys]
        axes[1, 2].bar(rsb_keys, rsb_values, color=['blue', 'green', 'red', 'purple'], alpha=0.7)
        axes[1, 2].set_ylabel('参数值')
        axes[1, 2].set_title('估计的RSB参数')
        axes[1, 2].grid(True, alpha=0.3)
        for i, v in enumerate(rsb_values):
            axes[1, 2].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()
        return fig

    # ----------------- 完整运行流程 -----------------
    def run_complete_analysis(self, n_steps=1000, dt=0.01, max_workers=None, seeds=None):
        print("=" * 60)
        print(f"       {self.n_systems}个系统的RSB分布分析（并行）")
        print("=" * 60)

        self.simulate_all_systems(n_steps=n_steps, dt=dt, max_workers=max_workers, seeds=seeds)
        magnetizations, overlaps, rsb_estimates = self.analyze_rsb_distribution()
        predicted_distribution = self.compare_with_rsb_prediction(rsb_estimates)

        print(f"\n=== 分析结果 ===")
        print(f"系统数量: {self.n_systems}")
        print(f"每个系统大小: {self.S}")
        print(f"平均磁化强度: {np.mean(magnetizations):.4f} ± {np.std(magnetizations):.4f}")
        print(f"正磁化状态比例: {rsb_estimates['weight_positive']:.4f}")
        print(f"负磁化状态比例: {rsb_estimates['weight_negative']:.4f}")
        print(f"估计的RSB参数:")
        print(f"  q_d = {rsb_estimates['q_d']:.4f} (自重叠)")
        print(f"  q_1 = {rsb_estimates['q_1']:.4f} (状态内重叠)")
        print(f"  q_0 = {rsb_estimates['q_0']:.4f} (状态间重叠)")
        print(f"  m_RSB = {rsb_estimates['m_RSB']:.4f} (Parisi参数)")

        self.plot_comprehensive_results(magnetizations, overlaps, rsb_estimates)

        return {
            'magnetizations': magnetizations,
            'overlaps': overlaps,
            'rsb_estimates': rsb_estimates,
            'predicted_distribution': predicted_distribution
        }

# ----------------- 参数扫描保持不变（修正小错误） -----------------
class ParameterSweepRSB:
    def __init__(self):
        self.results = {}

    def sweep_parameter(self, param_name, param_range, n_systems=50, max_workers=None):
        print(f"扫描参数 {param_name}...")
        results = {}
        for param_value in param_range:
            print(f"  {param_name} = {param_value:.3f}")
            analyzer = MultiSystemRSB(n_systems=n_systems, S=30)
            if param_name in analyzer.params:
                analyzer.params[param_name] = param_value
            elif param_name == 'beta':
                analyzer.rsb_params['beta'] = param_value

            result = analyzer.run_complete_analysis(n_steps=800, dt=0.01, max_workers=max_workers)
            results[param_value] = result

        self.results[param_name] = results
        return results

    def plot_parameter_sweep(self, param_name):
        if param_name not in self.results:
            print(f"没有 {param_name} 的扫描结果")
            return
        results = self.results[param_name]
        param_values = list(results.keys())
        q_diffs = [results[p]['rsb_estimates']['q_1'] - results[p]['rsb_estimates']['q_0'] for p in param_values]
        weights_pos = [results[p]['rsb_estimates']['weight_positive'] for p in param_values]
        m_RSB_vals = [results[p]['rsb_estimates']['m_RSB'] for p in param_values]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].plot(param_values, q_diffs, 'bo-', linewidth=2, markersize=6)
        axes[0].set_xlabel(param_name)
        axes[0].set_ylabel('q_1 - q_0')
        axes[0].set_title('RSB强度')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(param_values, weights_pos, 'ro-', linewidth=2, markersize=6, label='正状态')
        axes[1].plot(param_values, 1 - np.array(weights_pos), 'go-', linewidth=2, markersize=6, label='负状态')
        axes[1].set_xlabel(param_name)
        axes[1].set_ylabel('状态权重')
        axes[1].set_title('状态权重变化')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(param_values, m_RSB_vals, 'purpleo-', linewidth=2, markersize=6)
        axes[2].set_xlabel(param_name)
        axes[2].set_ylabel('m_RSB')
        axes[2].set_title('Parisi参数')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

# ----------------- 运行示例 -----------------
if __name__ == "__main__":
    print("主要分析（并行）：100个系统的RSB分布")
    analyzer = MultiSystemRSB(n_systems=100, S=50,
                              generate_c_in_worker=False,  # 主进程生成 c 并传入 worker（可改为 True）
                              phi_0=0.16, fixed_c_value=2.0 * np.sqrt(3.0) / 9.0,
                              sparse_e_prob=0.0,
                              seed_base=123456)
    # 可选指定 max_workers，例如 6；默认使用 CPU_count()-1
    results = analyzer.run_complete_analysis(n_steps=1000, dt=0.01, max_workers=None)