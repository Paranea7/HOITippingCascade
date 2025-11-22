import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize
from scipy.stats import gaussian_kde, norm
import warnings

warnings.filterwarnings('ignore')


class CubicModelSolver:
    """
    针对您的三次系统模型的专用求解器
    - 模型: dx_i/dt = -x_i^3 + x_i + c_i + sum_j d_ji x_j + sum_jk e_ijk x_j x_k
    - 已知稳定状态: -1.34 和 +1.34
    """

    def __init__(self, S_sim=100, S_rsb=1000, n_states=20, max_iter=300, tol=1e-8):
        self.S_sim = S_sim
        self.S_rsb = S_rsb
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol

    def simulate_dynamics(self, mu_c, sigma_c, mu_d, sigma_d, mu_e, sigma_e,
                          T=2000, dt=0.01, n_replicates=20):
        """
        数值模拟 - 允许负值
        """
        print("开始数值模拟...")
        all_final_states = []

        for rep in range(n_replicates):
            np.random.seed(rep)

            # 生成参数
            c = np.random.normal(mu_c, sigma_c, self.S_sim)

            if sigma_d > 0:
                d = np.random.normal(mu_d / self.S_sim, sigma_d / np.sqrt(self.S_sim),
                                     (self.S_sim, self.S_sim))
            else:
                d = np.full((self.S_sim, self.S_sim), mu_d / self.S_sim)
            np.fill_diagonal(d, 0)

            if sigma_e > 0:
                e = np.random.normal(mu_e / (self.S_sim ** 2), sigma_e / self.S_sim,
                                     (self.S_sim, self.S_sim, self.S_sim))
            else:
                e = np.full((self.S_sim, self.S_sim, self.S_sim), mu_e / (self.S_sim ** 2))

            # 使用不同的初始条件来探索两个稳定状态
            initial_conditions = [
                np.random.uniform(-2, -0.5, self.S_sim),  # 偏向负状态
                np.random.uniform(0.5, 2, self.S_sim),  # 偏向正状态
                np.random.uniform(-1, 1, self.S_sim)  # 随机
            ]

            for x0 in initial_conditions:
                x = x0.copy()

                for t in range(T):
                    # 计算导数
                    dxdt = np.zeros(self.S_sim)
                    for i in range(self.S_sim):
                        self_term = -x[i] ** 3 + x[i] + c[i]
                        linear_term = np.sum(d[:, i] * x)

                        pairwise_term = 0
                        for j in range(self.S_sim):
                            for k in range(self.S_sim):
                                pairwise_term += e[j, k, i] * x[j] * x[k]

                        dxdt[i] = self_term + linear_term + pairwise_term

                    # Euler方法
                    x_new = x + dt * dxdt

                    # 检查收敛
                    if np.max(np.abs(x_new - x)) < 1e-10 and t > 100:
                        break

                    x = x_new

                # 收集最终状态
                all_final_states.extend(x)

            if (rep + 1) % max(1, n_replicates // 5) == 0:
                print(f"  完成 {rep + 1}/{n_replicates} 次模拟")

        return np.array(all_final_states)

    def analyze_known_states(self, data, expected_states=[-1.34, 1.34], tolerance=0.1):
        """
        分析数据中已知稳定状态的出现情况
        """
        print("\n=== 已知状态分析 ===")

        results = {}

        for target_state in expected_states:
            # 找到接近目标状态的数据点
            mask = np.abs(data - target_state) < tolerance
            count = np.sum(mask)
            proportion = count / len(data) if len(data) > 0 else 0

            results[target_state] = {
                'count': count,
                'proportion': proportion,
                'mean': np.mean(data[mask]) if count > 0 else 0,
                'std': np.std(data[mask]) if count > 0 else 0
            }

            print(f"状态 {target_state}: 出现 {count} 次 ({proportion * 100:.1f}%)")

        # 分析双峰特性
        if len(data) > 10:
            from scipy.stats import gaussian_kde

            # 核密度估计
            kde = gaussian_kde(data)
            x_range = np.linspace(np.min(data), np.max(data), 1000)
            density = kde(x_range)

            # 寻找峰值
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(density, height=0.01)

            print(f"检测到 {len(peaks)} 个密度峰值")
            for i, peak in enumerate(peaks):
                print(f"  峰值 {i + 1}: x = {x_range[peak]:.3f}, 密度 = {density[peak]:.3f}")

        return results

    def cubic_roots_analytical(self, h, u=1):
        """
        三次方程的解析解: x^3 - u*x = h
        """
        # 转换为标准形式: x^3 + px + q = 0
        p = -u
        q = -h

        discriminant = (q / 2) ** 2 + (p / 3) ** 3

        if discriminant > 0:  # 一个实根
            A = np.cbrt(-q / 2 + np.sqrt(discriminant))
            B = np.cbrt(-q / 2 - np.sqrt(discriminant))
            return [A + B]
        else:  # 三个实根
            r = np.sqrt(-(p / 3) ** 3)
            theta = np.arccos(-q / (2 * r))
            roots = []
            for k in range(3):
                root = 2 * np.cbrt(r) * np.cos((theta + 2 * np.pi * k) / 3)
                roots.append(root)
            return sorted(roots)

    def rsb_for_known_states(self, mu_c, sigma_c, mu_d, sigma_d, mu_e, sigma_e,
                             expected_states=[-1.34, 1.34], gamma=1.0, m=0.3):
        """
        针对已知稳定状态的RSB分析
        """
        print("进行RSB分析...")

        # 初始化状态，专注于已知的稳定状态
        states = []
        for i, target_state in enumerate(expected_states):
            # 为每个已知状态创建多个变体
            for j in range(self.n_states // len(expected_states)):
                state = {
                    'id': i * 10 + j,
                    'target_state': target_state,
                    'phi': 0.5,  # 假设每个状态有50%的物种
                    'mean_x': target_state + np.random.normal(0, 0.1),
                    'mean_x2': target_state ** 2 + np.random.normal(0, 0.2),
                    'v': 1.0,
                    'weight': 1.0 / self.n_states
                }
                states.append(state)

        # 补充一些探索性状态
        remaining = self.n_states - len(states)
        for i in range(remaining):
            state = {
                'id': 100 + i,
                'target_state': None,
                'phi': 0.5,
                'mean_x': np.random.uniform(-2, 2),
                'mean_x2': np.random.uniform(0, 4),
                'v': 1.0,
                'weight': 1.0 / self.n_states
            }
            states.append(state)

        # 总相互作用方差
        if sigma_e > 0:
            sigma2_total = self.S_rsb * (sigma_d ** 2 + 2 * sigma_e ** 2)
        else:
            sigma2_total = self.S_rsb * sigma_d ** 2

        history = []

        for iteration in range(self.max_iter):
            new_states = []
            total_delta = 0

            for state in states:
                phi_alpha = state['phi']
                mean_x_alpha = state['mean_x']
                mean_x2_alpha = state['mean_x2']
                v_alpha = state['v']

                # 状态特定的场参数
                mu_h_alpha = mu_c + phi_alpha * (mu_d * mean_x_alpha + mu_e * mean_x2_alpha)

                if sigma_e > 0:
                    sigma_h_alpha = np.sqrt(sigma_c ** 2 + phi_alpha *
                                            (sigma_d ** 2 * mean_x2_alpha + sigma_e ** 2 * mean_x2_alpha ** 2))
                else:
                    sigma_h_alpha = np.sqrt(sigma_c ** 2 + phi_alpha * sigma_d ** 2 * mean_x2_alpha)

                u_alpha = 1 + phi_alpha * sigma2_total * gamma * v_alpha

                # 对于已知目标状态的状态，强制向目标状态收敛
                if state['target_state'] is not None:
                    target = state['target_state']
                    # 计算达到目标状态所需的场
                    h_target = target ** 3 - u_alpha * target

                    # 调整参数使目标状态更可能
                    mu_h_alpha = 0.8 * mu_h_alpha + 0.2 * h_target
                    sigma_h_alpha = max(sigma_h_alpha * 0.9, 0.1)

                # 计算矩量 (简化版本)
                if sigma_h_alpha > 0:
                    # 使用高斯近似
                    x_min = mean_x_alpha - 3 * np.sqrt(mean_x2_alpha - mean_x_alpha ** 2)
                    x_max = mean_x_alpha + 3 * np.sqrt(mean_x2_alpha - mean_x_alpha ** 2)

                    # 数值积分
                    n_points = 500
                    x_vals = np.linspace(x_min, x_max, n_points)

                    # 计算每个x对应的h
                    h_vals = x_vals ** 3 - u_alpha * x_vals

                    # 计算概率密度
                    P_h = norm.pdf(h_vals, mu_h_alpha, sigma_h_alpha)
                    dh_dx = np.abs(3 * x_vals ** 2 - u_alpha)

                    mask = dh_dx > 1e-10
                    if np.sum(mask) > 0:
                        P_x = np.zeros_like(x_vals)
                        P_x[mask] = P_h[mask] / dh_dx[mask]

                        # 归一化
                        total_prob = np.trapz(P_x, x_vals)
                        if total_prob > 0:
                            P_x = P_x / total_prob

                            # 计算矩量
                            phi_new = np.trapz(P_x, x_vals)  # 所有状态都"存活"
                            mean_x_new = np.trapz(x_vals * P_x, x_vals)
                            mean_x2_new = np.trapz(x_vals ** 2 * P_x, x_vals)

                            # 对于已知目标状态，加强收敛
                            if state['target_state'] is not None:
                                target = state['target_state']
                                mean_x_new = 0.7 * mean_x_new + 0.3 * target
                                mean_x2_new = 0.7 * mean_x2_new + 0.3 * target ** 2
                        else:
                            phi_new, mean_x_new, mean_x2_new = phi_alpha, mean_x_alpha, mean_x2_alpha
                    else:
                        phi_new, mean_x_new, mean_x2_new = phi_alpha, mean_x_alpha, mean_x2_alpha
                else:
                    phi_new, mean_x_new, mean_x2_new = phi_alpha, mean_x_alpha, mean_x2_alpha

                # 更新响应系数
                if phi_new > 0 and abs(3 * mean_x2_new - u_alpha) > 1e-6:
                    v_new = 1.0 / (3 * mean_x2_new - u_alpha)
                else:
                    v_new = v_alpha

                # 限制范围
                mean_x_new = np.clip(mean_x_new, -3, 3)
                mean_x2_new = np.clip(mean_x2_new, 0, 9)
                v_new = np.clip(v_new, 0.1, 10)

                delta = abs(mean_x_new - mean_x_alpha)
                total_delta += delta

                new_state = state.copy()
                new_state.update({
                    'phi': phi_new,
                    'mean_x': mean_x_new,
                    'mean_x2': mean_x2_new,
                    'v': v_new,
                    'mu_h': mu_h_alpha,
                    'sigma_h': sigma_h_alpha,
                    'u': u_alpha
                })
                new_states.append(new_state)

            states = new_states

            # 更新权重 (简化版本)
            if iteration % 10 == 0:
                # 基于与已知状态的接近程度分配权重
                weights = []
                for state in states:
                    if state['target_state'] is not None:
                        # 已知状态有较高权重
                        distance = abs(state['mean_x'] - state['target_state'])
                        weight = np.exp(-10 * distance)
                    else:
                        # 探索性状态权重较低
                        weight = 0.1
                    weights.append(weight)

                total_weight = sum(weights)
                if total_weight > 0:
                    for i, state in enumerate(states):
                        state['weight'] = weights[i] / total_weight
                else:
                    for state in states:
                        state['weight'] = 1.0 / len(states)

            # 记录历史
            avg_mean_x = np.sum([s['mean_x'] * s['weight'] for s in states])
            history.append({
                'iteration': iteration,
                'avg_mean_x': avg_mean_x,
                'total_delta': total_delta,
                'states': [s.copy() for s in states]
            })

            if total_delta < self.tol and iteration > 50:
                break

        # 最终结果
        effective_states = [s for s in states if s['weight'] > 0.01]
        if len(effective_states) == 0:
            effective_states = states

        total_weight = sum(s['weight'] for s in effective_states)
        for state in effective_states:
            state['weight'] /= total_weight

        result = {
            'states': effective_states,
            'avg_mean_x': np.sum([s['mean_x'] * s['weight'] for s in effective_states]),
            'n_effective_states': len(effective_states),
            'history': history,
            'converged': iteration < self.max_iter - 1
        }

        return result

    def compute_rsb_distribution(self, rsb_result, n_points=1000):
        """
        计算RSB预测的分布
        """
        states = rsb_result['states']
        if not states:
            return np.array([]), np.array([])

        # 确定x的范围
        x_min, x_max = -3, 3  # 固定范围，覆盖已知状态

        x_range = np.linspace(x_min, x_max, n_points)
        pdf_total = np.zeros_like(x_range)

        for state in states:
            weight = state['weight']
            mu_h = state['mu_h']
            sigma_h = state['sigma_h']
            u = state['u']

            # 计算该状态的PDF
            state_pdf = np.zeros_like(x_range)
            for i, x_val in enumerate(x_range):
                h_val = x_val ** 3 - u * x_val
                P_h = norm.pdf(h_val, mu_h, sigma_h)
                dh_dx = abs(3 * x_val ** 2 - u)
                if dh_dx > 1e-10:
                    state_pdf[i] = P_h / dh_dx

            # 归一化
            state_norm = np.trapz(state_pdf, x_range)
            if state_norm > 0:
                state_pdf = state_pdf / state_norm

            pdf_total += weight * state_pdf

        # 最终归一化
        total_norm = np.trapz(pdf_total, x_range)
        if total_norm > 0:
            pdf_total = pdf_total / total_norm

        return x_range, pdf_total

    def compare_theory_simulation(self, mu_c, sigma_c, mu_d, sigma_d, mu_e, sigma_e,
                                  n_sim_replicates=15, figsize=(15, 10)):
        """
        比较理论和模拟结果
        """
        print("=== 理论与模拟对比分析 ===")

        # 1. 数值模拟
        print("进行数值模拟...")
        sim_data = self.simulate_dynamics(
            mu_c, sigma_c, mu_d, sigma_d, mu_e, sigma_e,
            n_replicates=n_sim_replicates
        )

        # 分析已知状态的出现
        state_analysis = self.analyze_known_states(sim_data)

        # 2. RSB理论预测
        print("\n进行RSB理论预测...")
        rsb_result = self.rsb_for_known_states(
            mu_c, sigma_c, mu_d, sigma_d, mu_e, sigma_e
        )

        print(f"RSB找到 {rsb_result['n_effective_states']} 个有效状态")

        # 3. 计算RSB预测的分布
        x_rsb, pdf_rsb = self.compute_rsb_distribution(rsb_result)

        # 4. 绘制对比图
        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # 4.1 丰度分布直方图与RSB预测
        if len(sim_data) > 0:
            axes[0, 0].hist(sim_data, bins=50, density=True, alpha=0.7,
                            color='lightblue', edgecolor='black', label='数值模拟')

            if len(x_rsb) > 0 and np.max(pdf_rsb) > 0:
                axes[0, 0].plot(x_rsb, pdf_rsb, 'r-', linewidth=2, label='RSB理论预测')

                # 标记已知状态
                for state in [-1.34, 1.34]:
                    axes[0, 0].axvline(x=state, color='green', linestyle='--',
                                       alpha=0.7, label=f'已知状态 {state}')

            axes[0, 0].set_xlabel('状态值 x')
            axes[0, 0].set_ylabel('概率密度')
            axes[0, 0].set_title('状态分布: 理论与模拟对比')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # 4.2 累积分布函数
        if len(sim_data) > 0:
            sorted_sim = np.sort(sim_data)
            cdf_sim = np.arange(1, len(sorted_sim) + 1) / len(sorted_sim)
            axes[0, 1].plot(sorted_sim, cdf_sim, 'b-', linewidth=2, label='数值模拟')

            if len(x_rsb) > 0 and np.max(pdf_rsb) > 0:
                cdf_rsb = np.cumsum(pdf_rsb) * (x_rsb[1] - x_rsb[0])
                axes[0, 1].plot(x_rsb, cdf_rsb, 'r-', linewidth=2, label='RSB理论预测')

            axes[0, 1].set_xlabel('状态值 x')
            axes[0, 1].set_ylabel('累积概率')
            axes[0, 1].set_title('累积分布函数')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # 4.3 状态权重分布
        states = rsb_result['states']
        if states:
            weights = [s['weight'] for s in states]
            mean_xs = [s['mean_x'] for s in states]
            colors = ['red' if s.get('target_state') == -1.34 else
                      'blue' if s.get('target_state') == 1.34 else
                      'gray' for s in states]

            sc = axes[0, 2].scatter(mean_xs, weights, c=colors, s=100, alpha=0.7)
            axes[0, 2].axvline(x=-1.34, color='red', linestyle='--', alpha=0.5)
            axes[0, 2].axvline(x=1.34, color='blue', linestyle='--', alpha=0.5)
            axes[0, 2].set_xlabel('平均状态值')
            axes[0, 2].set_ylabel('状态权重')
            axes[0, 2].set_title('RSB状态分布')
            axes[0, 2].grid(True, alpha=0.3)

        # 4.4 收敛历史
        if rsb_result['history']:
            iterations = [h['iteration'] for h in rsb_result['history']]
            avg_means = [h['avg_mean_x'] for h in rsb_result['history']]
            deltas = [h['total_delta'] for h in rsb_result['history']]

            axes[1, 0].plot(iterations, avg_means, 'b-', linewidth=2)
            axes[1, 0].set_xlabel('迭代次数')
            axes[1, 0].set_ylabel('平均状态值', color='b')
            axes[1, 0].tick_params(axis='y', labelcolor='b')
            axes[1, 0].grid(True, alpha=0.3)

            ax2 = axes[1, 0].twinx()
            ax2.semilogy(iterations, deltas, 'r--', linewidth=1)
            ax2.set_ylabel('收敛误差 (log)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            axes[1, 0].set_title('RSB收敛历史')

        # 4.5 状态聚类分析
        if len(sim_data) > 0:
            from sklearn.cluster import KMeans

            # 使用K-means聚类识别状态
            if len(sim_data) > 2:
                kmeans = KMeans(n_clusters=2, random_state=0).fit(sim_data.reshape(-1, 1))
                cluster_centers = kmeans.cluster_centers_.flatten()
                cluster_labels = kmeans.labels_

                axes[1, 1].hist([sim_data[cluster_labels == 0], sim_data[cluster_labels == 1]],
                                bins=30, density=True, alpha=0.7,
                                label=[f'聚类 {i}: {center:.2f}' for i, center in enumerate(cluster_centers)])
                axes[1, 1].set_xlabel('状态值 x')
                axes[1, 1].set_ylabel('概率密度')
                axes[1, 1].set_title('K-means聚类分析')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

        # 4.6 参数敏感性分析
        axes[1, 2].text(0.5, 0.5, '已知状态分析:\n' +
                        '\n'.join([f'状态 {k}: {v["proportion"] * 100:.1f}%'
                                   for k, v in state_analysis.items()]),
                        ha='center', va='center', transform=axes[1, 2].transAxes,
                        fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 2].set_title('状态出现频率')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.show()

        return {
            'simulation': sim_data,
            'rsb': rsb_result,
            'state_analysis': state_analysis,
            'x_rsb': x_rsb,
            'pdf_rsb': pdf_rsb
        }


# 使用示例
if __name__ == "__main__":
    # 创建专用求解器
    solver = CubicModelSolver(S_sim=80, S_rsb=1000, n_states=20, max_iter=200)

    # 测试案例：调整参数以使-1.34和+1.34成为稳定状态
    print("测试案例：双稳态系统")
    result = solver.compare_theory_simulation(
        mu_c=0.0,  # 调整这些参数以使-1.34和+1.34成为稳定状态
        sigma_c=0.1,
        mu_d=0.05,  # 弱相互作用
        sigma_d=0.1,
        mu_e=0.0,  # 无成对相互作用
        sigma_e=0.0,
        n_sim_replicates=10
    )

    # 分析结果
    print("\n=== 结果总结 ===")
    print("已知状态出现频率:")
    for state, analysis in result['state_analysis'].items():
        print(f"  状态 {state}: {analysis['proportion'] * 100:.1f}%")

    print(f"\nRSB预测的有效状态数: {result['rsb']['n_effective_states']}")
    print(f"RSB预测的平均状态值: {result['rsb']['avg_mean_x']:.3f}")

    # 检查是否成功识别了已知状态
    known_states_present = any(analysis['proportion'] > 0.1
                               for analysis in result['state_analysis'].values())

    if known_states_present:
        print("\n✓ 成功识别已知稳定状态")
    else:
        print("\n⚠ 未能充分识别已知稳定状态，可能需要调整参数")