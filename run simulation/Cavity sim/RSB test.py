import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import trapz
import warnings

warnings.filterwarnings('ignore')


class TwoStateRSBSolver:
    """
    双稳态系统的简化RSB求解器
    """

    def __init__(self, S=1000, max_iter=100, tol=1e-6):
        self.S = S
        self.max_iter = max_iter
        self.tol = tol

        # 已知的稳定状态
        self.stable_states = [-1.015, 1.013]

    def two_state_rsb(self, mu_c, sigma_c, mu_d, sigma_d, mu_e=0, sigma_e=0):
        """
        双状态RSB分析 - 简化版本
        """
        print("进行双状态RSB分析...")

        # 初始化两个状态
        states = [
            {
                'id': 0,
                'target': -1.015,
                'weight': 0.5,
                'mean_x': -1.015,
                'mean_x2': 1.015 ** 2,
                'v': 1.0
            },
            {
                'id': 1,
                'target': 1.013,
                'weight': 0.5,
                'mean_x': 1.013,
                'mean_x2': 1.013 ** 2,
                'v': 1.0
            }
        ]

        history = []

        for iteration in range(self.max_iter):
            total_delta = 0

            # 计算总体矩量
            m_total = sum(s['weight'] * s['mean_x'] for s in states)
            q_total = sum(s['weight'] * s['mean_x2'] for s in states)

            # 更新每个状态
            for state in states:
                # 场参数
                phi = state['weight']
                mean_x = state['mean_x']
                mean_x2 = state['mean_x2']
                v = state['v']

                # 计算场参数
                mu_h = mu_c + phi * (mu_d * m_total + mu_e * q_total)

                if sigma_e > 0:
                    sigma_h = np.sqrt(sigma_c ** 2 + phi * (sigma_d ** 2 * q_total + sigma_e ** 2 * q_total ** 2))
                else:
                    sigma_h = np.sqrt(sigma_c ** 2 + phi * sigma_d ** 2 * q_total)

                # 总相互作用方差
                if sigma_e > 0:
                    sigma2_total = self.S * (sigma_d ** 2 + 2 * sigma_e ** 2)
                else:
                    sigma2_total = self.S * sigma_d ** 2

                u = 1 + phi * sigma2_total * v

                # 针对目标状态调整场参数
                target = state['target']
                h_target = target ** 3 - u * target
                mu_h = 0.8 * mu_h + 0.2 * h_target
                sigma_h = max(sigma_h * 0.9, 0.1)

                # 计算新矩量 - 专注于目标状态附近
                x_min, x_max = target - 0.5, target + 0.5
                x_vals = np.linspace(x_min, x_max, 200)

                h_vals = x_vals ** 3 - u * x_vals
                P_h = norm.pdf(h_vals, mu_h, sigma_h)
                dh_dx = np.abs(3 * x_vals ** 2 - u)

                mask = dh_dx > 1e-10
                if np.sum(mask) > 0:
                    P_x = np.zeros_like(x_vals)
                    P_x[mask] = P_h[mask] / dh_dx[mask]

                    # 归一化
                    total_prob = trapz(P_x, x_vals)
                    if total_prob > 0:
                        P_x = P_x / total_prob

                        mean_x_new = trapz(x_vals * P_x, x_vals)
                        mean_x2_new = trapz(x_vals ** 2 * P_x, x_vals)

                        # 向目标状态加强收敛
                        mix_ratio = min(0.5, 0.2 + 0.3 * iteration / self.max_iter)
                        mean_x_new = (1 - mix_ratio) * mean_x_new + mix_ratio * target
                        mean_x2_new = (1 - mix_ratio) * mean_x2_new + mix_ratio * target ** 2
                    else:
                        mean_x_new, mean_x2_new = mean_x, mean_x2
                else:
                    mean_x_new, mean_x2_new = mean_x, mean_x2

                # 更新响应系数
                if abs(3 * mean_x2_new - u) > 1e-6:
                    v_new = 1.0 / (3 * mean_x2_new - u)
                    v_new = np.clip(v_new, 0.1, 10.0)
                else:
                    v_new = v

                # 限制范围
                mean_x_new = np.clip(mean_x_new, -1.5, 1.5)
                mean_x2_new = np.clip(mean_x2_new, 0.5, 2.5)

                delta = abs(mean_x_new - mean_x)
                total_delta += delta

                # 更新状态
                state['mean_x'] = mean_x_new
                state['mean_x2'] = mean_x2_new
                state['v'] = v_new
                state['mu_h'] = mu_h
                state['sigma_h'] = sigma_h
                state['u'] = u

            # 更新权重 - 基于稳定性
            weights = []
            for state in states:
                distance = abs(state['mean_x'] - state['target'])
                weight = np.exp(-10 * distance)
                weights.append(weight)

            total_weight = sum(weights)
            for i, state in enumerate(states):
                state['weight'] = weights[i] / total_weight

            # 记录历史
            avg_mean_x = sum(s['weight'] * s['mean_x'] for s in states)
            history.append({
                'iteration': iteration,
                'avg_mean_x': avg_mean_x,
                'total_delta': total_delta,
                'weights': [s['weight'] for s in states],
                'means': [s['mean_x'] for s in states]
            })

            if total_delta < self.tol and iteration > 10:
                print(f"双状态RSB在迭代 {iteration} 收敛")
                break

        # 最终结果
        result = {
            'states': states,
            'avg_mean_x': avg_mean_x,
            'history': history,
            'converged': iteration < self.max_iter - 1
        }

        return result

    def compute_distribution(self, rsb_result, n_points=1000):
        """
        计算双状态分布
        """
        states = rsb_result['states']

        x_range = np.linspace(-1.5, 1.5, n_points)
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
            state_norm = trapz(state_pdf, x_range)
            if state_norm > 0:
                state_pdf = state_pdf / state_norm

            pdf_total += weight * state_pdf

        # 最终归一化
        total_norm = trapz(pdf_total, x_range)
        if total_norm > 0:
            pdf_total = pdf_total / total_norm

        return x_range, pdf_total

    def stability_analysis(self, mu_c, sigma_c, mu_d, sigma_d, mu_e=0, sigma_e=0):
        """
        双状态稳定性分析
        """
        print("进行双状态稳定性分析...")

        stability_results = {}

        for state_value in self.stable_states:
            # 简化的雅可比矩阵分析
            # 对于均匀状态，特征值主要取决于对角线元素
            eigenvalue = -3 * state_value ** 2 + 1

            stability_results[state_value] = {
                'max_real_eigenvalue': eigenvalue,
                'stable': eigenvalue < 0
            }

            status = "稳定" if eigenvalue < 0 else "不稳定"
            print(f"状态 {state_value}: 特征值 = {eigenvalue:.3f} ({status})")

        return stability_results

    def analyze_simulation(self, sim_data, tolerance=0.1):
        """
        分析模拟数据中的双状态
        """
        results = {}

        for target_state in self.stable_states:
            mask = np.abs(sim_data - target_state) < tolerance
            count = np.sum(mask)
            proportion = count / len(sim_data) if len(sim_data) > 0 else 0

            results[target_state] = {
                'count': count,
                'proportion': proportion,
                'mean': np.mean(sim_data[mask]) if count > 0 else 0,
                'std': np.std(sim_data[mask]) if count > 0 else 0
            }

            print(f"状态 {target_state}: 出现 {count} 次 ({proportion * 100:.1f}%)")

        return results

    def compare_theory_simulation(self, sim_data, mu_c, sigma_c, mu_d, sigma_d, mu_e=0, sigma_e=0, figsize=(12, 8)):
        """
        比较理论和模拟结果 - 简化版本
        """
        print("=== 双状态系统分析 ===")

        # 1. 分析模拟数据
        state_analysis = self.analyze_simulation(sim_data)

        # 2. RSB理论预测
        rsb_result = self.two_state_rsb(mu_c, sigma_c, mu_d, sigma_d, mu_e, sigma_e)

        # 3. 计算理论分布
        x_rsb, pdf_rsb = self.compute_distribution(rsb_result)

        # 4. 稳定性分析
        stability_results = self.stability_analysis(mu_c, sigma_c, mu_d, sigma_d, mu_e, sigma_e)

        # 5. 绘制结果
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 5.1 分布对比
        axes[0, 0].hist(sim_data, bins=50, density=True, alpha=0.7,
                        color='lightblue', edgecolor='black', label='数值模拟')
        axes[0, 0].plot(x_rsb, pdf_rsb, 'r-', linewidth=2, label='RSB理论')

        for state in self.stable_states:
            axes[0, 0].axvline(x=state, color='green', linestyle='--',
                               alpha=0.7, label=f'目标状态 {state}')

        axes[0, 0].set_xlabel('状态值 x')
        axes[0, 0].set_ylabel('概率密度')
        axes[0, 0].set_title('状态分布对比')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 5.2 状态权重
        states = rsb_result['states']
        labels = [f"状态 {s['target']}" for s in states]
        weights = [s['weight'] for s in states]

        axes[0, 1].bar(labels, weights, alpha=0.7, color=['red', 'blue'])
        axes[0, 1].set_ylabel('状态权重')
        axes[0, 1].set_title('RSB状态权重')
        axes[0, 1].grid(True, alpha=0.3)

        # 5.3 收敛历史
        history = rsb_result['history']
        if history:
            iterations = [h['iteration'] for h in history]
            weights1 = [h['weights'][0] for h in history]
            weights2 = [h['weights'][1] for h in history]

            axes[1, 0].plot(iterations, weights1, 'r-', label='状态1权重')
            axes[1, 0].plot(iterations, weights2, 'b-', label='状态2权重')
            axes[1, 0].set_xlabel('迭代次数')
            axes[1, 0].set_ylabel('状态权重')
            axes[1, 0].set_title('权重收敛历史')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # 5.4 分析总结
        summary_text = '稳定性分析:\n'
        for state, analysis in stability_results.items():
            status = "稳定" if analysis['stable'] else "不稳定"
            summary_text += f'状态 {state}: {analysis["max_real_eigenvalue"]:.3f} ({status})\n'

        summary_text += '\n状态出现频率:\n'
        for state, analysis in state_analysis.items():
            summary_text += f'状态 {state}: {analysis["proportion"] * 100:.1f}%\n'

        summary_text += f'\n理论权重:\n'
        for state in states:
            summary_text += f'状态 {state["target"]}: {state["weight"] * 100:.1f}%\n'

        axes[1, 1].text(0.5, 0.5, summary_text, ha='center', va='center',
                        transform=axes[1, 1].transAxes, fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 1].set_title('分析总结')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.show()

        return {
            'simulation': sim_data,
            'rsb': rsb_result,
            'state_analysis': state_analysis,
            'stability_analysis': stability_results,
            'x_rsb': x_rsb,
            'pdf_rsb': pdf_rsb
        }


# 使用示例
if __name__ == "__main__":
    # 创建双状态求解器
    solver = TwoStateRSBSolver(S=1000, max_iter=50)

    # 生成模拟数据（这里用混合高斯分布代替实际模拟）
    np.random.seed(42)
    n_samples = 1000
    data1 = np.random.normal(-1.015, 0.6, n_samples // 2)
    data2 = np.random.normal(1.013, 0.6, n_samples // 2)
    sim_data = np.concatenate([data1, data2])

    # 运行分析
    result = solver.compare_theory_simulation(
        sim_data=sim_data,
        mu_c=0.0,
        sigma_c=0.4,
        mu_d=0.5,
        sigma_d=0.3
    )

    # 输出结果
    print("\n=== 双状态分析结果 ===")
    print("模拟数据状态分布:")
    for state, analysis in result['state_analysis'].items():
        print(f"  状态 {state}: {analysis['proportion'] * 100:.1f}%")

    print("\n理论预测状态分布:")
    for state in result['rsb']['states']:
        print(f"  状态 {state['target']}: {state['weight'] * 100:.1f}%")