import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize
from scipy.stats import gaussian_kde, norm
import warnings

warnings.filterwarnings('ignore')


class RSBCubicSolver:
    """
    三次系统的RSB求解器 - 完整实现
    """

    def __init__(self, n_states=2, S=1000, max_iter=500, tol=1e-8):
        self.n_states = n_states  # 状态数量
        self.S = S
        self.max_iter = max_iter
        self.tol = tol
        self.history = []

    def cubic_equation_roots(self, h, u=1):
        """
        解三次方程: x^3 - u*x = h
        返回所有实数解
        """
        # 转换为标准形式: x^3 - u*x - h = 0
        # 使用解析解公式
        if u <= 0:
            # 特殊情况，直接数值求解
            def f(x):
                return x ** 3 - u * x - h

            roots = []
            for guess in [-2, 0, 2]:
                try:
                    root = optimize.fsolve(f, guess)[0]
                    if abs(f(root)) < 1e-8 and root not in roots:
                        roots.append(root)
                except:
                    continue
            return sorted(roots)

        # 正常情况的解析解
        discriminant = (h / 2) ** 2 - (u / 3) ** 3
        roots = []

        if discriminant > 0:  # 一个实根
            A = np.cbrt(-h / 2 + np.sqrt(discriminant))
            B = np.cbrt(-h / 2 - np.sqrt(discriminant))
            roots.append(A + B)
        else:  # 三个实根
            r = np.sqrt((u / 3) ** 3)
            if abs(h) < 1e-10:
                theta = np.pi / 2
            else:
                theta = np.arccos(-h / (2 * r))

            for k in range(3):
                root = 2 * np.sqrt(u / 3) * np.cos((theta + 2 * np.pi * k) / 3)
                roots.append(root)

        # 过滤实数解并排序
        real_roots = sorted([float(r) for r in roots if abs(r.imag) < 1e-10])
        return real_roots

    def state_branch_selector(self, h, u, state_id, branch_strategy='mixed'):
        """
        根据状态ID和分支策略选择三次方程的解
        """
        roots = self.cubic_equation_roots(h, u)
        if not roots:
            return 0.0

        if branch_strategy == 'positive':
            # 总是选择最大的正根
            positive_roots = [r for r in roots if r > 0]
            return max(positive_roots) if positive_roots else 0.0

        elif branch_strategy == 'stable':
            # 选择稳定分支 (导数负)
            stable_roots = [r for r in roots if (3 * r ** 2 - u) < 0]
            return stable_roots[0] if stable_roots else roots[0]

        elif branch_strategy == 'alternating':
            # 交替选择不同分支
            if state_id % 3 == 0:
                return max(roots)  # 最大根
            elif state_id % 3 == 1:
                return min(roots)  # 最小根
            else:
                # 中间根 (如果有三个根)
                return roots[len(roots) // 2] if len(roots) > 1 else roots[0]

        elif branch_strategy == 'mixed':
            # 混合策略：根据状态ID决定
            strategies = ['positive', 'stable', 'alternating']
            strategy = strategies[state_id % len(strategies)]
            return self.state_branch_selector(h, u, state_id, strategy)

        else:
            return roots[0]  # 默认选择第一个根

    def compute_state_moments(self, mu_h, sigma_h, u, state_id):
        """
        计算状态特定的矩量
        """
        # 使用数值积分计算状态特定的分布
        n_points = 2000
        h_min = max(mu_h - 5 * sigma_h, -10)
        h_max = min(mu_h + 5 * sigma_h, 10)
        h_vals = np.linspace(h_min, h_max, n_points)

        # 场分布
        P_h = norm.pdf(h_vals, mu_h, sigma_h)

        # 计算响应函数
        x_vals = np.array([self.state_branch_selector(h, u, state_id) for h in h_vals])

        # 变换到x空间: P(x) = P(h(x)) * |dh/dx|
        # dh/dx = 3x^2 - u
        dh_dx = np.abs(3 * x_vals ** 2 - u)
        mask = dh_dx > 1e-10
        if np.sum(mask) == 0:
            return 0, 0, 0

        P_x = np.zeros_like(x_vals)
        P_x[mask] = P_h[mask] / dh_dx[mask]

        # 归一化
        total_prob = np.trapz(P_x, x_vals)
        if total_prob <= 0:
            return 0, 0, 0

        P_x = P_x / total_prob

        # 存活物种 (x > 0)
        mask_survive = (x_vals > 0) & (P_x > 1e-10)
        if np.sum(mask_survive) == 0:
            return 0, 0, 0

        x_survive = x_vals[mask_survive]
        P_survive = P_x[mask_survive]

        phi = np.trapz(P_survive, x_survive)
        if phi <= 0:
            return 0, 0, 0

        mean_x = np.trapz(x_survive * P_survive, x_survive) / phi
        mean_x2 = np.trapz(x_survive ** 2 * P_survive, x_survive) / phi

        return phi, mean_x, mean_x2

    def compute_response_coefficient(self, mu_h, sigma_h, u, phi, mean_x2, state_id):
        """
        计算状态特定的响应系数
        """
        if phi <= 0:
            return 1.0

        # 数值积分计算平均响应系数
        n_points = 1000
        h_min = max(mu_h - 4 * sigma_h, -8)
        h_max = min(mu_h + 4 * sigma_h, 8)
        h_vals = np.linspace(h_min, h_max, n_points)

        P_h = norm.pdf(h_vals, mu_h, sigma_h)

        # 计算每个h对应的x和响应系数
        x_vals = np.array([self.state_branch_selector(h, u, state_id) for h in h_vals])
        v_vals = np.zeros_like(x_vals)

        mask = np.abs(3 * x_vals ** 2 - u) > 1e-8
        v_vals[mask] = 1.0 / (3 * x_vals[mask] ** 2 - u)

        # 限制响应系数范围
        v_vals = np.clip(v_vals, -10, 10)

        # 只考虑存活物种
        survive_mask = x_vals > 0
        if np.sum(survive_mask) == 0:
            return 1.0

        # 加权平均
        weights = P_h[survive_mask]
        if np.sum(weights) == 0:
            return 1.0

        mean_v = np.average(v_vals[survive_mask], weights=weights)
        return mean_v

    def compute_state_complexity(self, states, m):
        """
        计算状态复杂度和权重
        """
        n_states = len(states)
        if n_states == 0:
            return 0, np.ones(1)

        # 计算每个状态的自由能 (简化版本)
        free_energies = []
        for state in states:
            phi = state['phi']
            mean_x = state['mean_x']
            mean_x2 = state['mean_x2']

            if phi <= 0:
                # 死亡状态的自由能很高
                f_alpha = 1000
            else:
                # 简化的自由能表达式
                # 包含熵项和能量项
                entropy_term = -np.log(phi + 1e-10)
                energy_term = 0.5 * mean_x2 - 0.25 * state.get('mean_x4', mean_x2 ** 2)
                f_alpha = entropy_term + energy_term

            free_energies.append(f_alpha)

        f_alpha = np.array(free_energies)

        # 计算权重 (玻尔兹曼分布)
        weights = np.exp(-m * self.S * f_alpha)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            # 如果所有权重都为0，均匀分布
            weights = np.ones(n_states) / n_states

        # 计算复杂度
        Z = np.sum(np.exp(-m * self.S * f_alpha))
        if Z > 0:
            Sigma = np.log(Z) / self.S + m * np.sum(weights * f_alpha)
        else:
            Sigma = 0

        return Sigma, weights

    def initialize_states(self, n_states):
        """
        初始化状态种群
        """
        states = []
        for i in range(n_states):
            # 随机初始化，但确保多样性
            phi = np.random.uniform(0.1, 0.9)
            mean_x = np.random.uniform(0.05, 2.0)
            mean_x2 = np.random.uniform(0.01, 4.0)

            state = {
                'id': i,
                'phi': phi,
                'mean_x': mean_x,
                'mean_x2': mean_x2,
                'v': np.random.uniform(0.5, 2.0),
                'weight': 1.0 / n_states,
                'branch_strategy': ['positive', 'stable', 'alternating'][i % 3]
            }
            states.append(state)

        return states

    def rsb_iteration(self, mu_c, sigma_c, mu_d, sigma_d, mu_e, sigma_e,
                      gamma=1.0, m=0.5, verbose=True):
        """
        RSB迭代求解主函数
        """
        # 初始化状态种群
        states = self.initialize_states(self.n_states)

        # 总相互作用方差
        sigma2_total = self.S * (sigma_d ** 2 + 2 * sigma_e ** 2)

        print(f"开始RSB迭代: {self.n_states}个状态, m={m}")

        for iteration in range(self.max_iter):
            new_states = []
            total_delta = 0
            n_effective = 0

            for state in states:
                phi_alpha = state['phi']
                mean_x_alpha = state['mean_x']
                mean_x2_alpha = state['mean_x2']
                v_alpha = state['v']
                state_id = state['id']

                # 跳过死亡状态 (但保留少量权重)
                if phi_alpha < 0.01 and state['weight'] < 0.01:
                    new_states.append(state.copy())
                    continue

                # 状态特定的场参数
                mu_h_alpha = mu_c + phi_alpha * (mu_d * mean_x_alpha + mu_e * mean_x2_alpha)
                sigma_h_alpha = np.sqrt(sigma_c ** 2 + phi_alpha *
                                        (sigma_d ** 2 * mean_x2_alpha + sigma_e ** 2 * mean_x2_alpha ** 2))

                # 状态特定的有效自相互作用
                u_alpha = 1 + phi_alpha * sigma2_total * gamma * v_alpha

                # 更新状态特定的矩量
                phi_new, mean_x_new, mean_x2_new = self.compute_state_moments(
                    mu_h_alpha, sigma_h_alpha, u_alpha, state_id
                )

                # 更新响应系数
                v_new = self.compute_response_coefficient(
                    mu_h_alpha, sigma_h_alpha, u_alpha, phi_new, mean_x2_new, state_id
                )

                # 限制范围防止数值不稳定
                phi_new = np.clip(phi_new, 0, 1)
                mean_x_new = np.clip(mean_x_new, 0, 5)
                mean_x2_new = np.clip(mean_x2_new, 0, 25)
                v_new = np.clip(v_new, 0.1, 10)

                delta = abs(phi_new - phi_alpha) + abs(mean_x_new - mean_x_alpha)
                total_delta += delta

                if phi_new > 0.01:
                    n_effective += 1

                new_state = {
                    'id': state_id,
                    'phi': phi_new,
                    'mean_x': mean_x_new,
                    'mean_x2': mean_x2_new,
                    'v': v_new,
                    'weight': state['weight'],
                    'branch_strategy': state['branch_strategy'],
                    'mu_h': mu_h_alpha,
                    'sigma_h': sigma_h_alpha,
                    'u': u_alpha
                }
                new_states.append(new_state)

            states = new_states

            # 每10次迭代更新权重
            if iteration % 10 == 0 or iteration < 10:
                Sigma, weights = self.compute_state_complexity(states, m)
                for i, state in enumerate(states):
                    state['weight'] = weights[i]

            # 过滤有效状态
            effective_states = [s for s in states if s['weight'] > 0.01 and s['phi'] > 0.01]
            if len(effective_states) == 0:
                effective_states = [s for s in states if s['phi'] > 0]
            if len(effective_states) == 0:
                effective_states = states

            # 计算宏观平均
            avg_phi = np.sum([s['phi'] * s['weight'] for s in effective_states])
            avg_mean_x = np.sum([s['mean_x'] * s['weight'] for s in effective_states])

            # 记录历史
            self.history.append({
                'iteration': iteration,
                'states': [s.copy() for s in effective_states],
                'avg_phi': avg_phi,
                'avg_mean_x': avg_mean_x,
                'total_delta': total_delta,
                'n_effective': len(effective_states),
                'complexity': Sigma
            })

            if verbose and iteration % 20 == 0:
                print(f"Iter {iteration}: Avg φ={avg_phi:.4f}, Avg <x>={avg_mean_x:.4f}, "
                      f"States={len(effective_states)}, Δ={total_delta:.2e}")

            # 收敛检查
            if total_delta < self.tol and iteration > 20:
                if verbose:
                    print(f"RSB收敛于 {iteration} 次迭代")
                break

        # 最终过滤和权重归一化
        effective_states = [s for s in states if s['weight'] > 0.001 and s['phi'] > 0.001]
        if len(effective_states) == 0:
            effective_states = states

        total_weight = sum(s['weight'] for s in effective_states)
        for state in effective_states:
            state['weight'] /= total_weight

        # 最终结果
        result = {
            'states': effective_states,
            'avg_phi': np.sum([s['phi'] * s['weight'] for s in effective_states]),
            'avg_mean_x': np.sum([s['mean_x'] * s['weight'] for s in effective_states]),
            'avg_mean_x2': np.sum([s['mean_x2'] * s['weight'] for s in effective_states]),
            'n_effective_states': len(effective_states),
            'complexity': self.history[-1]['complexity'] if self.history else 0,
            'converged': iteration < self.max_iter - 1,
            'iterations': iteration + 1
        }

        return result

    def plot_state_distributions(self, result, figsize=(15, 10)):
        """
        绘制完整的状态分布图
        """
        states = result['states']
        if not states:
            print("没有有效状态可绘制")
            return

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # 1. 状态权重分布 (饼图)
        weights = [s['weight'] for s in states]
        labels = [f"State {s['id']}\n(φ={s['phi']:.3f})" for s in states]

        # 只显示权重较大的状态
        large_weight_mask = np.array(weights) > 0.05
        if np.sum(large_weight_mask) > 0:
            pie_weights = np.array(weights)[large_weight_mask]
            pie_labels = np.array(labels)[large_weight_mask]
        else:
            # 如果所有权重都很小，显示前5个
            idx = np.argsort(weights)[-5:]
            pie_weights = np.array(weights)[idx]
            pie_labels = np.array(labels)[idx]

        axes[0, 0].pie(pie_weights, labels=pie_labels, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('State Weight Distribution')

        # 2. 存活分数分布
        phis = [s['phi'] for s in states]
        weights = [s['weight'] for s in states]

        # 按权重加权的直方图
        bins = np.linspace(0, 1, 20)
        hist, bin_edges = np.histogram(phis, bins=bins, weights=weights)
        axes[0, 1].bar(bin_edges[:-1], hist, width=np.diff(bin_edges),
                       alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].set_xlabel('Survival Fraction φ')
        axes[0, 1].set_ylabel('Probability Density')
        axes[0, 1].set_title('Distribution of Survival Fractions')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 平均丰度分布
        mean_xs = [s['mean_x'] for s in states]
        hist, bin_edges = np.histogram(mean_xs, bins=20, weights=weights)
        axes[0, 2].bar(bin_edges[:-1], hist, width=np.diff(bin_edges),
                       alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 2].set_xlabel('Mean Abundance <x>')
        axes[0, 2].set_ylabel('Probability Density')
        axes[0, 2].set_title('Distribution of Mean Abundances')
        axes[0, 2].grid(True, alpha=0.3)

        # 4. φ vs <x> 散点图 (按权重着色)
        sc = axes[1, 0].scatter(phis, mean_xs, c=weights, s=100,
                                cmap='viridis', alpha=0.7, edgecolors='black')
        axes[1, 0].set_xlabel('Survival Fraction φ')
        axes[1, 0].set_ylabel('Mean Abundance <x>')
        axes[1, 0].set_title('State Space: φ vs <x>')
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(sc, ax=axes[1, 0], label='State Weight')

        # 5. 收敛历史
        if self.history:
            iterations = [h['iteration'] for h in self.history]
            avg_phis = [h['avg_phi'] for h in self.history]
            n_states = [h['n_effective'] for h in self.history]

            axes[1, 1].plot(iterations, avg_phis, 'b-', linewidth=2, label='Avg φ')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Average Survival Fraction', color='b')
            axes[1, 1].tick_params(axis='y', labelcolor='b')
            axes[1, 1].grid(True, alpha=0.3)

            ax2 = axes[1, 1].twinx()
            ax2.plot(iterations, n_states, 'r--', linewidth=2, label='# States')
            ax2.set_ylabel('Number of Effective States', color='r')
            ax2.tick_params(axis='y', labelcolor='r')

            axes[1, 1].set_title('Convergence History')
            axes[1, 1].legend(loc='upper left')
            ax2.legend(loc='upper right')

        # 6. 状态复杂度历史
        if self.history:
            complexities = [h['complexity'] for h in self.history]
            axes[1, 2].plot(iterations, complexities, 'g-', linewidth=2)
            axes[1, 2].set_xlabel('Iteration')
            axes[1, 2].set_ylabel('Complexity Σ')
            axes[1, 2].set_title('Complexity Evolution')
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 打印统计信息
        print(f"\n=== 状态分布统计 ===")
        print(f"总状态数: {len(states)}")
        print(f"有效状态数: {result['n_effective_states']}")
        print(f"平均存活分数: {result['avg_phi']:.4f}")
        print(f"平均丰度: {result['avg_mean_x']:.4f}")
        print(f"状态复杂度: {result['complexity']:.4f}")

        if len(states) > 0:
            print(f"\n主要状态详情:")
            sorted_states = sorted(states, key=lambda x: x['weight'], reverse=True)
            for i, state in enumerate(sorted_states[:5]):  # 显示前5个状态
                print(f"  状态 {state['id']}: 权重={state['weight']:.3f}, φ={state['phi']:.3f}, "
                      f"<x>={state['mean_x']:.3f}")

    def parameter_sweep_analysis(self, mu_d_range, mu_e_range, fixed_params, m=0.5):
        """
        参数扫描分析
        """
        results = np.zeros((len(mu_e_range), len(mu_d_range), 5))  # phi, mean_x, n_states, complexity, weight_entropy

        for i, mu_e in enumerate(mu_e_range):
            for j, mu_d in enumerate(mu_d_range):
                try:
                    # 重置求解器
                    self.history = []

                    result = self.rsb_iteration(
                        mu_d=mu_d, mu_e=mu_e,
                        **fixed_params, m=m, verbose=False
                    )

                    # 存储结果
                    results[i, j, 0] = result['avg_phi']  # 平均存活分数
                    results[i, j, 1] = result['avg_mean_x']  # 平均丰度
                    results[i, j, 2] = result['n_effective_states']  # 有效状态数
                    results[i, j, 3] = result['complexity']  # 复杂度

                    # 权重熵 (衡量状态分布的分散程度)
                    weights = [s['weight'] for s in result['states']]
                    if len(weights) > 1:
                        entropy = -sum(w * np.log(w + 1e-10) for w in weights)
                        results[i, j, 4] = entropy
                    else:
                        results[i, j, 4] = 0

                except Exception as e:
                    print(f"Error at (μ_d={mu_d:.3f}, μ_e={mu_e:.3f}): {e}")
                    results[i, j] = np.nan

        return results

    def plot_parameter_sweep(self, mu_d_range, mu_e_range, results):
        """
        绘制参数扫描结果
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 平均存活分数
        im1 = axes[0, 0].imshow(results[:, :, 0],
                                extent=[mu_d_range[0], mu_d_range[-1], mu_e_range[0], mu_e_range[-1]],
                                origin='lower', aspect='auto', cmap='viridis')
        axes[0, 0].set_xlabel('μ_d')
        axes[0, 0].set_ylabel('μ_e')
        axes[0, 0].set_title('Average Survival Fraction φ')
        plt.colorbar(im1, ax=axes[0, 0])

        # 平均丰度
        im2 = axes[0, 1].imshow(results[:, :, 1],
                                extent=[mu_d_range[0], mu_d_range[-1], mu_e_range[0], mu_e_range[-1]],
                                origin='lower', aspect='auto', cmap='plasma')
        axes[0, 1].set_xlabel('μ_d')
        axes[0, 1].set_ylabel('μ_e')
        axes[0, 1].set_title('Average Abundance <x>')
        plt.colorbar(im2, ax=axes[0, 1])

        # 有效状态数
        im3 = axes[0, 2].imshow(results[:, :, 2],
                                extent=[mu_d_range[0], mu_d_range[-1], mu_e_range[0], mu_e_range[-1]],
                                origin='lower', aspect='auto', cmap='Set1')
        axes[0, 2].set_xlabel('μ_d')
        axes[0, 2].set_ylabel('μ_e')
        axes[0, 2].set_title('Number of Effective States')
        plt.colorbar(im3, ax=axes[0, 2])

        # 复杂度
        im4 = axes[1, 0].imshow(results[:, :, 3],
                                extent=[mu_d_range[0], mu_d_range[-1], mu_e_range[0], mu_e_range[-1]],
                                origin='lower', aspect='auto', cmap='coolwarm')
        axes[1, 0].set_xlabel('μ_d')
        axes[1, 0].set_ylabel('μ_e')
        axes[1, 0].set_title('Complexity Σ')
        plt.colorbar(im4, ax=axes[1, 0])

        # 权重熵
        im5 = axes[1, 1].imshow(results[:, :, 4],
                                extent=[mu_d_range[0], mu_d_range[-1], mu_e_range[0], mu_e_range[-1]],
                                origin='lower', aspect='auto', cmap='hot')
        axes[1, 1].set_xlabel('μ_d')
        axes[1, 1].set_ylabel('μ_e')
        axes[1, 1].set_title('Weight Entropy (State Diversity)')
        plt.colorbar(im5, ax=axes[1, 1])

        # 多稳态区域标记
        multi_stable_mask = results[:, :, 2] > 1.5
        axes[1, 2].imshow(multi_stable_mask.astype(float),
                          extent=[mu_d_range[0], mu_d_range[-1], mu_e_range[0], mu_e_range[-1]],
                          origin='lower', aspect='auto', cmap='RdYlBu')
        axes[1, 2].set_xlabel('μ_d')
        axes[1, 2].set_ylabel('μ_e')
        axes[1, 2].set_title('Multistable Regions (N_states > 1)')

        plt.tight_layout()
        plt.show()


# 使用示例
if __name__ == "__main__":
    print("=== 三次系统RSB状态分布分析 ===")

    # 创建RSB求解器
    rsb_solver = RSBCubicSolver(n_states=15, S=1000, max_iter=200, tol=1e-6)

    # 测试案例1: 弱相互作用，期望单一状态
    print("\n--- 测试案例1: 弱相互作用 ---")
    result1 = rsb_solver.rsb_iteration(
        mu_c=0.1, sigma_c=0.1,
        mu_d=0.01, sigma_d=0.1,
        mu_e=0.0, sigma_e=0.0,
        gamma=1.0, m=0.3, verbose=True
    )

    # 绘制状态分布
    rsb_solver.plot_state_distributions(result1)

    # 测试案例2: 强相互作用，可能多稳态
    print("\n--- 测试案例2: 强相互作用 ---")
    rsb_solver2 = RSBCubicSolver(n_states=20, S=1000, max_iter=200, tol=1e-6)
    result2 = rsb_solver2.rsb_iteration(
        mu_c=0.1, sigma_c=0.1,
        mu_d=0.2, sigma_d=0.5,
        mu_e=0.05, sigma_e=0.2,
        gamma=1.0, m=0.3, verbose=True
    )

    # 绘制状态分布
    rsb_solver2.plot_state_distributions(result2)

    # 参数扫描分析
    print("\n--- 参数扫描分析 ---")
    mu_d_range = np.linspace(-0.3, 0.3, 15)
    mu_e_range = np.linspace(-0.1, 0.1, 15)

    fixed_params = {
        'mu_c': 0.1, 'sigma_c': 0.1,
        'sigma_d': 0.3, 'sigma_e': 0.1,
        'gamma': 1.0
    }

    results = rsb_solver.parameter_sweep_analysis(mu_d_range, mu_e_range, fixed_params)
    rsb_solver.plot_parameter_sweep(mu_d_range, mu_e_range, results)

    # 测试不同m值的影响
    print("\n--- 不同巴黎参数m的影响 ---")
    m_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    for m in m_values:
        solver_m = RSBCubicSolver(n_states=12, S=1000, max_iter=150, tol=1e-6)
        result_m = solver_m.rsb_iteration(
            mu_c=0.1, sigma_c=0.1,
            mu_d=0.1, sigma_d=0.3,
            mu_e=0.02, sigma_e=0.1,
            gamma=1.0, m=m, verbose=False
        )
        print(f"m = {m}: φ = {result_m['avg_phi']:.4f}, "
              f"States = {result_m['n_effective_states']}, "
              f"Complexity = {result_m['complexity']:.4f}")