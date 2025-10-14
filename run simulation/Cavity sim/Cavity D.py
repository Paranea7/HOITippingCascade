import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import fsolve
import warnings

warnings.filterwarnings('ignore')


class CubicCavitySolver:
    """
    三次系统空腔方法的迭代求解器
    """

    def __init__(self, S=1000, max_iter=1000, tol=1e-8):
        self.S = S  # 物种数
        self.max_iter = max_iter
        self.tol = tol
        self.history = []

    def cubic_response(self, h, branch='positive'):
        """
        解三次方程 x^3 - x = h
        返回指定分支的解
        """

        def equation(x, h_val):
            return x ** 3 - x - h_val

        # 对于每个h，寻找所有实数解
        solutions = []
        initial_guesses = [-2, -0.5, 0, 0.5, 2]

        for guess in initial_guesses:
            try:
                sol = fsolve(equation, guess, args=(h,), full_output=True)
                if sol[2] == 1 and abs(equation(sol[0], h)) < 1e-6:
                    sol_val = float(sol[0])
                    if sol_val not in [round(s, 6) for s in solutions]:
                        solutions.append(sol_val)
            except:
                continue

        solutions = sorted(solutions)

        if branch == 'positive':
            # 选择正分支（最大的正解）
            positive_sols = [s for s in solutions if s > 0]
            return positive_sols[0] if positive_sols else 0
        elif branch == 'stable':
            # 选择稳定分支（导数 < 0）
            stable_sols = [s for s in solutions if (3 * s ** 2 - 1) < 0]
            return stable_sols[0] if stable_sols else (solutions[0] if solutions else 0)
        else:
            return solutions[0] if solutions else 0

    def compute_moments(self, mu_h, sigma_h, u, branch='positive'):
        """
        计算矩量：φ, <x>, <x²>, <x³>
        """
        # 数值积分计算矩量
        x_min, x_max = -3, 3
        n_points = 1000
        x_vals = np.linspace(x_min, x_max, n_points)

        # 计算每个x对应的h
        h_vals = x_vals ** 3 - u * x_vals

        # 计算概率密度 P(h)
        P_h = np.exp(-0.5 * ((h_vals - mu_h) / sigma_h) ** 2) / (sigma_h * np.sqrt(2 * np.pi))

        # 变换到x空间：P(x) = P(h(x)) * |dh/dx|
        dh_dx = np.abs(3 * x_vals ** 2 - u)
        P_x = P_h * dh_dx

        # 归一化
        total_prob = np.trapz(P_x, x_vals)
        if total_prob > 0:
            P_x = P_x / total_prob

        # 计算存活物种的矩量 (x > 0)
        mask = x_vals > 0
        if np.sum(mask) == 0:
            return 0, 0, 0, 0

        x_surviving = x_vals[mask]
        P_surviving = P_x[mask]

        # 存活分数
        phi = np.trapz(P_surviving, x_surviving)

        if phi == 0:
            return 0, 0, 0, 0

        # 各阶矩
        mean_x = np.trapz(x_surviving * P_surviving, x_surviving) / phi
        mean_x2 = np.trapz(x_surviving ** 2 * P_surviving, x_surviving) / phi
        mean_x3 = np.trapz(x_surviving ** 3 * P_surviving, x_surviving) / phi

        return phi, mean_x, mean_x2, mean_x3

    def compute_response_coefficient(self, mu_h, sigma_h, u, phi, mean_x2, branch='positive'):
        """
        计算响应系数 v
        """
        if phi == 0:
            return 1.0

        # 数值积分计算平均响应系数
        x_min, x_max = 0.001, 3  # 只考虑正丰度
        n_points = 500
        x_vals = np.linspace(x_min, x_max, n_points)

        # 计算每个x对应的h和概率
        h_vals = x_vals ** 3 - u * x_vals
        P_h = np.exp(-0.5 * ((h_vals - mu_h) / sigma_h) ** 2) / (sigma_h * np.sqrt(2 * np.pi))
        dh_dx = np.abs(3 * x_vals ** 2 - u)
        P_x = P_h * dh_dx

        # 响应系数：v = 1/(3x² - u)
        # 避免除零
        denominator = 3 * x_vals ** 2 - u
        mask = np.abs(denominator) > 1e-10
        if np.sum(mask) == 0:
            return 1.0

        v_vals = np.zeros_like(x_vals)
        v_vals[mask] = 1.0 / denominator[mask]

        # 加权平均
        total_weight = np.trapz(P_x[mask], x_vals[mask])
        if total_weight == 0:
            return 1.0

        mean_v = np.trapz(v_vals[mask] * P_x[mask], x_vals[mask]) / total_weight

        return mean_v

    def iterate(self, mu_c, sigma_c, mu_d, sigma_d, mu_e=0, sigma_e=0,
                gamma=1.0, branch='positive', verbose=False):
        """
        执行迭代求解
        """
        # 初始化变量
        phi = 0.5  # 存活分数
        mean_x = 0.1  # 平均丰度
        mean_x2 = 0.1  # 二阶矩
        mean_x3 = 0.1  # 三阶矩
        v = 1.0  # 响应系数

        # 总相互作用方差
        sigma2_total = self.S * (sigma_d ** 2 + 2 * sigma_e ** 2 * mean_x2)

        for iteration in range(self.max_iter):
            # 1. 计算有效场参数
            mu_h = mu_c + phi * (mu_d * mean_x + mu_e * mean_x2)
            sigma_h = np.sqrt(sigma_c ** 2 + phi * (sigma_d ** 2 * mean_x2 + sigma_e ** 2 * mean_x2 ** 2))

            # 2. 计算有效自相互作用
            u = 1 + phi * sigma2_total * gamma * v

            # 3. 更新矩量
            phi_new, mean_x_new, mean_x2_new, mean_x3_new = self.compute_moments(
                mu_h, sigma_h, u, branch
            )

            # 4. 更新响应系数
            v_new = self.compute_response_coefficient(mu_h, sigma_h, u, phi_new, mean_x2_new, branch)

            # 检查收敛
            delta_phi = abs(phi_new - phi)
            delta_mean_x = abs(mean_x_new - mean_x)

            # 保存历史
            self.history.append({
                'iteration': iteration,
                'phi': phi_new,
                'mean_x': mean_x_new,
                'mean_x2': mean_x2_new,
                'mean_x3': mean_x3_new,
                'v': v_new,
                'mu_h': mu_h,
                'sigma_h': sigma_h,
                'u': u,
                'delta_phi': delta_phi
            })

            if verbose and iteration % 10 == 0:
                print(f"Iter {iteration}: φ={phi_new:.4f}, <x>={mean_x_new:.4f}, "
                      f"v={v_new:.4f}, Δφ={delta_phi:.2e}")

            # 更新变量
            phi, mean_x, mean_x2, mean_x3, v = phi_new, mean_x_new, mean_x2_new, mean_x3_new, v_new

            # 收敛检查
            if delta_phi < self.tol and delta_mean_x < self.tol:
                if verbose:
                    print(f"Converged after {iteration} iterations")
                break

        # 最终结果
        result = {
            'phi': phi,
            'mean_x': mean_x,
            'mean_x2': mean_x2,
            'mean_x3': mean_x3,
            'v': v,
            'mu_h': mu_h,
            'sigma_h': sigma_h,
            'u': u,
            'sigma2_total': sigma2_total,
            'converged': iteration < self.max_iter - 1,
            'iterations': iteration + 1
        }

        return result

    def plot_convergence(self):
        """绘制收敛历史"""
        if not self.history:
            print("No history available. Run iterate() first.")
            return

        iterations = [h['iteration'] for h in self.history]
        phi_vals = [h['phi'] for h in self.history]
        mean_x_vals = [h['mean_x'] for h in self.history]
        v_vals = [h['v'] for h in self.history]
        delta_phi = [h['delta_phi'] for h in self.history]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # φ 收敛
        axes[0, 0].plot(iterations, phi_vals, 'b-', linewidth=2)
        axes[0, 0].set_ylabel('Survival fraction φ')
        axes[0, 0].grid(True, alpha=0.3)

        # <x> 收敛
        axes[0, 1].plot(iterations, mean_x_vals, 'r-', linewidth=2)
        axes[0, 1].set_ylabel('Mean abundance <x>')
        axes[0, 1].grid(True, alpha=0.3)

        # v 收敛
        axes[1, 0].plot(iterations, v_vals, 'g-', linewidth=2)
        axes[1, 0].set_ylabel('Response coefficient v')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].grid(True, alpha=0.3)

        # Δφ 收敛
        axes[1, 1].semilogy(iterations, delta_phi, 'k-', linewidth=2)
        axes[1, 1].set_ylabel('Δφ (log scale)')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def test_parameter_sweep():
    """参数扫描测试"""
    solver = CubicCavitySolver(S=1000, max_iter=200, tol=1e-6)

    # 测试不同的相互作用强度
    mu_d_vals = np.linspace(-0.5, 0.5, 20)
    sigma_d_vals = np.linspace(0.1, 2.0, 20)

    results = []

    for mu_d in mu_d_vals:
        for sigma_d in sigma_d_vals:
            try:
                result = solver.iterate(
                    mu_c=0.1, sigma_c=0.1,
                    mu_d=mu_d, sigma_d=sigma_d,
                    mu_e=0, sigma_e=0,
                    gamma=1.0, branch='positive', verbose=False
                )

                results.append({
                    'mu_d': mu_d,
                    'sigma_d': sigma_d,
                    'phi': result['phi'],
                    'mean_x': result['mean_x'],
                    'v': result['v'],
                    'converged': result['converged']
                })
            except:
                continue

    return results


def plot_phase_diagram(results):
    """绘制相图"""
    mu_d = [r['mu_d'] for r in results]
    sigma_d = [r['sigma_d'] for r in results]
    phi = [r['phi'] for r in results]
    mean_x = [r['mean_x'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # φ 相图
    sc1 = axes[0].scatter(mu_d, sigma_d, c=phi, cmap='viridis', s=50)
    axes[0].set_xlabel('μ_d')
    axes[0].set_ylabel('σ_d')
    axes[0].set_title('Survival fraction φ')
    plt.colorbar(sc1, ax=axes[0])

    # <x> 相图
    sc2 = axes[1].scatter(mu_d, sigma_d, c=mean_x, cmap='plasma', s=50)
    axes[1].set_xlabel('μ_d')
    axes[1].set_ylabel('σ_d')
    axes[1].set_title('Mean abundance <x>')
    plt.colorbar(sc2, ax=axes[1])

    plt.tight_layout()
    plt.show()


# 使用示例
if __name__ == "__main__":
    # 创建求解器
    solver = CubicCavitySolver(S=1000, max_iter=200, tol=1e-8)

    print("=== 三次系统空腔方法求解 ===")

    # 测试案例1: 弱相互作用
    print("\n--- 测试案例1: 弱相互作用 ---")
    result1 = solver.iterate(
        mu_c=0.1, sigma_c=0.1,  # 内禀增长率
        mu_d=0.01, sigma_d=0.1,  # 线性相互作用
        mu_e=0, sigma_e=0,  # 成对相互作用 (设为0简化)
        gamma=1.0,  # 对称相互作用
        branch='positive',  # 选择正分支
        verbose=True
    )

    print(f"最终结果:")
    print(f"  存活分数 φ = {result1['phi']:.4f}")
    print(f"  平均丰度 <x> = {result1['mean_x']:.4f}")
    print(f"  响应系数 v = {result1['v']:.4f}")
    print(f"  有效自相互作用 u = {result1['u']:.4f}")
    print(f"  收敛: {result1['converged']}")

    # 绘制收敛图
    solver.plot_convergence()

    # 测试案例2: 强竞争
    print("\n--- 测试案例2: 强竞争 ---")
    solver2 = CubicCavitySolver(S=1000, max_iter=200, tol=1e-8)
    result2 = solver2.iterate(
        mu_c=0.1, sigma_c=0.1,
        mu_d=0.2, sigma_d=0.5,  # 更强的竞争
        mu_e=0, sigma_e=0,
        gamma=1.0,
        branch='positive',
        verbose=True
    )

    print(f"最终结果:")
    print(f"  存活分数 φ = {result2['phi']:.4f}")
    print(f"  平均丰度 <x> = {result2['mean_x']:.4f}")

    # 参数扫描
    print("\n--- 参数扫描 ---")
    results = test_parameter_sweep()
    plot_phase_diagram(results)

    # 测试多分支
    print("\n--- 测试不同分支 ---")
    for branch in ['positive', 'stable']:
        solver_branch = CubicCavitySolver(S=1000, max_iter=200, tol=1e-8)
        result_branch = solver_branch.iterate(
            mu_c=0.0, sigma_c=0.2,
            mu_d=-0.1, sigma_d=0.3,
            mu_e=0, sigma_e=0,
            gamma=1.0,
            branch=branch,
            verbose=False
        )
        print(f"分支 '{branch}': φ = {result_branch['phi']:.4f}, <x> = {result_branch['mean_x']:.4f}")