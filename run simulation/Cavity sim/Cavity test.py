import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

class SpecificBistableCavitySolver:
    """
    固定均值为 ±1.324718 的双稳态系统空穴法求解器
    """

    def __init__(self, params):
        """
        初始化求解器

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

        # 固定均值 - 势函数的最小值位置
        self.μ1 = -1.324718  # 状态1均值固定
        self.μ2 = 1.324718  # 状态2均值固定
        self.μ_sq = 1.754876  # 均值的平方，预计算以提高效率

        # 收敛参数
        self.max_iter = params.get('max_iter', 1000)
        self.tol = params.get('tol', 1e-6)
        self.relaxation = params.get('relaxation', 0.5)

        # 初始化状态参数
        self.initialize_parameters()

    def initialize_parameters(self):
        """
        初始化状态参数
        """
        # 初始猜测值
        self.phi = 0.6  # 共存比例

        # 两个状态的方差 - 初始猜测
        self.σ1_sq = 0.05
        self.σ2_sq = 0.05

        # 响应参数
        self.v1 = 1.0
        self.v2 = 1.0

        # 初始化总体矩与中间量，避免未定义属性访问
        # 总体均值 m（根据当前 phi 和固定均值计算）
        self.m = (1 - self.phi) * self.μ1 + self.phi * self.μ2

        # 各状态的二阶矩与总体二阶矩和总体方差
        self.x2_1 = self.μ_sq + self.σ1_sq
        self.x2_2 = self.μ_sq + self.σ2_sq
        self.x2_avg = (1 - self.phi) * self.x2_1 + self.phi * self.x2_2
        self.σ_sq = self.x2_avg - self.m ** 2

        # 反馈项与噪声强度的初始值
        weighted_x2 = (
                (1 - self.phi) ** 2 * self.x2_1 ** 2 +
                2 * self.phi * (1 - self.phi) * self.x2_1 * self.x2_2 +
                self.phi ** 2 * self.x2_2 ** 2
        )
        self.F = self.ρ_d * self.σ_d ** 2 + 2 * self.σ_e ** 2 * weighted_x2

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

        # 存储迭代历史
        self.history = {
            'phi': [],
            'σ1_sq': [],
            'σ2_sq': [],
            'v1': [],
            'v2': [],
            'm': []
        }

    def calculate_overall_moments(self):
        """
        计算总体矩
        """
        # 总体均值 (固定公式)
        self.m = (1 - self.phi) * self.μ1 + self.phi * self.μ2

        # 各状态的二阶矩
        self.x2_1 = self.μ_sq + self.σ1_sq
        self.x2_2 = self.μ_sq + self.σ2_sq

        # 总体二阶矩
        self.x2_avg = (1 - self.phi) * self.x2_1 + self.phi * self.x2_2

        # 总体方差
        self.σ_sq = self.x2_avg - self.m ** 2

    def calculate_feedback_term(self):
        """
        计算反馈项 F
        """
        # 计算加权二阶矩组合
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
        计算各状态的噪声强度
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
        计算有效驱动力
        """
        self.μ_eff = self.μ_c + self.μ_d * self.m + self.μ_e * self.m ** 2

    def update_states(self):
        """
        更新状态参数
        """
        # 计算分母项
        denom1 = 1 - self.v1 * self.F
        denom2 = 1 - self.v2 * self.F

        # 更新状态方差 (均值固定，只更新方差)
        σ1_sq_new = self.D1 / (2 * denom1 ** 2)
        σ2_sq_new = self.D2 / (2 * denom2 ** 2)

        # 更新共存比例
        # 使用状态2的分布计算共存概率
        z = (self.μ_eff / denom2) / np.sqrt(2 * self.D2 / denom2 ** 2)
        phi_new = 0.5 * (1 + erf(z))

        # 更新响应参数
        v1_new = 1 / denom1
        v2_new = 1 / denom2

        return σ1_sq_new, σ2_sq_new, phi_new, v1_new, v2_new

    def solve_iterative(self, verbose=True):
        """
        迭代求解自洽方程组
        """
        self.initialize_parameters()

        for i in range(self.max_iter):
            # 存储当前值
            self.history['phi'].append(self.phi)
            self.history['σ1_sq'].append(self.σ1_sq)
            self.history['σ2_sq'].append(self.σ2_sq)
            self.history['v1'].append(self.v1)
            self.history['v2'].append(self.v2)
            self.history['m'].append(self.m)

            # 计算中间量
            self.calculate_overall_moments()
            self.calculate_feedback_term()
            self.calculate_noise_strengths()
            self.calculate_effective_drive()

            # 更新状态参数
            σ1_sq_new, σ2_sq_new, phi_new, v1_new, v2_new = self.update_states()

            # 应用松弛迭代
            self.σ1_sq = self.relaxation * σ1_sq_new + (1 - self.relaxation) * self.σ1_sq
            self.σ2_sq = self.relaxation * σ2_sq_new + (1 - self.relaxation) * self.σ2_sq
            self.phi = self.relaxation * phi_new + (1 - self.relaxation) * self.phi
            self.v1 = self.relaxation * v1_new + (1 - self.relaxation) * self.v1
            self.v2 = self.relaxation * v2_new + (1 - self.relaxation) * self.v2

            # 确保方差为正
            self.σ1_sq = max(self.σ1_sq, 1e-10)
            self.σ2_sq = max(self.σ2_sq, 1e-10)

            # 限制共存比例在合理范围内
            self.phi = np.clip(self.phi, 0.01, 0.99)

            # 检查收敛
            converged = self.check_convergence()

            if verbose and i % 50 == 0:
                print(f"Iteration {i}: φ = {self.phi:.4f}, σ₁² = {self.σ1_sq:.4f}, σ₂² = {self.σ2_sq:.4f}")

            if converged:
                if verbose:
                    print(f"收敛于第 {i} 次迭代")
                break
        else:
            if verbose:
                print("警告: 达到最大迭代次数但未收敛")

        # 计算最终结果
        self.calculate_overall_moments()

        return self.get_results()

    def check_convergence(self):
        """
        检查收敛条件
        """
        if len(self.history['phi']) < 2:
            return False

        # 计算相对变化
        delta_phi = abs(self.history['phi'][-1] - self.history['phi'][-2])
        delta_σ1 = abs(self.history['σ1_sq'][-1] - self.history['σ1_sq'][-2])
        delta_σ2 = abs(self.history['σ2_sq'][-1] - self.history['σ2_sq'][-2])

        # 检查是否所有参数都收敛
        converged = (
                delta_phi < self.tol and
                delta_σ1 < self.tol and
                delta_σ2 < self.tol
        )

        return converged

    def get_results(self):
        """
        获取求解结果
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
            'history': self.history
        }

    def plot_convergence(self):
        """
        绘制收敛过程
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 共存比例收敛
        axes[0, 0].plot(self.history['phi'])
        axes[0, 0].set_xlabel('迭代次数')
        axes[0, 0].set_ylabel('共存比例 φ')
        axes[0, 0].set_title('共存比例收敛过程')
        axes[0, 0].grid(True, alpha=0.3)

        # 状态方差收敛
        axes[0, 1].plot(self.history['σ1_sq'], label='状态1方差 σ₁²')
        axes[0, 1].plot(self.history['σ2_sq'], label='状态2方差 σ₂²')
        axes[0, 1].set_xlabel('迭代次数')
        axes[0, 1].set_ylabel('状态方差')
        axes[0, 1].set_title('状态方差收敛过程')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 响应参数收敛
        axes[1, 0].plot(self.history['v1'], label='状态1响应 v₁')
        axes[1, 0].plot(self.history['v2'], label='状态2响应 v₂')
        axes[1, 0].set_xlabel('迭代次数')
        axes[1, 0].set_ylabel('响应参数')
        axes[1, 0].set_title('响应参数收敛过程')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 总体均值变化
        axes[1, 1].plot(self.history['m'])
        axes[1, 1].set_xlabel('迭代次数')
        axes[1, 1].set_ylabel('总体均值 m')
        axes[1, 1].set_title('总体均值变化过程')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_bimodal_distribution(self, x_range=(-3, 3), n_points=1000):
        """
        绘制双峰分布
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
        plt.plot(x, gaussian1, 'b--', alpha=0.7,
                 label=f'状态1 (非共存), μ₁={self.μ1:.6f}, σ₁²={self.σ1_sq:.3f}')
        plt.plot(x, gaussian2, 'r--', alpha=0.7,
                 label=f'状态2 (共存), μ₂={self.μ2:.6f}, σ₂²={self.σ2_sq:.3f}')
        plt.plot(x, total_dist, 'k-', linewidth=2,
                 label=f'总体分布, φ={self.phi:.3f}')

        # 标记稳定状态
        plt.axvline(x=self.μ1, color='b', linestyle=':', alpha=0.5, label=f'状态1均值 μ₁={self.μ1:.6f}')
        plt.axvline(x=self.μ2, color='r', linestyle=':', alpha=0.5, label=f'状态2均值 μ₂={self.μ2:.6f}')
        plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5, label='不稳定状态 x=0')

        plt.xlabel('x')
        plt.ylabel('概率密度')
        plt.title('固定均值 (±1.324718) 的双稳态系统概率分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


# 测试代码
if __name__ == "__main__":
    # 参数设置
    params = {
        'μ_c': 0.0,  # 外部输入均值
        'μ_d': 0.5,  # 线性耦合均值
        'μ_e': 0.2,  # 二次耦合均值
        'σ_c': 0.31,  # 外部输入涨落
        'σ_d': 0.3,  # 线性耦合涨落
        'σ_e': 0.1,  # 二次耦合涨落
        'ρ_d': 1.0,  # 线性耦合相关性
        'S': 200,  # 物种数量
        'max_iter': 1000,
        'tol': 1e-6,
        'relaxation': 0.3  # 松弛因子
    }

    # 创建求解器实例
    solver = SpecificBistableCavitySolver(params)

    # 求解自洽方程
    results = solver.solve_iterative(verbose=True)

    # 打印结果
    print("\n固定均值 (±1.324718) 的双稳态系统空穴法求解结果:")
    print(f"总体均值 m = {results['m']:.6f}")
    print(f"总体方差 σ² = {results['σ_sq']:.6f}")
    print(f"共存比例 φ = {results['phi']:.6f}")
    print(f"状态1: μ₁ = {results['μ1']:.6f} (固定), σ₁² = {results['σ1_sq']:.6f}")
    print(f"状态2: μ₂ = {results['μ2']:.6f} (固定), σ₂² = {results['σ2_sq']:.6f}")
    print(f"响应参数: v₁ = {results['v1']:.6f}, v₂ = {results['v2']:.6f}")
    print(f"反馈项 F = {results['F']:.6f}")
    print(f"噪声强度: D₁ = {results['D1']:.6f}, D₂ = {results['D2']:.6f}")

    # 绘制收敛过程
    solver.plot_convergence()

    # 绘制双峰分布
    solver.plot_bimodal_distribution()


# 参数扫描示例
def parameter_scan_specific():
    """
    参数扫描：观察系统行为随线性耦合强度的变化
    """
    μ_d_values = np.linspace(0.1, 2.0, 20)
    phi_values = []
    σ1_sq_values = []
    σ2_sq_values = []
    m_values = []

    for μ_d in μ_d_values:
        params = {
            'μ_c': 0.5, 'μ_d': μ_d, 'μ_e': 0.2,
            'σ_c': 0.1, 'σ_d': 0.3, 'σ_e': 0.1,
            'ρ_d': 1.0, 'S': 500,
            'max_iter': 1000, 'tol': 1e-5, 'relaxation': 0.3
        }

        solver = SpecificBistableCavitySolver(params)
        results = solver.solve_iterative(verbose=False)
        phi_values.append(results['phi'])
        σ1_sq_values.append(results['σ1_sq'])
        σ2_sq_values.append(results['σ2_sq'])
        m_values.append(results['m'])

    # 绘制结果
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 共存比例随μ_d的变化
    axes[0, 0].plot(μ_d_values, phi_values, 'o-', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel('线性耦合强度 μ_d')
    axes[0, 0].set_ylabel('共存比例 φ')
    axes[0, 0].set_title('共存比例随线性耦合强度的变化')
    axes[0, 0].grid(True, alpha=0.3)

    # 两个状态方差随μ_d的变化
    axes[0, 1].plot(μ_d_values, σ1_sq_values, 's-', linewidth=2, markersize=4, label='状态1方差 σ₁²')
    axes[0, 1].plot(μ_d_values, σ2_sq_values, '^-', linewidth=2, markersize=4, label='状态2方差 σ₂²')
    axes[0, 1].set_xlabel('线性耦合强度 μ_d')
    axes[0, 1].set_ylabel('状态方差')
    axes[0, 1].set_title('状态方差随线性耦合强度的变化')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 总体均值随μ_d的变化
    axes[1, 0].plot(μ_d_values, m_values, 'o-', linewidth=2, markersize=4)
    axes[1, 0].set_xlabel('线性耦合强度 μ_d')
    axes[1, 0].set_ylabel('总体均值 m')
    axes[1, 0].set_title('总体均值随线性耦合强度的变化')
    axes[1, 0].grid(True, alpha=0.3)

    # 总体方差随μ_d的变化
    overall_variance = [σ1_sq_values[i] + σ2_sq_values[i] for i in range(len(σ1_sq_values))]
    axes[1, 1].plot(μ_d_values, overall_variance, 'o-', linewidth=2, markersize=4)
    axes[1, 1].set_xlabel('线性耦合强度 μ_d')
    axes[1, 1].set_ylabel('总体方差 σ²')
    axes[1, 1].set_title('总体方差随线性耦合强度的变化')
    axes[1, 1].grid(True, alpha=0.3)

    plt.show()


# 运行参数扫描
print("\n进行参数扫描...")
parameter_scan_specific()

import time
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

# ---------- 辅助函数 ----------
def gaussian_pdf(x, mu, sigma_sq):
    return np.exp(-0.5 * (x - mu)**2 / sigma_sq) / np.sqrt(2 * np.pi * sigma_sq)

def fit_gmm(samples, n_components=2):
    samples = np.asarray(samples).reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
    gmm.fit(samples)
    return gmm

# ---------- 参数生成 ----------
def generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, rng=None):
    """
    生成模型参数示例：
    c_i: 外部输入每个物种的随机偏置（长度 s）
    d_ji: 线性耦合矩阵的一列或行（此处生成一个随机矩阵，但在动力学中只用到列和行的汇总）
    e_ijk: 二次耦合张量的简化表示（这里生成稀疏或简化版本）
    返回用于仿真的参数数组（可根据你的动力学方程调整）
    """
    if rng is None:
        rng = np.random.default_rng()
    # c_i: 每个物种的外部输入，均值 mu_c，方差 sigma_c^2
    c_i = rng.normal(loc=mu_c, scale=sigma_c, size=s)
    # 线性耦合：生成矩阵 D with mean mu_d and std sigma_d, 并构造无自耦合对角为0
    D = rng.normal(loc=mu_d, scale=sigma_d, size=(s, s))
    np.fill_diagonal(D, 0.0)
    # 为保证可控性，这里使用 d_ji = D.T（可以根据真实模型修改）
    d_ij = D.copy()
    d_ji = D.T.copy()
    # 二次耦合：生成稀疏三维张量 e_ijk（这里用较小规模近似：每对 i,j 只有一个 k 成分）
    # 为节省内存，仅生成一个简化形式 e_ijk_sum[i] = sum_jk e_ijk x_j x_k 的系数近似
    # 下面生成对称的简化二次耦合权重矩阵 E_ij，然后在动力学中用 x^T E x 项
    E = rng.normal(loc=mu_e, scale=sigma_e, size=(s, s))
    E = (E + E.T) / 2.0  # 对称化
    return c_i, d_ij, d_ji, E

# ---------- 动力学仿真 (RK4) ----------
def dynamics_derivative(x, c_i, d_ji, E):
    """
    计算系统的时间导数 dx/dt 的示例实现。
    这里使用一个示例性动力学：
    dx_i/dt = -x_i + c_i + (1/s) * sum_j d_ji[j,i] * x_j + x^T E[:,i]  （简化形式）
    你可以改为你的具体模型方程。
    """
    s = x.size
    # 线性耦合项 (1/s) ∑_j d_ji[j,i] x_j 等同于 (d_ji.T @ x)/s
    linear_term = (d_ji.T @ x) / float(s)
    # 二次项：每个 i，取 x^T E[:, i] = ∑_j x_j * E[j,i]
    quad_term = E.T @ x  # 维度 s
    dx = -x + c_i + linear_term + quad_term
    return dx

def rk4_step(x, dt, c_i, d_ji, E):
    k1 = dynamics_derivative(x, c_i, d_ji, E)
    k2 = dynamics_derivative(x + 0.5 * dt * k1, c_i, d_ji, E)
    k3 = dynamics_derivative(x + 0.5 * dt * k2, c_i, d_ji, E)
    k4 = dynamics_derivative(x + dt * k3, c_i, d_ji, E)
    x_new = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x_new

def dynamics_simulation(s, c_i, d_ji, E, x_init, t_steps, dt=0.01, record_every=10):
    """
    运行 RK4 仿真，返回最终状态和历时保存（若需要）
    record_every: 每多少步记录一次以节省内存
    """
    x = x_init.copy().astype(float)
    traj = []
    for t in range(t_steps):
        x = rk4_step(x, dt, c_i, d_ji, E)
        if (t % record_every) == 0:
            traj.append(x.copy())
    traj = np.array(traj)  # shape (~t_steps/record_every, s)
    return x, traj

# ---------- 估算数值 phi（从最终状态） ----------
def estimate_phi_from_final_states(x_final, threshold=0.0):
    """
    使用阈值法估算数值上处于右峰（μ2）占比。
    由于理论峰 μ2 = +1.324718， μ1 = -1.324718，可用 0 作分界。
    threshold 可调整以更接近真实判定。
    """
    # 右峰数量比例
    right_count = np.sum(x_final > threshold)
    phi_num = float(right_count) / x_final.size
    return phi_num

# ---------- 绘图函数 ----------
def plot_final_state_distribution(x_final, bins=100, x_range=(-3, 3)):
    plt.figure(figsize=(8, 5))
    plt.hist(x_final, bins=bins, range=x_range, density=True, alpha=0.6, color='C0', label='数值仿真（直方图）')
    # KDE
    samples = x_final.reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(samples)
    xs = np.linspace(x_range[0], x_range[1], 500).reshape(-1, 1)
    log_dens = kde.score_samples(xs)
    dens = np.exp(log_dens)
    plt.plot(xs[:, 0], dens, 'k-', lw=2, label='KDE（bandwidth=0.1）')
    plt.xlabel('x')
    plt.ylabel('概率密度')
    plt.title('数值仿真最终分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_theory_and_fit(solver, x_samples, x_range=(-3, 3), bins=100, method='kde'):
    """
    将理论双高斯密度（solver 的 σ1_sq, σ2_sq, phi, μ1, μ2）与数值样本
    的 KDE 或 GMM 拟合进行对比。
    method: 'kde' 或 'gmm'
    """
    xs = np.linspace(x_range[0], x_range[1], 500)
    # 理论双高斯
    gauss1 = (1 - solver.phi) * gaussian_pdf(xs, solver.μ1, solver.σ1_sq)
    gauss2 = solver.phi * gaussian_pdf(xs, solver.μ2, solver.σ2_sq)
    theory = gauss1 + gauss2

    plt.figure(figsize=(8, 5))
    plt.plot(xs, theory, 'r-', lw=2, label='理论双高斯 (穴位法)')
    # 样本拟合
    samples = np.asarray(x_samples).reshape(-1, 1)
    if method == 'kde':
        kde = KernelDensity(kernel='gaussian', bandwidth=0.12).fit(samples)
        log_dens = kde.score_samples(xs.reshape(-1, 1))
        dens = np.exp(log_dens)
        plt.plot(xs, dens, 'k--', lw=2, label='数值样本 KDE')
    else:
        # 用 GMM 拟合 2 个分量
        gmm = fit_gmm(samples, n_components=2)
        logprob = gmm.score_samples(xs.reshape(-1, 1))
        dens = np.exp(logprob)
        plt.plot(xs, dens, 'k--', lw=2, label='数值样本 GMM')
        # 标出 GMM 的均值/权重
        means = gmm.means_.flatten()
        covs = np.array([gmm.covariances_[i].flatten()[0] for i in range(gmm.n_components)])
        weights = gmm.weights_.flatten()
        for i in range(gmm.n_components):
            plt.axvline(means[i], color='C'+str(2+i), linestyle=':', label=f'GMM μ{i}={means[i]:.3f}')
    # 理论峰位置
    plt.axvline(solver.μ1, color='b', linestyle=':', alpha=0.6, label=f'理论 μ1={solver.μ1:.6f}')
    plt.axvline(solver.μ2, color='r', linestyle=':', alpha=0.6, label=f'理论 μ2={solver.μ2:.6f}')

    plt.xlabel('x')
    plt.ylabel('概率密度')
    plt.title('理论双高斯与数值样本拟合比较')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_compare_distribution_two_panel(solver, x_final, x_range=(-3, 3), bins=100):
    xs = np.linspace(x_range[0], x_range[1], 500)
    gauss1 = (1 - solver.phi) * gaussian_pdf(xs, solver.μ1, solver.σ1_sq)
    gauss2 = solver.phi * gaussian_pdf(xs, solver.μ2, solver.σ2_sq)
    theory = gauss1 + gauss2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # 左：直方图 + 理论
    axes[0].hist(x_final, bins=bins, range=x_range, density=True, alpha=0.6, color='C0', label='数值直方图')
    axes[0].plot(xs, theory, 'r-', lw=2, label='理论双高斯')
    axes[0].set_title('直方图与理论')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 右：KDE 与理论
    samples = x_final.reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.12).fit(samples)
    log_dens = kde.score_samples(xs.reshape(-1, 1))
    dens = np.exp(log_dens)
    axes[1].plot(xs, dens, 'k--', lw=2, label='数值 KDE')
    axes[1].plot(xs, theory, 'r-', lw=2, label='理论双高斯')
    axes[1].set_title('KDE 与理论')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ---------- 在脚本末尾加入主流程（保持你原有的迭代测试不变） ----------
# 下面是新增的数值仿真与比较流程，请将其加入到你已有的 if __name__ == "__main__": 块之后或合并同一块中
# 我将直接给出一个示例调用，使用你在问题中提供的参数。

def run_simulation_and_compare():
    # 数值仿真参数
    s = 2000
    mu_c = 0.0
    sigma_c = 0.4
    mu_d = 0.2
    sigma_d = 0.3
    rho_d = 1.0
    mu_e = 0.2
    sigma_e = 0.1

    rng = np.random.default_rng(42)
    c_i, d_ij, d_ji, E = generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, rng=rng)
    x_init = np.full(s, 0.6)
    t_steps = 3000
    dt = 0.01

    print("\n开始数值仿真（RK4）...")
    t0 = time.time()
    x_final, traj = dynamics_simulation(s, c_i, d_ji, E, x_init, t_steps, dt=dt, record_every=10)
    t1 = time.time()
    print(f"数值仿真完成，耗时 {(t1 - t0):.2f} 秒")

    # 绘制最终分布并估算 phi
    plot_final_state_distribution(x_final)
    phi_num = estimate_phi_from_final_states(x_final)
    print(f"数值仿真 φ_num(final) = {phi_num:.4f}")

    # 用当前 solver（如果已执行过 solve_iterative 并保存在变量 solver）进行理论对比
    try:
        # 假设之前创建了变量 solver（SpecificBistableCavitySolver 或 BistableCavitySolver）
        # 若没有，请创建一个一个与仿真参数相近的 solver 实例并求解
        global solver  # 仅在交互式环境可用；否则请传入 solver 实例
        print("进行理论（穴位法）与数值样本比较...")
        plot_theory_and_fit(solver, x_final, x_range=(-3, 3), method='kde')
        plot_compare_distribution_two_panel(solver, x_final, x_range=(-3, 3))
    except Exception as e:
        print("无法直接访问 solver 变量，尝试创建新的求解器用于比较：", e)
        params_tmp = {
            'μ_c': mu_c, 'μ_d': mu_d, 'μ_e': mu_e,
            'σ_c': sigma_c, 'σ_d': sigma_d, 'σ_e': sigma_e,
            'ρ_d': rho_d, 'S': s,
            'max_iter': 2000, 'tol': 1e-6, 'relaxation': 0.3
        }
        solver_tmp = SpecificBistableCavitySolver(params_tmp)
        solver_tmp.solve_iterative(verbose=False)
        plot_theory_and_fit(solver_tmp, x_final, x_range=(-3, 3), method='kde')
        plot_compare_distribution_two_panel(solver_tmp, x_final, x_range=(-3, 3))
    print("仿真与比较完成。")

# 如果需要，可在主程序末尾调用 run_simulation_and_compare()
run_simulation_and_compare()