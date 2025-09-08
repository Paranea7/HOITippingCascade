import numpy as np
from scipy.integrate import quad
from scipy.special import erf

def cavity_method(params, max_iter=1000, tol=1e-6):
    """
    空穴法自洽迭代计算稳态解
    参数：
        params : dict - 包含模型参数的字典
        max_iter : int - 最大迭代次数
        tol : float - 收敛容差
    返回：
        m, sigma2, phi, v : float - 均值、方差、共存比例、响应参数
    """
    required_keys = ['μ_c', 'μ_d', 'μ_e', 'σ_c', 'σ_d', 'σ_e']
    for key in required_keys:
        if key not in params:
            raise ValueError(f"Missing required parameter: {key}")

    # 解析参数
    μ_c = params['μ_c']
    μ_d = params['μ_d']
    μ_e = params['μ_e']
    σ_c = params['σ_c']
    σ_d = params['σ_d']
    σ_e = params['σ_e']
    S = params.get('S', np.inf)  # 默认为无穷大系统
    ρ_d = params.get('ρ_d', 0.0)  # 线性耦合相关性

    # 初始猜测值
    m = μ_c / (1 - μ_d) if μ_d < 1 else 0.5
    sigma2 = 0.2
    phi = 2.0
    v = 2.0

    for i in range(max_iter):
        finite_S_corr = μ_d / S if S < np.inf else 0.0

        # 计算有效噪声强度D
        D = σ_c**2 + σ_d**2 * phi * sigma2 + σ_e**2 * (phi * sigma2)**2

        # 计算分母项
        denominator = (1 - v * (ρ_d * σ_d**2 * phi + σ_e**2 * phi**2 * sigma2**2) + finite_S_corr)

        μ0 = (μ_c + μ_d * phi * m + μ_e * phi**2 * m**2 - finite_S_corr * m) / denominator
        σ0 = np.sqrt(D) / denominator

        # 计算共存比例phi
        z = μ0 / (σ0 * np.sqrt(2))
        phi_new = 0.5 * (1 + erf(z))

        # 计算均值m和方差sigma2
        def integrand(x):
            return x * np.exp(-(x - μ0)**2 / (2 * σ0**2))

        def integrand2(x):
            return x**2 * np.exp(-(x - μ0)**2 / (2 * σ0**2))

        m_num, _ = quad(integrand, 0, np.inf)
        m_new = m_num / (phi_new * σ0 * np.sqrt(2 * np.pi))

        m2, _ = quad(integrand2, 0, np.inf)
        sigma2_new = m2 / (phi_new * σ0 * np.sqrt(2 * np.pi)) - m_new**2

        # 更新响应参数v
        v_new = 1 / denominator

        # 检查收敛
        if (abs(m_new - m) < tol and abs(sigma2_new - sigma2) < tol and abs(phi_new - phi) < tol):
            print(f"Converged at iteration {i}")
            return m_new, sigma2_new, phi_new, v_new

        # 更新变量
        m, sigma2, phi, v = m_new, sigma2_new, phi_new, v_new

    print("Warning: Reached max iterations without convergence")
    return m, sigma2, phi, v

# 示例参数设置
params = {
    'μ_c': 0.0,
    'μ_d': 1.0,
    'μ_e': 0.2,
    'σ_c': 1.0,
    'σ_d': 1.0,
    'σ_e': 0.0,
    'S': 50,
    'ρ_d': 1.0
}

# 运行计算
m, sigma2, phi, v = cavity_method(params)

# 输出结果
print(f"平均丰度 m = {m:.4f}")
print(f"丰度方差 σ² = {sigma2:.4f}")
print(f"共存比例 φ = {phi:.4f}")
print(f"响应参数 v = {v:.4f}")

# 可视化参数扫描示例
import matplotlib.pyplot as plt
σ_d_values = np.linspace(0.1, 1.0, 20)
phi_values = []
for σ_d in σ_d_values:
    params['σ_d'] = σ_d
    _, _, phi, _ = cavity_method(params)
    phi_values.append(phi)

plt.plot(σ_d_values, phi_values, 'o-')
plt.xlabel('σ_d (Linear Interaction Strength)')
plt.ylabel('Coexistence Fraction φ')
plt.title('Phase Transition with Increasing σ_d')
plt.grid(True)
plt.show()