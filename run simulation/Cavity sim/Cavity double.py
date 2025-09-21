import numpy as np
from scipy.special import erf
from scipy.integrate import quad


def bimodal_cavity_method(params, max_iter=1000, tol=1e-6):
    """
    双峰分布空穴法计算
    """
    # 解析参数
    μ_c, μ_d, μ_e = params['μ_c'], params['μ_d'], params['μ_e']
    σ_c, σ_d, σ_e = params['σ_c'], params['σ_d'], params['σ_e']
    ρ_d = params.get('ρ_d', 0.0)

    # 初始猜测
    phi = 0.5  # 初始共存比例
    v = 1.0  # 初始响应参数
    μ1, μ2 = -1.0, 1.0  # 两个状态的均值
    σ1_sq, σ2_sq = 0.1, 0.1  # 两个状态的方差

    for i in range(max_iter):
        # 计算总体矩
        m = (1 - phi) * μ1 + phi * μ2
        x2_avg = (1 - phi) * (μ1 ** 2 + σ1_sq) + phi * (μ2 ** 2 + σ2_sq)
        σ_sq = x2_avg - m ** 2

        # 计算噪声强度
        D = σ_c ** 2 + σ_d ** 2 * x2_avg + σ_e ** 2 * x2_avg ** 2

        # 计算分母项
        denominator = 1 - v * (ρ_d * σ_d ** 2 * phi + σ_e ** 2 * phi ** 2 * (μ2 ** 2 + σ2_sq))

        # 计算有效驱动力
        μ_eff = μ_c + μ_d * m + μ_e * m ** 2

        # 更新两个状态的均值和方差
        μ1_new = μ_eff / denominator
        μ2_new = μ_eff / denominator
        σ1_sq_new = D / (2 * denominator ** 2)
        σ2_sq_new = D / (2 * denominator ** 2)

        # 更新共存比例
        # 使用数值积分计算状态2的概率
        def integrand(x):
            return np.exp(-(x - μ_eff / denominator) ** 2 / (2 * D / denominator ** 2)) / np.sqrt(
                2 * np.pi * D / denominator ** 2)

        # 计算状态2的概率（x > 0的区域）
        phi_new, _ = quad(integrand, 0, np.inf)

        # 更新响应参数
        v_new = 1 / denominator

        # 检查收敛
        if (abs(phi_new - phi) < tol and abs(v_new - v) < tol and
                abs(μ1_new - μ1) < tol and abs(μ2_new - μ2) < tol and
                abs(σ1_sq_new - σ1_sq) < tol and abs(σ2_sq_new - σ2_sq) < tol):
            print(f"收敛于第 {i} 次迭代")
            break

        phi, v, μ1, μ2, σ1_sq, σ2_sq = phi_new, v_new, μ1_new, μ2_new, σ1_sq_new, σ2_sq_new
    else:
        print("警告: 达到最大迭代次数但未收敛")

    # 计算最终总体矩
    m = (1 - phi) * μ1 + phi * μ2
    x2_avg = (1 - phi) * (μ1 ** 2 + σ1_sq) + phi * (μ2 ** 2 + σ2_sq)
    σ_sq = x2_avg - m ** 2

    return m, σ_sq, phi, v, μ1, μ2, σ1_sq, σ2_sq


# 测试双峰分布空穴法
params = {
    'μ_c': 0.0, 'μ_d': 0.3, 'μ_e': 0.0,
    'σ_c': 2*np.sqrt(3)/9, 'σ_d': 0.4, 'σ_e': 0.0,
    'ρ_d': 1.0
}

m, σ_sq, phi, v, μ1, μ2, σ1_sq, σ2_sq = bimodal_cavity_method(params)
print(f"双峰分布结果:")
print(f"总体均值 m = {m:.4f}, 总体方差 σ² = {σ_sq:.4f}")
print(f"共存比例 φ = {phi:.4f}, 响应参数 v = {v:.4f}")
print(f"状态1: μ₁ = {μ1:.4f}, σ₁² = {σ1_sq:.4f}")
print(f"状态2: μ₂ = {μ2:.4f}, σ₂² = {σ2_sq:.4f}")