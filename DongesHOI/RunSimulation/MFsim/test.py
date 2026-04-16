import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count


# ==========================================
# 1. 微观动力学核心 (完全基于 S 维交互)
# ==========================================
def micro_dynamic_step(params):
    s, mu_d, mu_e, sig_d, sig_e, t_steps = params
    dt = 0.05
    sig_u = 0.12  # 外部噪声强度

    # --- A. 杂乱初始化 (Disordered Init) ---
    # 一半节点在 -1，一半在 1 -> m=0, q=1
    x = np.ones(s)
    x[:s // 2] = -1.0
    x += np.random.normal(0, 0.05, s)  # 加入微小扰动

    # --- B. 生成微观参数矩阵 ---
    # 外部偏置
    u_i = np.random.normal(0, sig_u, s)
    # 线性耦合 d_ji ~ N(mu_d/S, sig_d^2/S)
    d_ji = np.random.normal(mu_d / s, sig_d / np.sqrt(s), (s, s))
    np.fill_diagonal(d_ji, 0)
    # 三体耦合 e_ijk ~ N(mu_e/S^2, sig_e^2/S^2)
    # 注意：这是最真实的微观形式，天然对应理论中的 q
    e_ijk = np.random.normal(mu_e / (s ** 2), sig_e / s, (s, s, s))

    m_history = []

    # --- C. 演化循环 ---
    for t in range(t_steps):
        # 微观交互项计算 (Tensor Contraction)
        # sum_{j,k} e_ijk * x_j * x_k
        hoi_term = np.einsum('ijk,j,k->i', e_ijk, x, x)
        linear_term = d_ji @ x

        # 演化方程
        drift = x - x ** 3 + u_i + linear_term + hoi_term
        x += drift * dt

        # 限制范围
        x = np.clip(x, -2.5, 2.5)
        m_history.append(np.mean(x))

    return m_history, x


# ==========================================
# 2. 实验对比与绘图
# ==========================================
def main():
    S = 100  # 节点数
    t_steps = 3000  # 演化步数
    sig_d = 0.0
    sig_e = 0.0

    # 测试不同的 mu_e (三体强度)
    # 如果系统是 q 驱动，mu_e > 0 会打破 m=0 的平衡
    mu_e_list = [0.0, 0.2, 0.5]
    mu_d = -0.2  # 稍微给一点负向线性耦合，看三体项能否克服它

    plt.figure(figsize=(10, 6))

    print(f"Starting Disordered Initialization Test (S={S})...")

    for me in mu_e_list:
        print(f"Simulating mu_e = {me}...")
        m_hist, final_x = micro_dynamic_step((S, mu_d, me, sig_d, sig_e, t_steps))

        plt.plot(m_hist, label=rf'$\mu_e = {me}, \mu_d = {mu_d}$')

    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title(r"Disordered Initialization: $m(0) \approx 0, q(0) \approx 1$")
    plt.xlabel("Time Steps")
    plt.ylabel("Mean State $m$")
    plt.legend()
    plt.grid(alpha=0.3)

    # 结论提示
    print("\n--- 判别标准 ---")
    print("1. 如果 m 迅速从 0 升向 1: 说明微观动力学受 q 驱动 (因为此时 m^2=0，只有 q=1 能提供推力)")
    print("2. 如果 m 长期保持在 0 附近: 说明微观动力学更符合 m^2 逻辑 (场强随 m=0 而消失)")

    plt.show()


if __name__ == "__main__":
    main()
