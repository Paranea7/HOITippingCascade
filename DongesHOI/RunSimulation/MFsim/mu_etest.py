import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf


# --- 核心物理函数 (加入鲁棒性) ---
def get_m_out(m_in, mu_e, mu_d=0.2, mu_u=0.0, sigma=0.2, h_c=0.3849):
    q = 1.0
    # 内部自洽迭代 q
    for _ in range(20):
        # M 必须包含 m_in，否则曲线只是平移
        M = mu_u + mu_d * m_in + mu_e * q
        gamma = np.sqrt(sigma ** 2 + 0.1 * q)  # 简化 Gamma 模型以突出 M
        phi = 0.5 * (1 + erf((M - h_c) / (np.sqrt(2) * gamma)))
        q_new = np.clip(1 + (2 * phi - 1) * M + M ** 2 / 4, 0, 3)
        if np.abs(q_new - q) < 1e-5: break
        q = q_new

    # 返回 m_out 公式
    return (2 * phi - 1 + (mu_u + mu_e * q) / 2) / (1 - mu_d / 2)


# --- 绘图逻辑 ---
m_range = np.linspace(-1.5, 2.0, 500)
mu_e_list = [0.0, 0.4, 0.8, 1.2]  # 逐渐增加三体项
colors = plt.cm.viridis(np.linspace(0, 1, len(mu_e_list)))

plt.figure(figsize=(7, 6), dpi=120)

for i, me in enumerate(mu_e_list):
    m_outs = [get_m_out(mi, mu_e=me) for mi in m_range]
    plt.plot(m_range, m_outs, color=colors[i], label=rf"$\mu_e = {me}$", lw=2)

# 绘制 y = x 辅助线
plt.plot([-1.5, 2], [-1.5, 2], 'k--', alpha=0.6, label="$m_{out} = m_{in}$")

plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5)
plt.title("Effect of Three-body Interaction $\mu_e$ on System Bifurcation", fontsize=11)
plt.xlabel("$m_{in}$ (Input state)")
plt.ylabel("$m_{out}$ (Output state)")
plt.legend(frameon=False)
plt.grid(True, ls=':', alpha=0.4)
plt.xlim(-1.5, 2.0)
plt.ylim(-1.5, 2.0)
plt.show()