import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# 1. 物理参数配置 (严格微观定义)
# ==========================================
S = 10  # 系统规模 (节点数)
DT = 0.02  # 时间步长
STEPS = 1400  # 模拟帧数
H_C = 0.3849  # 负分支消失临界阈值

# 相互作用均值与标准差
MU_D, SIG_D = 0.1, 0.0  # 线性耦合 (mu_d, sigma_d)
MU_E, SIG_E = 0.8, 0.0  # 二阶耦合 (mu_e, sigma_e)
MU_U, SIG_U = 0.0, 0.5  # 外部输入 (mu_u, sigma_u)

# ==========================================
# 2. 微观参数初始化与“种子”逻辑
# ==========================================
# 1. 生成外部场 u_i
u = np.random.normal(MU_U, SIG_U, S)

# 2. 种子初始化：若 u_i 已经让负阱消失，则初始位置设为 +1
x = np.full(S, -1.0) + np.random.normal(0, 0.02, S)
seeds_mask = u > H_C
x[seeds_mask] = 1.0
print(f"Initialization: {np.sum(seeds_mask)} nodes set as seeds at +1.0")

# 3. 生成耦合参数 (严格遵循 S 缩放规则)
d_ji = np.random.normal(MU_D / S, SIG_D / np.sqrt(S), (S, S))
np.fill_diagonal(d_ji, 0)

e_ijk = np.random.normal(MU_E / (S ** 2), SIG_E / S, (S, S, S))
for i in range(S):  # 移除自反馈项
    e_ijk[i, i, :] = 0;
    e_ijk[i, :, i] = 0;
    e_ijk[:, i, i] = 0

# ==========================================
# 3. 动画与可视化设置
# ==========================================
fig = plt.figure(figsize=(14, 6), dpi=120)
ax_net = fig.add_subplot(121)
ax_dyn = fig.add_subplot(122)

theta = np.linspace(0, 2 * np.pi, S, endpoint=False)
pos_x, pos_y = np.cos(theta), np.sin(theta)

scatter = ax_net.scatter(pos_x, pos_y, c=x, cmap='coolwarm', s=70,
                         edgecolors='k', linewidths=0.8, vmin=-1.2, vmax=1.2, zorder=3)
ax_net.set_title(r"Seeded Micro-dynamics Cascade ($u_i > H_c \to +1$)")
ax_net.axis('off')

m_hist = []
line, = ax_dyn.plot([], [], lw=2, color='firebrick')
ax_dyn.set_xlim(0, STEPS)
ax_dyn.set_ylim(-1.1, 1.1)
ax_dyn.set_xlabel("Time Steps")
ax_dyn.set_ylabel("Order Parameter $m(t)$")
ax_dyn.set_title("Global Trajectory (No Mean-Field Approximation)")
ax_dyn.grid(ls=':', alpha=0.6)


# ==========================================
# 4. 动力学演化 (直接计算交互场)
# ==========================================
def update(frame):
    global x
    # 严格微观计算：不使用 m 或 m^2，直接对每个节点求和
    # 线性项：sum_j d_ji * x_j
    linear_field = d_ji @ x
    # 二阶项：sum_{j,k} e_ijk * x_j * x_k (使用张量收缩)
    higher_order_field = np.einsum('ijk,j,k->i', e_ijk, x, x)

    # 总有效场
    F_total = u + linear_field + higher_order_field

    # 演化方程: dx = (x - x^3 + F_total) * dt
    dx = (x - x ** 3 + F_total)
    x += dx * DT
    x = np.clip(x, -2, 2)

    m = np.mean(x)
    scatter.set_array(x)
    m_hist.append(m)
    line.set_data(range(len(m_hist)), m_hist)

    if m > 0:
        ax_dyn.set_facecolor('#fffafa')

    return scatter, line


ani = animation.FuncAnimation(fig, update, frames=STEPS, interval=30, blit=True)
plt.tight_layout()
plt.show()
