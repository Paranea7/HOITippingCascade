import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# 1. 核心物理参数 (严格对齐你的设置)
# ==========================================
S = 50  # 系统规模
DT = 0.02  # 时间步长
STEPS = 1400  # 模拟帧数
H_C = 0.3849  # 势垒消失临界阈值

# --- 相互作用参数 ---
MU_D, SIG_D = 0.1, 0.0  # 线性耦合 (mu_d, sigma_d)
MU_E, SIG_E = 8, 0.0  # 二阶耦合 (mu_e, sigma_e)
MU_U, SIG_U = 0.0, 1.0 # 外部输入 (mu_u, sigma_u)

# ==========================================
# 2. 参数生成与种子初始化 (Seeded Initialization)
# ==========================================
# 1. 生成外部输入 u_i
u = np.random.normal(MU_U, SIG_U, S)

# 2. 生成耦合矩阵 (遵循 Scaling 规则)
d_ji = np.random.normal(MU_D / S, SIG_D / np.sqrt(S), (S, S))
np.fill_diagonal(d_ji, 0)

e_ijk = np.random.normal(MU_E / (S ** 2), SIG_E / S, (S, S, S))
for i in range(S):
    e_ijk[i, i, :] = 0;
    e_ijk[i, :, i] = 0;
    e_ijk[:, i, i] = 0

# 3. 核心修改：种子初始化
# 找出天然失稳的节点 (u_i > H_c)
seeds_mask = u > H_C
x = np.full(S, -1.0) + np.random.normal(0, 0.02, S)  # 默认在负分支
x[seeds_mask] = 1.0  # 种子强制设为正分支

print(f"Total nodes: {S} | Initial seeds at +1: {np.sum(seeds_mask)}")

# ==========================================
# 3. 动画界面设置
# ==========================================
fig = plt.figure(figsize=(14, 6), dpi=120)
ax_net = fig.add_subplot(121)
ax_dyn = fig.add_subplot(122)

# 圆形布局
theta = np.linspace(0, 2 * np.pi, S, endpoint=False)
pos_x, pos_y = np.cos(theta), np.sin(theta)

scatter = ax_net.scatter(pos_x, pos_y, c=x, cmap='coolwarm', s=70,
                         edgecolors='k', linewidths=0.8, vmin=-1.2, vmax=1.2, zorder=3)
ax_net.set_title(r"Seeded Cascade Evolution ($u_i > H_c \to +1$)")
ax_net.axis('off')

m_hist = []
line, = ax_dyn.plot([], [], lw=2, color='firebrick')
ax_dyn.set_xlim(0, STEPS)
ax_dyn.set_ylim(-1.1, 1.1)
ax_dyn.set_xlabel("Time Steps")
ax_dyn.set_ylabel("Order Parameter $m(t)$")
ax_dyn.set_title("Global Mean-field Dynamics")
ax_dyn.grid(ls=':', alpha=0.6)


# ==========================================
# 4. 动力学演化
# ==========================================
def update(frame):
    global x
    # 计算总场：F_i = u_i + sum(d_ji*x_j) + sum(e_ijk*x_j*x_k)
    linear_field = d_ji @ x
    higher_order_field = np.einsum('ijk,j,k->i', e_ijk, x, x)
    F_i = u + linear_field + higher_order_field

    # 演化方程
    dx = (x - x ** 3 + F_i)
    x += dx * DT
    x = np.clip(x, -2, 2)

    # 更新数据
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
