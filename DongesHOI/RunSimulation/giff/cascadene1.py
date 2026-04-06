import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# 1. 核心物理参数 (包含所有理论变量)
# ==========================================
S = 50  # 系统规模 (节点数)
DT = 0.02  # 时间步长
STEPS = 1400  # 模拟帧数

# --- 线性项 (Linear 2-body) ---
MU_D = 0.1  # 线性耦合均值 (mu_d)
SIG_D = 0.0  # 线性耦合标准差 (sigma_d)

# --- 二阶项 (2nd-order 3-body) ---
MU_E = 0.5  # 二阶耦合均值 (mu_e)
SIG_E = 0.0  # 二阶耦合标准差 (sigma_e)

# --- 外部输入 (External Input) ---
MU_U = 0.386  # 外部场均值 (mu_u)
SIG_U = 0.128  # 外部场噪声强度 (sigma_u)

# ==========================================
# 2. 参数初始化 (严格遵循 Scaling 规则)
# ==========================================
# 外部输入 u_i ~ N(mu_u, sig_u^2)
u = np.random.normal(MU_U, SIG_U, S)

# 线性耦合 d_ji ~ N(mu_d/S, sig_d^2/S)
d_ji = np.random.normal(MU_D / S, SIG_D / np.sqrt(S), (S, S))
np.fill_diagonal(d_ji, 0)  # 移除自反馈

# 二阶耦合 e_ijk ~ N(mu_e/S^2, sig_e^2/S^2)
e_ijk = np.random.normal(MU_E / (S ** 2), SIG_E / S, (S, S, S))
# 移除冗余相互作用 (i=j, i=k, j=k)
for i in range(S):
    e_ijk[i, i, :] = 0
    e_ijk[i, :, i] = 0
    e_ijk[:, i, i] = 0

# 初始状态：全员处于负分支 -1 附近
x = np.full(S, -1.0) + np.random.normal(0, 0.02, S)

# ==========================================
# 3. 动画与绘图设置
# ==========================================
fig = plt.figure(figsize=(14, 6), dpi=120)
ax_net = fig.add_subplot(121)  # 网络视图
ax_dyn = fig.add_subplot(122)  # 序参量视图

# 节点坐标 (圆形布局)
theta = np.linspace(0, 2 * np.pi, S, endpoint=False)
pos_x, pos_y = np.cos(theta), np.sin(theta)

# 初始化散点图
scatter = ax_net.scatter(pos_x, pos_y, c=x, cmap='coolwarm', s=60,
                         edgecolors='k', linewidths=0.5, vmin=-1.2, vmax=1.2, zorder=3)
ax_net.set_title(r"Micro-state Cascade ($S=%d$)" % S)
ax_net.axis('off')

# 初始化时间曲线
m_hist = []
line, = ax_dyn.plot([], [], lw=2, color='firebrick')
ax_dyn.set_xlim(0, STEPS)
ax_dyn.set_ylim(-1.2, 1.2)
ax_dyn.set_xlabel("Time Steps")
ax_dyn.set_ylabel("Order Parameter $m(t)$")
ax_dyn.set_title("Global Mean-field Dynamics")
ax_dyn.grid(ls=':', alpha=0.6)


# ==========================================
# 4. 动力学演化更新函数
# ==========================================
def update(frame):
    global x
    # 计算当前总有效场 F_i
    # F_i = u_i + sum(d_ji * x_j) + sum(e_ijk * x_j * x_k)
    linear_field = d_ji @ x
    higher_order_field = np.einsum('ijk,j,k->i', e_ijk, x, x)

    F_i = u + linear_field + higher_order_field

    # 演化方程: dx = (x - x^3 + F_i) * dt
    dx = (x - x ** 3 + F_i)
    x += dx * DT
    x = np.clip(x, -2, 2)

    # 更新视觉组件
    m = np.mean(x)
    scatter.set_array(x)
    m_hist.append(m)
    line.set_data(range(len(m_hist)), m_hist)

    # 翻转提示
    if m > 0:
        ax_dyn.set_facecolor('#fff5f5')

    return scatter, line


ani = animation.FuncAnimation(fig, update, frames=STEPS, interval=40, blit=True)
plt.tight_layout()
plt.show()
