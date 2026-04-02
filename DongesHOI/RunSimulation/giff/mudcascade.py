import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# 1. 物理参数配置 (线性耦合版本)
# ==========================================
S = 50            # 节点数量
MU_D = 0.8        # 线性耦合强度 (线性相互作用)
MU_U = 0.45       # 外部场/偏置 (必须大于 H_c ~ 0.385 以触发从 -1 开始的级联)
SIG_U = 0.128     # 噪声强度
DT = 0.05         # 时间步长
STEPS = 1000      # 动画帧数

# 初始状态：所有节点处于负分支 -1 附近
x = np.full(S, -1.0) + np.random.normal(0, 0.05, S)
# 外部输入噪声 (Quenched Noise)
xi = np.random.normal(0, SIG_U, S)

# 网络节点布局 (圆形布局展示全连网络)
theta = np.linspace(0, 2 * np.pi, S, endpoint=False)
pos_x = np.cos(theta)
pos_y = np.sin(theta)

# ==========================================
# 2. 图形界面初始化
# ==========================================
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121)  # 左侧：网络快照
ax2 = fig.add_subplot(122)  # 右侧：序参量 m(t) 演化

# 绘制背景连线 (代表全连接，稀疏显示以保持清晰)
for i in range(0, S, 5):
    for j in range(i + 1, S, 10):
        ax1.plot([pos_x[i], pos_x[j]], [pos_y[i], pos_y[j]], color='gray', alpha=0.05, lw=0.5)

scatter = ax1.scatter(pos_x, pos_y, c=x, cmap='coolwarm', s=50, edgecolors='k', zorder=3, vmin=-1.2, vmax=1.2)
ax1.set_title(r"Network Cascade ($\mu_d m + \mu_u$ Drive)")
ax1.axis('off')

# 时间序列曲线初始化
m_history = []
line, = ax2.plot([], [], lw=2, color='#0072B2')
ax2.set_xlim(0, STEPS)
ax2.set_ylim(-1.2, 1.2)
ax2.set_xlabel("Time Steps")
ax2.set_ylabel("Order Parameter $m(t)$")
ax2.set_title("Global Dynamics (Linear Feedback)")
ax2.grid(ls=':', alpha=0.6)

# ==========================================
# 3. 动画更新核心逻辑
# ==========================================
def update(frame):
    global x
    m = np.mean(x)

    # 线性动力学方程: dx = x - x^3 + mu_d * m + mu_u + xi
    # 这里的 mu_d * m 提供了线性协同反馈
    drift = x - x**3 + MU_D * m + MU_U + xi
    x += drift * DT
    x = np.clip(x, -2, 2)

    # 更新节点颜色
    scatter.set_array(x)

    # 更新序参量曲线
    m_history.append(m)
    line.set_data(range(len(m_history)), m_history)

    # 级联触发后的视觉提示 (当 m 越过势垒中点 0 时)
    if m > 0:
        ax2.set_facecolor('#f0f7ff')  # 变为淡蓝色背景

    return scatter, line

# 创建动画
ani = animation.FuncAnimation(fig, update, frames=STEPS, interval=30, blit=True)

plt.tight_layout()
plt.show()
