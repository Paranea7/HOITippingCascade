import numpy as np
import matplotlib.pyplot as plt

# ---------- 参数 ----------
a = b = 1
c_crit = 2/(3*np.sqrt(3))
d12 = 0.2;  d21 = 0.0          # 主-从
Ngrid = 200                   # 网格分辨率
c1_arr = np.linspace(-0.5, 0.8, Ngrid)
c2_arr = np.linspace(-0.5, 0.8, Ngrid)
stab_map = np.zeros((Ngrid, Ngrid), dtype=int)

# ---------- 工具函数 ----------
def fixed_points_and_stability(c1, c2):
    """返回稳定点列表 [[x1,x2], ...] """
    # 初值网格
    x1s = np.linspace(-2,2,7); x2s = np.linspace(-2,2,7)
    roots = []
    for x10 in x1s:
        for x20 in x2s:
            x1,x2 = x10,x20
            for it in range(50):
                f1 = -x1**3 + x1 + c1 + d21*x2
                f2 = -x2**3 + x2 + c2 + d12*x1
                J11 = -3*x1**2 + 1;  J12 = d21
                J21 = d12;          J22 = -3*x2**2 + 1
                detJ = J11*J22 - J12*J21
                if abs(detJ)<1e-14: break
                dx1 = ( J22*f1 - J12*f2)/detJ
                dx2 = (-J21*f1 + J11*f2)/detJ
                x1 -= dx1; x2 -= dx2
                if np.hypot(dx1,dx2)<1e-10: break
            if np.hypot(f1,f2)<1e-8:
                duplicate = any([np.hypot(x1-r[0],x2-r[1])<1e-2 for r in roots])
                if not duplicate: roots.append([x1,x2])
    # 稳定性判别
    stable = []
    for x1,x2 in roots:
        J = np.array([[-3*x1**2+1, d21],
                      [d12,       -3*x2**2+1]])
        lam = np.linalg.eigvals(J)
        if np.all(np.real(lam)<0): stable.append([x1,x2])
    return stable

# ---------- 主循环：算稳定点个数 ----------
print('开始计算稳定性矩阵...')
for i, c1 in enumerate(c1_arr):
    for j, c2 in enumerate(c2_arr):
        stab_map[i,j] = len(fixed_points_and_stability(c1, c2))
    if i%20==0: print('row',i,'done')

# ---------- 左半幅：彩色网格 ----------
fig = plt.figure(figsize=(10,4.5))
ax1 = fig.add_subplot(121)
im = ax1.imshow(stab_map, extent=(c1_arr[0], c1_arr[-1], c2_arr[0], c2_arr[-1]),
                origin='lower', cmap='viridis', aspect='auto')
# 画单单元临界线（白色虚线）
ax1.axhline( c_crit, color='w', ls='--', lw=1)
ax1.axhline(-c_crit, color='w', ls='--', lw=1)
ax1.axvline( c_crit, color='w', ls='--', lw=1)
ax1.axvline(-c_crit, color='w', ls='--', lw=1)
ax1.set_xlabel(r'$c_1$'); ax1.set_ylabel(r'$c_2$')
ax1.set_title('Stable equilibria number (d12=0.2, d21=0)')

# ---------- 右半幅：4 个相空间图 ----------
pts = [(-0.2, -0.2), (0.2, -0.2), (0.2, 0.2), (0.5, 0.2)]
titles = ['4 fixed pts', '3 fixed pts', '2 fixed pts', '1 fixed pt']

for idx, ((c1,c2),title) in enumerate(zip(pts, titles)):
    ax = fig.add_subplot(2, 4, 5+idx)
    # 画流线
    X1, X2 = np.meshgrid(np.linspace(-1.5,1.5,30), np.linspace(-1.5,1.5,30))
    U = -X1**3 + X1 + c1 + d21*X2
    V = -X2**3 + X2 + c2 + d12*X1
    ax.streamplot(X1, X2, U, V, color='grey', linewidth=0.5, density=1.5)
    # 画稳定/不稳定点
    pts_all = fixed_points_and_stability(c1,c2)
    for x1,x2 in pts_all:
        ax.plot(x1, x2, 'o', color='orange', markersize=8)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-1.2,1.2); ax.set_ylim(-1.2,1.2)
    ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')

plt.tight_layout()
plt.savefig('fig5_replica.png', dpi=400)
plt.show()