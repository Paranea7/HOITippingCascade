import numpy as np
import matplotlib.pyplot as plt

def find_roots(c1, c2, d12):
    coefficients1 = [1, 0, -1, -c1]
    roots1 = np.roots(coefficients1)
    real_roots1 = roots1[np.isreal(roots1)].real

    if real_roots1.size > 0:
        root1 = real_roots1[0]
    else:
        root1 = 0

    coefficients2_coupled = [1, 0, -1, -(c2 + d12 * root1)]
    roots2_coupled = np.roots(coefficients2_coupled)
    real_roots2_coupled = roots2_coupled[np.isreal(roots2_coupled)].real

    coefficients2_uncoupled = [1, 0, -1, -c2]
    roots2_uncoupled = np.roots(coefficients2_uncoupled)
    real_roots2_uncoupled = roots2_uncoupled[np.isreal(roots2_uncoupled)].real

    return real_roots1, real_roots2_coupled, real_roots2_uncoupled

# 设置参数 c 的范围
c1_values = np.linspace(-2, 2, 200)
c2_values = np.linspace(-4, 4, 200)

# 耦合强度
d12_pos = 0.2
d12_neg = -0.2

# 存储每个 c 对应的实数根
fixed_points1_pos = []
fixed_points2_coupled_pos = []
fixed_points2_uncoupled_pos = []

fixed_points1_neg = []
fixed_points2_coupled_neg = []
fixed_points2_uncoupled_neg = []

# 计算耦合强度为正和负时的不动点
for c1 in c1_values:
    for c2 in c2_values:
        roots1_pos, roots2_coupled_pos, roots2_uncoupled_pos = find_roots(c1, c2, d12_pos)
        #roots1_neg, roots2_coupled_neg, roots2_uncoupled_neg = find_roots(c1, c2, d12_neg)

        for root in roots1_pos:
            fixed_points1_pos.append((c1, root))
        for root in roots2_coupled_pos:
            fixed_points2_coupled_pos.append((c2, root))
        for root in roots2_uncoupled_pos:
            fixed_points2_uncoupled_pos.append((c2, root))

        '''for root in roots1_neg:
            fixed_points1_neg.append((c1, root))
        for root in roots2_coupled_neg:
            fixed_points2_coupled_neg.append((c2, root))
        for root in roots2_uncoupled_neg:
            fixed_points2_uncoupled_neg.append((c2, root))'''


# 将不动点的坐标分开为 x 和 y
c1_points1_pos, x1_points_pos = zip(*fixed_points1_pos) if fixed_points1_pos else ([], [])
c2_points2_coupled_pos, x2_points_coupled_pos = zip(*fixed_points2_coupled_pos) if fixed_points2_coupled_pos else ([], [])
c2_points2_uncoupled_pos, x2_points_uncoupled_pos = zip(*fixed_points2_uncoupled_pos) if fixed_points2_uncoupled_pos else ([], [])

'''
c1_points1_neg, x1_points_neg = zip(*fixed_points1_neg) if fixed_points1_neg else ([], [])
c2_points2_coupled_neg, x2_points_coupled_neg = zip(*fixed_points2_coupled_neg) if fixed_points2_coupled_neg else ([], [])
c2_points2_uncoupled_neg, x2_points_uncoupled_neg = zip(*fixed_points2_uncoupled_neg) if fixed_points2_uncoupled_neg else ([], [])
'''
# 绘制散点图
plt.figure(figsize=(12, 8))

# 耦合强度为正的结果
plt.scatter(c1_points1_pos, x1_points_pos, label='Fixed points of x1 (d12 = +0.2)', color='blue', s=5)
plt.scatter(c2_points2_coupled_pos, x2_points_coupled_pos, label='Fixed points of x2 with coupling (d12 = +0.2)', color='red', s=5)
plt.scatter(c2_points2_uncoupled_pos, x2_points_uncoupled_pos, label='Fixed points of x2 without coupling (d12 = +0.2)', color='green', s=5)

'''
# 耦合强度为负的结果
plt.scatter(c1_points1_neg, x1_points_neg, label='Fixed points of x1 (d12 = -0.2)', color='blue', s=5, marker='x')
plt.scatter(c2_points2_coupled_neg, x2_points_coupled_neg, label='Fixed points of x2 with coupling (d12 = -0.2)', color='red', s=5, marker='x')
plt.scatter(c2_points2_uncoupled_neg, x2_points_uncoupled_neg, label='Fixed points of x2 without coupling (d12 = -0.2)', color='green', s=5, marker='x')
'''
# 添加系统1的状态线
plt.axhline(y=-1, color='purple', linestyle='--', linewidth=1, label='System 1 State -1')
plt.axhline(y=1, color='orange', linestyle='--', linewidth=1, label='System 1 State 1')

plt.xlabel('c1 and c2')
plt.ylabel('Fixed Points')
plt.title('Fixed Points vs. c1 and c2 for the Coupled and Uncoupled Systems with Different Coupling Strengths')
plt.legend()
plt.grid(True)
plt.show()