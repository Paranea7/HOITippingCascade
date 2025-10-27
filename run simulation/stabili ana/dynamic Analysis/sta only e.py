#!/usr/bin/env python3
"""
scan_cubic_model_multi_c12.py

改动说明（主要）：
- 预生成一组随机起点（random_initial_points），对每个网格点使用相同的这组随机起点，
  避免每个网格点使用不同随机起点导致相图出现随机花纹。
- 保留确定性起点与额外角点。
- 并行安全：将 pregen 随机点数组作为任务参数传入子进程。
- 若想关闭随机起点，将 RANDOM_INITIAL_COUNT 设为 0。

用途：
  模型（S=3）:
    dx_i/dt = -x_i^3 + x_i + c_i + sum_{j,k=1..3} e_{i j k} x_j x_k

  在 eps123 - eps231 平面上扫描，对多组 (c1,c2) 固定值（且 c3 固定）分别统计稳定平衡点数。

输出:
  OUT_DIR/eps123_eps231_c1_{c1str}_c2_{c2str}.npz  （包含 X, Y, result 等）
  可选热图 PNG（如果 matplotlib 可用）

依赖: numpy, scipy, tqdm; matplotlib 可选
"""
import os
from copy import deepcopy
import numpy as np
from functools import partial
from tqdm import tqdm
from scipy.optimize import root

# 可选绘图
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# 并行设置
USE_MULTIPROCESS = True
NUM_WORKERS = None  # None -> cpu_count()

if USE_MULTIPROCESS:
    from multiprocessing import Pool, cpu_count

# -----------------------
# 配置区（可修改）
# -----------------------

# 默认 eps 字典（所有 e_{ijk} 初始为 0）
DEFAULT_EPS = {
    'eps123': 0.0,
    'eps231': 0.0,
    'eps312': 0.0,
    'eps321': 0.0,
    'eps213': 0.0,
    'eps132': 0.0
}

SCAN_PARAM_X = 'eps123'
SCAN_PARAM_Y = 'eps231'

# 多组 (c1,c2)。固定 c3
C12_GROUPS = [
    (0.0, 0.0),
    (0.2, 0.0),
    (0.0, 0.2),
    (0.2, -0.2),
    (-0.5, 0.3),
    (0.3, 0.5),
    (0.2, 0.5),
    (-0.2, -0.3),
    (-0.3, -0.5),
]

C3_FIXED = 0.0

# 网格设置（可调）
X_vals = np.linspace(-0.5, 0.5, 101)    # eps123 values
Y_vals = np.linspace(-0.5, 0.5, 101)    # eps231 values

# 基础确定性起点（保留原有的）
DETERMINISTIC_INITIAL_XS = [
    np.array([0.0, 0.0, 0.0]),
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
    np.array([-1.0, 0.0, 0.0]),
    np.array([0.0, -1.0, 0.0]),
    np.array([0.0, 0.0, -1.0]),
    np.array([0.5, 0.5, 0.5]),
    np.array([-0.5, 0.5, -0.5]),
    np.array([0.2, -0.2, 0.1])
]

# 额外增加的一组角点与中点，有助于覆盖更多 basin
EXTRA_INITIAL_XS = [
    np.array([2.0, 0.0, 0.0]),
    np.array([0.0, 2.0, 0.0]),
    np.array([0.0, 0.0, 2.0]),
    np.array([-2.0, 0.0, 0.0]),
    np.array([0.0, -2.0, 0.0]),
    np.array([0.0, 0.0, -2.0]),
    np.array([1.0, 1.0, 0.0]),
    np.array([1.0, 0.0, 1.0]),
    np.array([0.0, 1.0, 1.0]),
    np.array([-1.0, -1.0, -1.0])
]

# 随机起点设置（预生成一组并在所有网格点复用）
RANDOM_INITIAL_COUNT = 101   # 每个网格点将尝试这组随机起点（若设为0则关闭随机起点）
RANDOM_INITIAL_SCALE = 1.5  # 随机起点范围为 Uniform(-scale, scale) 每分量
RANDOM_SEED = 12345         # 固定种子以便复现；设为 None 则每次不同

# 容忍与判定阈值（可调）
F_TOL = 1e-8         # 求解残差容忍
ROOT_TOL = 1e-6      # 去重两个根的距离阈值
STABILITY_TOL = 1e-9 # 判定特征值实部应 < -STABILITY_TOL 才视为稳定
MAX_STABLE_DISPLAY = 1000

# 求根器设置（可调）
ROOT_METHOD = 'hybr'   # scipy.optimize.root 可选方法：hybr, lm, broyden1 ...
ROOT_TOL_SOLVER = 1e-10  # 传给 root 的 tol

# 输出
OUT_DIR = "scan_results_multi_c12_cubic"
PLOT_TEMPLATE = "eps123_eps231_c1_{c1s}_c2_{c2s}.png"
NPZ_TEMPLATE = "eps123_eps231_c1_{c1s}_c2_{c2s}.npz"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# 模型 f 和 雅可比 J 的定义（基于给定公式）
# -----------------------
def f_and_J(x, c, eps):
    x1, x2, x3 = float(x[0]), float(x[1]), float(x[2])
    c1, c2, c3 = float(c[0]), float(c[1]), float(c[2])

    e123 = float(eps.get('eps123', 0.0))
    e132 = float(eps.get('eps132', 0.0))
    e213 = float(eps.get('eps213', 0.0))
    e231 = float(eps.get('eps231', 0.0))
    e312 = float(eps.get('eps312', 0.0))
    e321 = float(eps.get('eps321', 0.0))

    f1 = -x1**3 + x1 + c1 + e123 * x2 * x3 + e132 * x3 * x2
    f2 = -x2**3 + x2 + c2 + e231 * x3 * x1 + e213 * x1 * x3
    f3 = -x3**3 + x3 + c3 + e312 * x1 * x2 + e321 * x2 * x1

    f = np.array([f1, f2, f3], dtype=float)

    J = np.zeros((3, 3), dtype=float)
    J[0, 0] = -3.0 * x1**2 + 1.0
    J[0, 1] = e123 * x3 + e132 * x3
    J[0, 2] = e123 * x2 + e132 * x2

    J[1, 0] = e231 * x3 + e213 * x3
    J[1, 1] = -3.0 * x2**2 + 1.0
    J[1, 2] = e231 * x1 + e213 * x1

    J[2, 0] = e312 * x2 + e321 * x2
    J[2, 1] = e312 * x1 + e321 * x1
    J[2, 2] = -3.0 * x3**2 + 1.0

    return f, J

# -----------------------
# 构造 eps 字典（用于每个网格点）
# -----------------------
def make_eps_dict(var_x_val, var_y_val):
    eps = deepcopy(DEFAULT_EPS)
    eps[SCAN_PARAM_X] = float(var_x_val)
    eps[SCAN_PARAM_Y] = float(var_y_val)
    return eps

# -----------------------
# 寻根、去重、稳定判定（接收预生成随机起点）
# -----------------------
def find_roots_and_count_stable(eps, c_vec, random_initial_points=None):
    """
    对给定 eps、c_vec 使用多起点（确定性 + 预生成随机起点）求根、去重并统计稳定解数量。
    random_initial_points: None 或 ndarray(shape=(M,3))
      如果提供，则直接将这些点加入每个网格点的初始点集合（所有网格点使用相同的这组点）。
    返回: (n_stable, roots_list, stable_roots_list)
    """
    roots = []

    # 合并确定性起点集合
    initial_points = list(DETERMINISTIC_INITIAL_XS) + list(EXTRA_INITIAL_XS)

    # 使用预生成的随机起点（若提供）
    if random_initial_points is not None and len(random_initial_points) > 0:
        for rp in random_initial_points:
            initial_points.append(np.asarray(rp, dtype=float))
    else:
        # 向后兼容：若没有提供预生成随机点，但全局配置要求随机点，则基于 RANDOM_SEED 生成
        if RANDOM_INITIAL_COUNT > 0:
            rng = np.random.RandomState(RANDOM_SEED)
            rand_pts = rng.uniform(-RANDOM_INITIAL_SCALE, RANDOM_INITIAL_SCALE,
                                   size=(RANDOM_INITIAL_COUNT, 3))
            for rp in rand_pts:
                initial_points.append(rp.astype(float))

    # 尝试每个起点求解
    for x0 in initial_points:
        try:
            sol = root(lambda xx: f_and_J(xx, c_vec, eps)[0], x0, method=ROOT_METHOD, tol=ROOT_TOL_SOLVER)
        except Exception:
            # 若方法失败（例如数值问题），跳过该起点
            continue
        if not sol.success:
            continue
        x_sol = sol.x.astype(float)
        fval = f_and_J(x_sol, c_vec, eps)[0]
        if np.linalg.norm(fval) > F_TOL:
            # 残差过大，认为未收敛到解
            continue
        # 去重：若与已有 root 距离小于 ROOT_TOL 则认为重复
        duplicated = False
        for r in roots:
            if np.linalg.norm(r - x_sol) < ROOT_TOL:
                duplicated = True
                break
        if not duplicated:
            roots.append(x_sol)

    # 判定稳定性
    stable_roots = []
    for r in roots:
        _, J = f_and_J(r, c_vec, eps)
        eigs = np.linalg.eigvals(J)
        if np.all(np.real(eigs) < -STABILITY_TOL):
            stable_roots.append(r)

    n_stable = int(min(len(stable_roots), MAX_STABLE_DISPLAY))
    return n_stable, roots, stable_roots

# -----------------------
# 单点计算（用于并行/串行）—— 接收预生成随机起点
# -----------------------
def compute_point(args):
    """
    args = (i, j, xv, yv, c_vec, pregen_random_initials)
    返回 (i, j, n_stable)
    """
    i, j, xv, yv, c_vec, pregen_random_initials = args
    eps = make_eps_dict(xv, yv)
    n_stable, roots, stable_roots = find_roots_and_count_stable(eps, c_vec, random_initial_points=pregen_random_initials)
    return (i, j, n_stable)

# -----------------------
# 扫描函数（针对某组 c1,c2）
# -----------------------
def run_scan_for_c12(c1, c2, use_multiprocess=USE_MULTIPROCESS, num_workers=NUM_WORKERS):
    c_vec = (float(c1), float(c2), float(C3_FIXED))
    X = np.array(X_vals, dtype=float)
    Y = np.array(Y_vals, dtype=float)
    nx = X.size
    ny = Y.size
    result = np.zeros((nx, ny), dtype=int)

    # 预生成一组随机起点（对所有网格点一致），以避免网格点间随机差异
    if RANDOM_INITIAL_COUNT > 0:
        rng = np.random.RandomState(RANDOM_SEED)
        pregen_random_initials = rng.uniform(-RANDOM_INITIAL_SCALE, RANDOM_INITIAL_SCALE,
                                             size=(RANDOM_INITIAL_COUNT, 3))
    else:
        pregen_random_initials = None

    # 构建任务列表：把同一组 pregen_random_initials 传给每个任务
    tasks = []
    for i, xv in enumerate(X):
        for j, yv in enumerate(Y):
            tasks.append((i, j, xv, yv, c_vec, pregen_random_initials))

    if use_multiprocess:
        workers = num_workers or cpu_count()
        with Pool(processes=workers) as pool:
            for (i, j, n_stable) in tqdm(pool.imap_unordered(compute_point, tasks),
                                         total=len(tasks), desc=f"Scanning c1={c1},c2={c2}"):
                result[i, j] = n_stable
    else:
        for args in tqdm(tasks, desc=f"Scanning c1={c1},c2={c2}"):
            i, j, n_stable = compute_point(args)
            result[i, j] = n_stable

    # 保存结果
    c1s = format_c_for_filename(c1)
    c2s = format_c_for_filename(c2)
    npz_path = os.path.join(OUT_DIR, NPZ_TEMPLATE.format(c1s=c1s, c2s=c2s))
    np.savez_compressed(npz_path, X=X, Y=Y, result=result, c_vec=c_vec)
    print(f"Saved: {npz_path}")

    # 绘图（可选） — 离散色阶 0..num_levels-1
    if HAS_MPL:
        import matplotlib.colors as mcolors
        # 动态设定 num_levels 至少覆盖实际结果的范围
        maxval = int(result.max())
        num_levels = max(8, maxval + 1)  # 最少 8 个级别（0..7），否则扩展到实际最大值
        cmap = plt.get_cmap('viridis', num_levels)
        boundaries = np.arange(0, num_levels + 1, 1)
        # 不要在 imshow 中同时传 norm 与 vmin/vmax；这里只传 norm
        norm = mcolors.BoundaryNorm(boundaries, ncolors=num_levels)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(result.T, origin='lower',
                       extent=(X.min(), X.max(), Y.min(), Y.max()),
                       aspect='auto', cmap=cmap, norm=norm)
        ax.set_xlabel(SCAN_PARAM_X)
        ax.set_ylabel(SCAN_PARAM_Y)
        ax.set_title(f"n_stable (c1={c1}, c2={c2}, c3={C3_FIXED})")

        # colorbar ticks 置于每个颜色块中央：使用 0..num_levels-1
        ticks = np.arange(0, num_levels)
        cbar = fig.colorbar(im, ax=ax, boundaries=boundaries, ticks=ticks)
        cbar.set_ticklabels([str(i) for i in ticks])
        cbar.set_label('n_stable')

        plt.tight_layout()
        png_path = os.path.join(OUT_DIR, PLOT_TEMPLATE.format(c1s=c1s, c2s=c2s))
        plt.savefig(png_path, dpi=200)
        plt.close(fig)
        print(f"Saved plot: {png_path}")
    else:
        print("matplotlib not available; skipping plot.")

    return result

def format_c_for_filename(c):
    if float(c) == 0.0:
        return "0"
    s = f"{float(c):.6g}"
    s = s.replace('-', 'm').replace('.', 'p')
    return s

# -----------------------
# 主入口
# -----------------------
def main():
    all_results = {}
    for (c1, c2) in C12_GROUPS:
        print(f"Starting scan for c1={c1}, c2={c2}, c3={C3_FIXED}")
        res = run_scan_for_c12(c1, c2, use_multiprocess=USE_MULTIPROCESS)
        all_results[(c1, c2)] = res
        unique, counts = np.unique(res, return_counts=True)
        print(f"Summary for c1={c1}, c2={c2}:")
        for u, cnt in zip(unique, counts):
            print(f"  {u}: {cnt} grid points")
    print("All done. Results saved in", OUT_DIR)

if __name__ == "__main__":
    main()