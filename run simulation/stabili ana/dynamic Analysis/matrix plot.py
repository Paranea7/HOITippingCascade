#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_36_per_eps.py

从已有的 stabmap .npy 文件中构建并绘制：对于每个 eps（三体项组合）绘制一个包含
6x6=36 个小子图的大图（每个小子图表示对应 d12-d21 组合下的 c1-c2 stab_map 图像）。

脚本假定 .npy 文件命名中含有 d12、d21 和 eps 标识（与 stability_3d_fullgrid.py 保存的格式兼容）。
如果文件名格式不同，请把一个或几个示例文件名发给我以便我调整解析函数。
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from collections import defaultdict, OrderedDict
from math import ceil, sqrt

# ============ 配置 ============
IN_DIR = 'stability_results_3d_fullgrid'   # 你的 .npy 文件目录
OUT_PLOTS_DIR = 'd12d21_panels'
os.makedirs(OUT_PLOTS_DIR, exist_ok=True)

# 你之前用于显示 stab_map 的 colormap/norm 配置（应与生成时一致）
MAX_STABLE_DISPLAY = 4
cmap = plt.get_cmap('viridis', MAX_STABLE_DISPLAY + 1)
bounds = np.arange(-0.5, MAX_STABLE_DISPLAY + 1.5, 1.0)
norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N, clip=True)

# 期望 d12_list 与 d21_list 的长度（根据你脚本中默认设置）
EXPECTED_D12_LEN = 6
EXPECTED_D21_LEN = 6

# 目标：对前 9 个 eps 组合分别生成一张包含 36 子图的大图
N_EPS_TO_PLOT = 9
# 小图网格尺寸（会根据实际 d12,d21 的数量自动调整；默认 6x6）
GRID_NCOL = EXPECTED_D12_LEN
GRID_NROW = EXPECTED_D21_LEN

# ============ 辅助函数（解析文件名） ============
def extract_value_after_key(fn, key):
    """在文件名 fn（小写）中查找 key 后面的浮点数值（如 d12+0.200 或 d12_+0.200 等）。"""
    pattern = re.compile(re.escape(key) + r'[_\-]?([+\-]?\d+\.\d+|[+\-]?\d+)')
    m = pattern.search(fn)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None

def extract_eps_tag(fn):
    """
    从文件名中提取 eps 标识的短标签。
    尝试匹配像 e123_+0.200 或 e(1,2,3)_+0.200 等片段；若存在多个 e.. 段则拼接。
    返回字符串 tag（用于分组）。
    """
    # 找到所有形如 e..._±number 的片段
    matches = re.findall(r'(e[0-9,_\(\)]+[_\-]?[+\-]?\d+\.\d+|e[0-9,_\(\)]+[_\-]?[+\-]?\d+)', fn)
    if matches:
        # 规范化字符，替换 + -> p, - -> m, 逗号和括号去掉
        parts = []
        for s in matches:
            s2 = s.replace('+', 'p').replace('-', 'm')
            s2 = re.sub(r'[,\(\)]', '', s2)
            parts.append(s2)
        return '__'.join(parts)
    # 备用：在 "__" 分割段中查找包含 'e' 的段
    for segment in fn.split('__'):
        if 'e' in segment and any(ch.isdigit() for ch in segment):
            s2 = segment.replace('+', 'p').replace('-', 'm')
            s2 = re.sub(r'[,\(\)]', '', s2)
            return s2
    # 若无法识别，返回整个文件名缩短版（安全保底）
    # 保留 alnum 和 _ 作为标签
    tag = re.sub(r'[^0-9a-zA-Z_]+', '_', fn)[:60]
    return 'eps_' + tag

# ============ 扫描目录并索引文件 ============
files = [f for f in os.listdir(IN_DIR) if f.endswith('.npy')]
if not files:
    raise SystemExit(f'No .npy files in {IN_DIR}')

# data_index: eps_tag -> dict[(d12,d21)] -> filepath
data_index = defaultdict(dict)
d12_set = set()
d21_set = set()
eps_set = set()

for fn in files:
    fn_low = fn.lower()
    d12 = extract_value_after_key(fn_low, 'd12')
    d21 = extract_value_after_key(fn_low, 'd21')
    if d12 is None or d21 is None:
        # 尝试更宽松的解析：在 'd12' 后取接下来的若干字符里寻找浮点数
        try:
            idx = fn_low.index('d12')
            suf = fn_low[idx+3: idx+12]
            m = re.search(r'([+\-]?\d+\.\d+|[+\-]?\d+)', suf)
            if m:
                d12 = float(m.group(1))
        except ValueError:
            pass
        try:
            idx = fn_low.index('d21')
            suf = fn_low[idx+3: idx+12]
            m = re.search(r'([+\-]?\d+\.\d+|[+\-]?\d+)', suf)
            if m:
                d21 = float(m.group(1))
        except ValueError:
            pass

    if d12 is None or d21 is None:
        print(f'Warning: could not parse d12/d21 from filename: {fn} - skipping')
        continue

    tag = extract_eps_tag(fn_low)
    data_index[tag][(d12, d21)] = os.path.join(IN_DIR, fn)
    d12_set.add(d12)
    d21_set.add(d21)
    eps_set.add(tag)

d12_list = sorted(d12_set)
d21_list = sorted(d21_set)
eps_list = sorted(eps_set)

print(f'Found {len(files)} files. Parsed {len(d12_list)} unique d12, {len(d21_list)} unique d21, {len(eps_list)} eps tags.')

# ============ 选择要绘制的 eps 标签（9 个） ============
if len(eps_list) == 0:
    raise SystemExit('No eps tags found in filenames.')

n_eps_to_plot = min(N_EPS_TO_PLOT, len(eps_list))
selected_eps = eps_list[:n_eps_to_plot]
print(f'Selected {n_eps_to_plot} eps tags to plot (first {n_eps_to_plot}):')
for t in selected_eps:
    print('  ', t)

# ============ 主绘图函数 ============
def plot_panel_for_eps(eps_tag, mapping, d12_list, d21_list, outdir, panel_size=(12,10)):
    """
    mapping: dict {(d12,d21): filepath}
    d12_list, d21_list: sorted lists
    将在 panel_size 大小下绘制 len(d21_list) 行 x len(d12_list) 列 的小子图，
    每个子图为对应 stab_map 的 imshow（使用全局 cmap,norm）。
    """
    ncol = len(d12_list)
    nrow = len(d21_list)
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=panel_size, squeeze=False)
    # iterate over d21 (rows) and d12 (cols)
    for i_row, d21 in enumerate(d21_list):
        for j_col, d12 in enumerate(d12_list):
            ax = axes[i_row][j_col]
            fp = mapping.get((d12, d21))
            if fp is None:
                # no file for this combo
                ax.axis('off')
                continue
            try:
                arr = np.load(fp)
            except Exception as e:
                print(f'  Error loading {fp}: {e}')
                ax.axis('off')
                continue
            # imshow: we want origin='lower' to match earlier saving
            ax.imshow(arr, origin='lower', cmap=cmap, norm=norm, aspect='auto',
                      extent=None)  # extent omitted so pixel grid shown
            ax.set_xticks([]); ax.set_yticks([])
            # put small label for d12/d21 values
            ax.text(0.02, 0.95, f'd12={d12:.3f}\n d21={d21:.3f}', color='white',
                    fontsize=6, transform=ax.transAxes, va='top', ha='left',
                    bbox=dict(facecolor='black', alpha=0.3, pad=1, linewidth=0))
    # add a global colorbar to the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, boundaries=bounds, ticks=np.arange(0, MAX_STABLE_DISPLAY+1))
    cbar.set_label('number of stable equilibria')

    fig.suptitle(f'eps: {eps_tag}  (d12 x d21 grid)', fontsize=10)
    plt.tight_layout(rect=[0,0,0.9,0.95])
    outfn = os.path.join(outdir, f'panel_eps__{eps_tag}.png')
    fig.savefig(outfn, dpi=200)
    plt.close(fig)
    print(f'  Saved panel to {outfn}')
    return outfn

# ============ 为每个选定 eps 生成 panel（36 小图） ============
for eps_tag in selected_eps:
    mapping = data_index.get(eps_tag, {})
    # If some (d12,d21) combos are missing, we still create the grid and leave blank cells.
    # If the number of d12/d21 is not 6x6, the grid will adapt to actual sizes.
    # If you want to enforce 6x6 ordering from EXPECTED lists, you can sort d12_list/d21_list accordingly.
    panel_out = plot_panel_for_eps(eps_tag, mapping, d12_list, d21_list, OUT_PLOTS_DIR,
                                   panel_size=(GRID_NCOL*1.2, GRID_NROW*1.0))

print('All done.')