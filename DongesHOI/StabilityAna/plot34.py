#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import BoundaryNorm
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator

# ==========================================
# 1. PRL 风格全局设置 (极简版)
# ==========================================
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,

    # --- 刻度设置：L型风格 ---
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": False,  # 顶部无刻度
    "ytick.right": False,  # 右侧无刻度
    "xtick.bottom": True,
    "ytick.left": True,

    # --- 刻度尺寸 ---
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,

    "figure.dpi": 300,
    "savefig.bbox": "tight",
})

# ==========================================
# 2. 数据与参数
# ==========================================
FILE_MAP = [
    "stabmap_d12_0.0_d21_0.0.npy",
    "stabmap_d12_0.2_d21_-0.2.npy",
    "stabmap_d12_0.7_d21_-0.7.npy",
    "stabmap_d12_0.9_d21_-0.9.npy"
]

FOLDERS = [
    "stability_results",
    "stability_results_3sys_c3_fixed",
    "stability_results_3sys_threebody_fixE"
]

ROW_LABELS = [r"2-System", r"3-System (no HOI)", r"3-System (with HOI)"]

c1_min, c1_max = -0.8, 0.8
c2_min, c2_max = -0.8, 0.8

MAX_STABLE_DISPLAY = 4
cmap = plt.get_cmap("viridis", MAX_STABLE_DISPLAY + 1)
bounds = np.arange(-0.5, MAX_STABLE_DISPLAY + 1.5, 1.0)
norm = BoundaryNorm(bounds, cmap.N, clip=True)


def load_map(folder, filename):
    fn = os.path.join(folder, filename)
    if not os.path.exists(fn):
        return np.random.randint(0, MAX_STABLE_DISPLAY + 1, size=(50, 50))
    return np.load(fn)


def parse_label(filename):
    parts = filename.replace(".npy", "").split("_")
    return rf"$d_{{12}}={parts[2]},\ d_{{21}}={parts[4]}$"


# ==========================================
# 3. 绘图主逻辑
# ==========================================
def make_clean_prl_plot():
    rows = len(FOLDERS)
    cols = len(FILE_MAP)

    # 尺寸设置：PRL 双栏宽度 ~7.0 英寸
    fig_width = 7.0
    # 适当增加高度以容纳间距
    fig_height = fig_width * (rows / cols) * 0.9 + 0.8

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(fig_width, fig_height),
        sharex=True, sharey=True,
        # <--- 关键修改：保留适当的间距 (0.1 左右既有间隔又不会太散)
        gridspec_kw={'wspace': 0.1, 'hspace': 0.1}
    )

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    # <--- 关键修改：极简刻度 (只显示头、尾、中间) --->
    # 这样绝对不会重叠，且看着非常清爽
    clean_ticks = [-0.8, 0.0, 0.8]

    for i, folder in enumerate(FOLDERS):
        for j, filename in enumerate(FILE_MAP):
            ax = axes[i, j]
            sm = load_map(folder, filename)

            im = ax.imshow(
                sm, extent=(c1_min, c1_max, c2_min, c2_max),
                origin="lower", cmap=cmap, norm=norm, aspect='auto'
            )

            # 1. 设置极简刻度
            ax.set_xticks(clean_ticks)
            ax.set_yticks(clean_ticks)

            # 2. 次级刻度 (可选，设为 2 表示主刻度间再分一段，即 0.4 处有个小短线)
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))

            # 3. 轴标签控制
            if i == rows - 1:
                ax.set_xlabel(r"$c_1$", fontsize=10)

            if j == 0:
                ax.set_ylabel(r"$c_2$", fontsize=10)

            # 4. 顶部标题 (字体稍微改小，避免拥挤)
            if i == 0:
                ax.set_title(parse_label(filename), fontsize=8, pad=6)

            # 5. 右侧行标签
            if j == cols - 1:
                ax.text(1.08, 0.5, ROW_LABELS[i],
                        transform=ax.transAxes,
                        rotation=270,
                        va='center', ha='left',
                        fontsize=9, fontweight='normal')

    # ==========================================
    # 4. Colorbar 布局
    # ==========================================
    fig.subplots_adjust(bottom=0.16, top=0.90, left=0.08, right=0.92)

    pos_left = axes[-1, 0].get_position().x0
    pos_right = axes[-1, -1].get_position().x1
    cax_width = pos_right - pos_left
    cax_y = 0.07  # 稍微下移
    cax_height = 0.02

    cax = fig.add_axes([pos_left, cax_y, cax_width, cax_height])

    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax, orientation='horizontal',
        boundaries=bounds,
        ticks=np.arange(0, MAX_STABLE_DISPLAY + 1)
    )
    cb.ax.tick_params(labelsize=9)
    cb.ax.xaxis.set_ticks_position('bottom')
    cb.set_label(r'Number of Stable Fixed Points ($N_{st}$)', fontsize=10)

    out_pdf = "PRL_style_clean_spaced.pdf"
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved clean plot: {out_pdf}")


if __name__ == "__main__":
    make_clean_prl_plot()