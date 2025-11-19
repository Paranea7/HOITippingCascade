#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import BoundaryNorm

# 三体耦合扫描列表（必须与计算程序一致）
e123_vals = [0.0, 0.2, 0.3, 0.5, 0.7, 0.9]
e231_vals = [0.0, -0.2, -0.3, -0.5, -0.7, -0.9]

# c1, c2 扫描范围
c1_min, c1_max = -0.8, 0.8
c2_min, c2_max = -0.8, 0.8

# 显示最多 4 个稳态
MAX_STABLE_DISPLAY = 4

cmap = plt.get_cmap("viridis", MAX_STABLE_DISPLAY + 1)
bounds = np.arange(-0.5, MAX_STABLE_DISPLAY + 1.5, 1.0)
norm = BoundaryNorm(bounds, cmap.N, clip=True)

# 输出目录
OUTDIR = "stability_results_3sys_threebody_fixE"

def load_map(e123, e231):
    fn = os.path.join(OUTDIR, f"stabmap_e123_{e123}_e231_{e231}.npy")
    return np.load(fn)

def make_combined_pdf():
    rows = len(e231_vals)
    cols = len(e123_vals)

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(3*cols, 3.0*rows),
        sharex=True,
        sharey=True
    )
    axes = np.atleast_2d(axes)

    xticks = np.arange(c1_min, c1_max + 0.01, 0.2)
    yticks = np.arange(c2_min, c2_max + 0.01, 0.2)

    for i, e231 in enumerate(e231_vals):
        for j, e123 in enumerate(e123_vals):
            ax = axes[i, j]
            sm = load_map(e123, e231)

            ax.imshow(
                sm, extent=(c1_min, c1_max, c2_min, c2_max),
                origin="lower", cmap=cmap, norm=norm
            )

            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.set_xticklabels([f"{x:.1f}" for x in xticks])
            ax.set_yticklabels([f"{y:.1f}" for y in yticks])
            ax.tick_params(labelsize=7)

    fig.subplots_adjust(
        left=0.12, bottom=0.12, right=0.90, top=0.95,
        wspace=0.15, hspace=0.15
    )

    # 左侧标注 e231
    for i, e231 in enumerate(e231_vals):
        ax = axes[i, 0]
        pos = ax.get_position()
        fig.text(
            pos.x0 - 0.02,
            pos.y0 + pos.height/2,
            f"e231={e231}",
            ha="right", va="center", fontsize=12
        )

    # 底部标注 e123
    for j, e123 in enumerate(e123_vals):
        ax = axes[-1, j]
        pos = ax.get_position()
        fig.text(
            pos.x0 + pos.width/2,
            pos.y0 - 0.03,
            f"e123={e123}",
            ha="center", va="top", fontsize=12
        )

    # 右侧 colorbar
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        boundaries=bounds,
        ticks=np.arange(0, MAX_STABLE_DISPLAY + 1)
    )

    out_pdf = os.path.join(OUTDIR, "combined_stabmap_threebody.pdf")
    plt.savefig(out_pdf)
    plt.close(fig)
    print("Saved:", out_pdf)


if __name__ == "__main__":
    make_combined_pdf()