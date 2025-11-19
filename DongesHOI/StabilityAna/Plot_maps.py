#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import BoundaryNorm

d12_vals = [0.0, 0.2, 0.3, 0.5, 0.7, 0.9]
d21_vals = [0.0, -0.2, -0.3, -0.5, -0.7, -0.9]

c1_min, c1_max = -0.8, 0.8
c2_min, c2_max = -0.8, 0.8
MAX_STABLE_DISPLAY = 4

cmap = plt.get_cmap("viridis", MAX_STABLE_DISPLAY + 1)
bounds = np.arange(-0.5, MAX_STABLE_DISPLAY + 1.5, 1.0)
norm = BoundaryNorm(bounds, cmap.N, clip=True)

OUTDIR = "stability_results_3sys_threebody_fixE"

def load_map(d12, d21):
    fn = os.path.join(OUTDIR, f"stabmap_d12_{d12}_d21_{d21}.npy")
    return np.load(fn)

def make_combined_pdf():
    rows = len(d21_vals)
    cols = len(d12_vals)

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(3*cols, 3.0*rows),   # 稍微增加整体尺寸
        sharex=True,
        sharey=True
    )
    axes = np.atleast_2d(axes)

    xticks = np.arange(c1_min, c1_max + 0.01, 0.2)
    yticks = np.arange(c2_min, c2_max + 0.01, 0.2)

    for i, d21 in enumerate(d21_vals):
        for j, d12 in enumerate(d12_vals):
            ax = axes[i, j]
            sm = load_map(d12, d21)

            ax.imshow(
                sm, extent=(c1_min, c1_max, c2_min, c2_max),
                origin="lower", cmap=cmap, norm=norm
            )

            # 所有子图都显示刻度数字
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.set_xticklabels([f"{x:.1f}" for x in xticks])
            ax.set_yticklabels([f"{y:.1f}" for y in yticks])

            # 小号刻度
            ax.tick_params(labelsize=7)

    # 子图间距调大
    fig.subplots_adjust(
        left=0.12, bottom=0.12, right=0.90, top=0.95,
        wspace=0.15,   # 原为 0.05
        hspace=0.15    # 原为 0.05
    )

    # 外圈标 d21（左侧）
    for i, d21 in enumerate(d21_vals):
        ax = axes[i, 0]
        pos = ax.get_position()
        fig.text(
            pos.x0 - 0.02,                 # 稍微更靠近
            pos.y0 + pos.height/2,
            f"d21={d21}",
            ha="right", va="center", fontsize=12
        )

    # 外圈标 d12（底部）
    for j, d12 in enumerate(d12_vals):
        ax = axes[-1, j]
        pos = ax.get_position()
        fig.text(
            pos.x0 + pos.width/2,
            pos.y0 - 0.03,                 # 稍微更靠近
            f"d12={d12}",
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

    out_pdf = os.path.join(OUTDIR, "combined_stabmap_grid.pdf")
    plt.savefig(out_pdf)
    plt.close(fig)
    print("Saved:", out_pdf)


if __name__ == "__main__":
    make_combined_pdf()