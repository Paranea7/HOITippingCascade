#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import BoundaryNorm

# --- 参数：d12 升序；d21 负号保留 ---
d12_vals = [0.0, 0.2, 0.3, 0.5, 0.7, 0.9]
d21_vals = [0.0, -0.2, -0.3, -0.5, -0.7, -0.9]

c1_min, c1_max = 0.0, 0.8
c2_min, c2_max = 0.0, 0.8
MAX_STABLE_DISPLAY = 4

cmap = plt.get_cmap("viridis", MAX_STABLE_DISPLAY + 1)
bounds = np.arange(-0.5, MAX_STABLE_DISPLAY + 1.5, 1.0)
norm = BoundaryNorm(bounds, cmap.N, clip=True)

OUTDIR = "stability_results"


def load_map(d12, d21):
    fn = os.path.join(OUTDIR, f"stabmap_d12_{d12}_d21_{d21}.npy")
    if os.path.exists(fn):
        return np.load(fn)
    else:
        print(f"Warning: {fn} not found, using random data for demo.")
        return np.random.randint(0, 5, (200, 200))


def make_combined_pdf():
    rows = len(d12_vals)
    cols = len(d21_vals)

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(5.0 * cols, 5.0 * rows),
        sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes)

    xticks = np.arange(c1_min, c1_max + 0.01, 0.4)
    yticks = np.arange(c2_min, c2_max + 0.01, 0.4)

    # ========== 绘制所有子图 ==========
    for i, d12 in enumerate(d12_vals):
        for j, d21 in enumerate(d21_vals):
            ax = axes[i, j]
            sm = load_map(d12, d21)

            ax.imshow(
                sm, extent=(c1_min, c1_max, c2_min, c2_max),
                origin="lower", cmap=cmap, norm=norm, aspect='auto'
            )

            ax.set_xticks(xticks)
            ax.set_yticks(yticks)

            # --- 外圈标签：底排标 c1，左列标 c2 ---
            if i == rows - 1:
                ax.set_xlabel(r"$c_1$", fontsize=20)
                ax.set_xticklabels([f"{x:.1f}" for x in xticks], fontsize=16)
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

            if j == 0:
                ax.set_ylabel(r"$c_2$", fontsize=20)
                ax.set_yticklabels([f"{y:.1f}" for y in yticks], fontsize=16)
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)

            ax.tick_params(axis='both', which='major', length=6)

    # ========== 布局 ==========
    fig.subplots_adjust(
        left=0.12, bottom=0.15, right=0.95, top=0.95,
        wspace=0.08, hspace=0.08
    )

    # ========== 左侧标 d12 ==========
    for i, d12 in enumerate(d12_vals):
        ax = axes[i, 0]
        pos = ax.get_position()
        fig.text(
            pos.x0 - 0.05,
            pos.y0 + pos.height / 2,
            f"$d_{{12}}={d12}$",
            ha="right", va="center",
            rotation="vertical",
            fontsize=24
        )

    # ========== 底部标 d21 ==========
    for j, d21 in enumerate(d21_vals):
        ax = axes[-1, j]
        pos = ax.get_position()
        fig.text(
            pos.x0 + pos.width / 2,
            pos.y0 - 0.03,
            f"$d_{{21}}={d21}$",
            ha="center", va="top",
            fontsize=24
        )

    # ========== Colorbar ==========
    cax = fig.add_axes([0.2, 0.05, 0.6, 0.02])
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax, orientation='horizontal',
        boundaries=bounds,
        ticks=np.arange(0, MAX_STABLE_DISPLAY + 1)
    )
    cb.ax.tick_params(labelsize=15)
    cb.set_label('Number of Stable Fixed Points', fontsize=24)

    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)

    out_pdf = os.path.join(OUTDIR, "combined_stabmap_grid_final.pdf")
    plt.savefig(out_pdf)
    plt.close(fig)
    print("Saved:", out_pdf)


if __name__ == "__main__":
    make_combined_pdf()