#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
miuphase.py

从 CSV 文件读取 mu_d x mu_e 网格的生存率数据并绘制热图，
同时在图上绘制辅助线 mu_e = mu_d, mu_e = 2*mu_d, mu_e = 0.5*mu_d。

增强：如果没有通过命令行提供 --csv，会尝试在当前目录查找最新的 .csv 文件并使用它。
"""

from __future__ import annotations
import os
import sys
import glob
import csv
import argparse
from typing import Tuple, Optional, List
import numpy as np
import matplotlib.pyplot as plt

def find_latest_csv_in_dir(directory: str = ".") -> Optional[str]:
    """
    在指定目录中查找最新修改的 .csv 文件，返回其路径；若不存在则返回 None。
    """
    pattern = os.path.join(directory, "*.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]

def read_grid_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    读取由 save_grid_csv 输出的 CSV 文件，返回 mu_d_vals, mu_e_vals, grid
    grid 的形状为 (len(mu_e_vals), len(mu_d_vals))

    期望 CSV 格式：
    第一行 header: ["mu_e\\mu_d", mu_d1, mu_d2, ...]
    后续每行: [mu_e, val(mu_d1), val(mu_d2), ...]
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        # 跳过完全空行
        rows = [r for r in reader if any(cell.strip() for cell in r)]

    if len(rows) < 2:
        raise ValueError("CSV 文件行数不足或内容为空，无法解析网格数据。")

    header = rows[0]
    if len(header) < 2:
        raise ValueError("CSV header 不包含 mu_d 值，文件格式不符合预期。")

    # 解析 mu_d 值（从 header 的第 2 列开始）
    try:
        mu_d_vals = np.array([float(x) for x in header[1:]], dtype=float)
    except ValueError as e:
        raise ValueError(f"无法解析 header 中的 mu_d 值: {e}")

    mu_e_list: List[float] = []
    grid_rows: List[List[float]] = []
    for r in rows[1:]:
        # 忽略空行或仅包含空白的行
        if not any(cell.strip() for cell in r):
            continue
        if len(r) < 2:
            raise ValueError(f"CSV 中某行列数太少，无法解析：{r}")
        try:
            mu_e = float(r[0])
        except ValueError:
            raise ValueError(f"无法解析 mu_e 值: '{r[0]}'")
        mu_e_list.append(mu_e)

        # 取与 mu_d_vals 长度一致的列（若行更长则截断，若更短则报错）
        row_vals = r[1:1 + len(mu_d_vals)]
        if len(row_vals) < len(mu_d_vals):
            raise ValueError(f"行数据长度与 header 中 mu_d 数量不匹配 (期望 {len(mu_d_vals)} 个值)，行: {r}")
        try:
            row_f = [float(x) for x in row_vals]
        except ValueError as e:
            raise ValueError(f"无法解析生存率数值: {e}")
        grid_rows.append(row_f)

    mu_e_vals = np.array(mu_e_list, dtype=float)
    grid = np.array(grid_rows, dtype=float)

    if grid.shape != (len(mu_e_vals), len(mu_d_vals)):
        raise ValueError(f"读取后的网格形状不匹配: {grid.shape} vs ({len(mu_e_vals)},{len(mu_d_vals)})")

    return mu_d_vals, mu_e_vals, grid

def plot_heatmap_with_guides(mu_d_vals: np.ndarray,
                             mu_e_vals: np.ndarray,
                             grid: np.ndarray,
                             title: str = "Survival rate",
                             cmap: str = 'viridis',
                             vmin: Optional[float] = None,
                             vmax: Optional[float] = None,
                             out_png: Optional[str] = None,
                             show: bool = True,
                             figsize: Tuple[int, int] = (8, 6)):
    """
    使用 imshow 绘制 heatmap，并在上面绘制辅助线 mu_e = slope * mu_d (slope 为 1, 2, 0.5)
    extent 使用 mu_d 和 mu_e 的最小/最大值，origin='lower' 使得 mu_e 从小到大向上。
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 如果 mu_d_vals 或 mu_e_vals 不是单调的，取最小最大作为坐标边界
    xmin, xmax = float(np.min(mu_d_vals)), float(np.max(mu_d_vals))
    ymin, ymax = float(np.min(mu_e_vals)), float(np.max(mu_e_vals))

    im = ax.imshow(grid, origin='lower', aspect='auto',
                   extent=[xmin, xmax, ymin, ymax],
                   cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xlabel("mu_d")
    ax.set_ylabel("mu_e")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Survival rate")

    # 绘制辅助线：mu_e = slope * mu_d
    slopes = [
        (1.0, 'mu_e = mu_d', 'white', 1.8),
        (2.0, 'mu_e = 2 mu_d', 'yellow', 1.2),
        (0.5, 'mu_e = 0.5 mu_d', 'cyan', 1.2),
    ]
    # 使用足够细的采样以保证线在图中连续
    x_vals = np.linspace(xmin, xmax, 2000)
    for slope, label, color, lw in slopes:
        y_vals = slope * x_vals
        mask = (y_vals >= ymin) & (y_vals <= ymax)
        if np.any(mask):
            ax.plot(x_vals[mask], y_vals[mask], color=color, linewidth=lw, label=label)

    # 图例放在合适位置
    ax.legend(loc='upper right', fontsize='small', framealpha=0.7)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    fig.tight_layout()

    if out_png:
        out_dir = os.path.dirname(out_png)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(out_png, dpi=150)
        print(f"Saved figure to: {out_png}")

    if show:
        plt.show()
    else:
        plt.close(fig)

def parse_args_with_default() -> argparse.Namespace:
    """
    解析命令行参数。如果 --csv 未提供，尝试使用当前目录下最新的 CSV 文件（如果存在）。
    若找不到任何 CSV，则打印用法并退出（非零退出码）。
    """
    parser = argparse.ArgumentParser(description="Plot heatmap from CSV and draw guides mu_e = mu_d, mu_e = 2*mu_d, mu_e = 0.5*mu_d.")
    parser.add_argument("--csv", "-c", required=False, help="Input CSV file (output of save_grid_csv). If omitted, script will try to find latest CSV in current directory.")
    parser.add_argument("--out", "-o", default=None, help="Output PNG path (optional). If not provided, image will still be shown.")
    parser.add_argument("--no-show", action="store_true", help="Do not display the figure interactively (only save).")
    parser.add_argument("--cmap", default='viridis', help="Colormap for heatmap (default 'viridis').")
    parser.add_argument("--vmin", type=float, default=None, help="Colorbar vmin (default: automatic).")
    parser.add_argument("--vmax", type=float, default=None, help="Colorbar vmax (default: automatic).")
    args = parser.parse_args()

    if not args.csv:
        candidate = find_latest_csv_in_dir(".")
        if candidate:
            print(f"No --csv provided. Using latest CSV in current directory: {candidate}")
            args.csv = candidate
        else:
            parser.print_usage()
            print("\nError: --csv is required and no CSV found in current directory.")
            sys.exit(2)
    return args

def main():
    args = parse_args_with_default()
    csv_path: str = args.csv
    out_png: Optional[str] = args.out
    show: bool = not args.no_show

    try:
        mu_d_vals, mu_e_vals, grid = read_grid_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV '{csv_path}': {e}", file=sys.stderr)
        sys.exit(3)

    title = f"Survival rate (from {os.path.basename(csv_path)})"
    try:
        plot_heatmap_with_guides(mu_d_vals, mu_e_vals, grid, title=title,
                                 cmap=args.cmap, vmin=args.vmin, vmax=args.vmax,
                                 out_png=out_png, show=show)
    except Exception as e:
        print(f"Error plotting figure: {e}", file=sys.stderr)
        sys.exit(4)

if __name__ == "__main__":
    main()