#!/usr/bin/env python3
"""
plot_all_csvs.py

在脚本顶部设置 csv_path（文件或目录），运行后遍历目录下所有 .csv（可递归）
并为每个 CSV 生成 PNG，保存到 CSV 所在目录的子目录 out_subdir。

修改项（脚本顶部）：
- csv_path: 要处理的目录或单个 CSV 文件
- recursive: 是否递归遍历子目录
- out_subdir: 每个 CSV 所在目录下保存 PNG 的子目录名
- fig_width, fig_height, dpi: 图像尺寸和分辨率
"""

import os
import csv
import sys
import fnmatch
import numpy as np
import matplotlib.pyplot as plt

# ====== 在这里设置输入路径和选项 ======
# 可以是单个 CSV 文件（例如 "/full/path/to/file.csv"）或目录（例如 "outputcsvdrandom1"）
csv_path = "outputcsvdrandom0"

# 是否递归查找子目录中的 CSV 文件
recursive = True

# 每个 CSV 的输出子目录（相对于 CSV 所在目录）
out_subdir = "output_plots"

# 可选设置：图像尺寸（英寸）和分辨率
fig_width = 10.0
fig_height = 6.0
dpi = 150
# =============================================================

def find_csv_files(path, recursive=True, pattern="*.csv"):
    """
    返回匹配的 CSV 文件绝对路径列表。
    如果 path 是文件且匹配 pattern，则返回 [path]。
    如果 path 是目录，则查找目录下匹配的文件（递归或非递归）。
    """
    path = os.path.abspath(path)
    if os.path.isfile(path):
        if fnmatch.fnmatch(os.path.basename(path), pattern):
            return [path]
        else:
            return []
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Path not found: {path}")

    matches = []
    if recursive:
        for root, dirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    matches.append(os.path.join(root, name))
    else:
        for name in os.listdir(path):
            full = os.path.join(path, name)
            if os.path.isfile(full) and fnmatch.fnmatch(name, pattern):
                matches.append(full)
    return matches

def read_csv_file(path):
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError("Empty CSV or missing header")
        data = [row for row in reader if row and not all(cell.strip()=='' for cell in row)]
    return header, data

def parse_csv(header, data):
    """
    Parse CSV produced by simd_no_plot.save_results_csv:
    header: ['sigma_d', 'mean_mu_d_<mu1>', 'se_mu_d_<mu1>', 'mean_mu_d_<mu2>', 'se_mu_d_<mu2>', ...]
    Returns:
      sigma_d (np.array),
      mu_labels (list of str),
      means (list of np.array),
      ses (list of np.array)
    """
    ncols = len(header)
    if ncols < 2:
        raise ValueError("CSV must have at least sigma_d and one mean column")

    try:
        sigma_d = np.array([float(row[0]) for row in data], dtype=float)
    except Exception as e:
        raise ValueError(f"Failed to parse sigma_d: {e}")

    mu_labels = []
    means = []
    ses = []

    col = 1
    while col < ncols:
        mean_label = header[col].strip()
        se_label = header[col+1].strip() if (col+1) < ncols else None

        if mean_label.startswith("mean_mu_d_"):
            mu_label = mean_label.replace("mean_mu_d_", "")
        else:
            mu_label = mean_label

        # parse mean column
        mean_vals = []
        for r,row in enumerate(data, start=1):
            try:
                mean_vals.append(float(row[col]))
            except Exception as e:
                raise ValueError(f"Failed parsing mean at row {r}, col {col}: {e}")
        mean_arr = np.array(mean_vals, dtype=float)

        # parse se column if present
        if se_label is not None and (col+1) < ncols:
            se_vals = []
            for r,row in enumerate(data, start=1):
                try:
                    se_vals.append(float(row[col+1]))
                except Exception:
                    # if missing or unparsable, set 0
                    se_vals.append(0.0)
            se_arr = np.array(se_vals, dtype=float)
        else:
            se_arr = np.zeros_like(mean_arr)

        mu_labels.append(mu_label)
        means.append(mean_arr)
        ses.append(se_arr)

        col += 2

    return sigma_d, mu_labels, means, ses

def plot_csv_to_png(csv_path, out_png_path=None, dpi=150, figsize=(10,6), quiet=False):
    header, data = read_csv_file(csv_path)
    if not data:
        raise ValueError("CSV has header but no data rows")
    sigma_d, mu_labels, means, ses = parse_csv(header, data)

    fig, ax = plt.subplots(figsize=figsize)
    color_cycle = plt.rcParams.get('axes.prop_cycle').by_key().get('color', None)

    for j, mu_label in enumerate(mu_labels):
        m = means[j]
        s = ses[j]
        s = np.where(np.isfinite(s) & (s >= 0), s, 0.0)
        color = color_cycle[j % len(color_cycle)] if color_cycle else None
        ax.errorbar(sigma_d, m, yerr=s, fmt='o-', capsize=4, label=f"mu_d={mu_label}", color=color, markersize=4)

    ax.set_title(f"Survival Rate vs Sigma_d ({os.path.basename(csv_path)})")
    ax.set_xlabel("Sigma_d")
    ax.set_ylabel("Survival Rate")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best', fontsize='small')

    if out_png_path is None:
        out_png_path = os.path.splitext(csv_path)[0] + ".png"
    os.makedirs(os.path.dirname(out_png_path) or ".", exist_ok=True)
    fig.savefig(out_png_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    if not quiet:
        print(f"Saved: {out_png_path}")
    return out_png_path

def process_all(csv_path, recursive=True, out_subdir="output_plots", dpi=150, figsize=(10,6)):
    files = find_csv_files(csv_path, recursive=recursive)
    if not files:
        print("No CSV files found.")
        return []

    outputs = []
    for f in files:
        try:
            csv_dir = os.path.dirname(f) or "."
            out_dir = os.path.join(csv_dir, out_subdir)
            os.makedirs(out_dir, exist_ok=True)
            out_png = os.path.join(out_dir, os.path.splitext(os.path.basename(f))[0] + ".png")
            plot_csv_to_png(f, out_png_path=out_png, dpi=dpi, figsize=figsize, quiet=True)
            print(f"Processed: {f} -> {out_png}")
            outputs.append(out_png)
        except Exception as e:
            print(f"Error processing {f}: {e}", file=sys.stderr)
    return outputs

def main():
    global csv_path, recursive, out_subdir, fig_width, fig_height, dpi

    outs = process_all(csv_path, recursive=recursive, out_subdir=out_subdir, dpi=dpi, figsize=(fig_width, fig_height))
    print("Done. Generated files:", len(outs))

if __name__ == '__main__':
    main()