#!/usr/bin/env python3
"""
plot_and_compare_all.py

自动读取 outputcsvd 中的仿真 CSV 文件，并生成“单变量变化”的比较图：
 - 比较不同 s（固定 mu_e、sigma_e）
 - 比较不同 mu_e（固定 s、sigma_e）
 - 比较不同 sigma_e（固定 s、mu_e）

默认行为：遍历所有可用的 (s, mu_e, sigma_e) 组合并为每个可用 mu_d 生成比较图。
输出图像保存在 compare_plots 目录中。

注意：脚本假定 CSV 文件名严格为: s_{s}_mue_{mu_e}_sigmae_{sigma_e}.csv
若你的文件命名不同，请修改 FNAME_RE 正则表达式。
"""

import os
import re
import glob
import csv
import argparse
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# 可配置常量
CSV_DIR_DEFAULT = "outputcsvdrandom1"
OUT_DIR_DEFAULT = "compare_plotsrandom1"
FNAME_RE = re.compile(r"s_(?P<s>\d+)_mue_(?P<mu_e>[\d\.]+)_sigmae_(?P<sigma_e>[\d\.]+)\.csv")

def find_csv_files(csv_dir):
    files = glob.glob(os.path.join(csv_dir, "*.csv"))
    parsed = []
    for f in files:
        name = os.path.basename(f)
        m = FNAME_RE.match(name)
        if not m:
            # 忽略不匹配的文件
            continue
        s = int(m.group("s"))
        mu_e = float(m.group("mu_e"))
        sigma_e = float(m.group("sigma_e"))
        parsed.append((f, s, mu_e, sigma_e))
    return parsed

def load_csv_data(path):
    """
    读取 CSV，返回 sigma_d_values (np.array)，和 dict mapping mu_d->(means, ses)
    CSV 格式假设为:
      header: sigma_d, mean_mu_d_0.2, se_mu_d_0.2, mean_mu_d_0.3, se_mu_d_0.3, ...
      rows: 每个 sigma_d 一行
    """
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    sigma_d = np.array([float(r[0]) for r in rows], dtype=float)

    # 解析 mu_d 顺序
    mu_d_list = []
    for col in header[1::2]:
        # 例如 col == "mean_mu_d_0.2"
        try:
            mu_d_val = float(col.split("_")[-1])
        except Exception:
            raise ValueError(f"无法解析 CSV header 列名: {col} in {path}")
        mu_d_list.append(mu_d_val)

    data = {}
    for i, mu_d in enumerate(mu_d_list):
        means = np.array([float(r[1 + 2*i]) for r in rows], dtype=float)
        ses = np.array([float(r[1 + 2*i + 1]) for r in rows], dtype=float)
        data[mu_d] = (means, ses)

    return sigma_d, data

def plot_series_with_error(x, series, xlabel, title, outpath, show_se=True, fill_se=True):
    """
    series: list of tuples (label, y_vals, y_err_or_none)
    show_se: 是否显示标准误
    fill_se: 若 show_se 为 True，是否绘制半透明误差带（否则仅 errorbar）
    """
    plt.figure(figsize=(8,6))
    for label, y, yerr in series:
        if show_se and (yerr is not None):
            if fill_se:
                plt.plot(x, y, '-o', label=label)
                # 填充误差带
                lower = y - yerr
                upper = y + yerr
                plt.fill_between(x, lower, upper, alpha=0.2)
            else:
                plt.errorbar(x, y, yerr=yerr, fmt='-o', capsize=3, label=label)
        else:
            plt.plot(x, y, '-o', label=label)
    plt.xlabel(xlabel)
    plt.ylabel("Survival Rate")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def build_lookup(parsed_files):
    """
    parsed_files: list of (path, s, mu_e, sigma_e)
    返回 dict lookup[(s, mu_e, sigma_e)] = path
    以及 sets of values found for s, mu_e, sigma_e
    """
    lookup = {}
    s_set = set()
    mu_e_set = set()
    sigma_e_set = set()
    for path, s, mu_e, sigma_e in parsed_files:
        key = (s, float(mu_e), float(sigma_e))
        lookup[key] = path
        s_set.add(s)
        mu_e_set.add(float(mu_e))
        sigma_e_set.add(float(sigma_e))
    return lookup, sorted(s_set), sorted(mu_e_set), sorted(sigma_e_set)

def generate_all_comparisons(csv_dir, out_dir, show_se=True, fill_se=True, overwrite=False):
    parsed = find_csv_files(csv_dir)
    if not parsed:
        print("未找到 CSV 文件，目录：", csv_dir)
        return []

    lookup, s_all, mu_e_all, sigma_e_all = build_lookup(parsed)
    print(f"Found files for s in {s_all}, mu_e in {mu_e_all}, sigma_e in {sigma_e_all}")

    generated = []

    # 为每个已知组合读取一次 CSV 并缓存 data，避免重复读取
    cache = {}
    for key, path in lookup.items():
        try:
            sigma_d_vals, data = load_csv_data(path)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue
        cache[key] = (sigma_d_vals, data)

    # 获取所有 mu_d 值（从任一 CSV）
    any_key = next(iter(cache))
    sigma_d_master, data_master = cache[any_key]
    mu_d_values = sorted(list(data_master.keys()))
    print("Detected mu_d values:", mu_d_values)

    # 1) 比较不同 s：对每对 (mu_e, sigma_e) 生成一张图，曲线为不同 s（每条曲线为 sigma_d vs mean）
    for mu_e in mu_e_all:
        for sigma_e in sigma_e_all:
            # 收集所有 s 的数据（若存在 CSV）
            series = []
            x_vals = None
            for s in s_all:
                key = (s, mu_e, sigma_e)
                path = lookup.get(key)
                if path is None:
                    continue
                sigma_d_vals, data = cache[key]
                x_vals = sigma_d_vals
                for mu_d in mu_d_values:
                    means, ses = data[mu_d]
                    # 每个 mu_d 单独保存一张图；为此我们先构建字典：mu_d -> series列表
                    # 收集到 dict 中（延后绘图）
            # 对每 mu_d 单独绘图
            for mu_d in mu_d_values:
                series = []
                x_vals = None
                for s in s_all:
                    key = (s, mu_e, sigma_e)
                    path = lookup.get(key)
                    if path is None:
                        continue
                    sigma_d_vals, data = cache[key]
                    x_vals = sigma_d_vals
                    means, ses = data[mu_d]
                    series.append((f"s={s}", means, ses if show_se else None))
                if not series:
                    continue
                outname = f"compare_s_mu_e_{mu_e}_sigmae_{sigma_e}_mu_d_{mu_d}.png"
                outpath = os.path.join(out_dir, outname)
                if (not overwrite) and os.path.exists(outpath):
                    print("Already exists, skip:", outpath)
                    generated.append(outpath)
                else:
                    title = f"Varying s | mu_e={mu_e}, sigma_e={sigma_e}, mu_d={mu_d}"
                    plot_series_with_error(x_vals, series, xlabel="sigma_d", title=title, outpath=outpath, show_se=show_se, fill_se=fill_se)
                    print("Saved:", outpath)
                    generated.append(outpath)

    # 2) 比较不同 mu_e：对每对 (s, sigma_e) 生成一张图，曲线为不同 mu_e
    for s in s_all:
        for sigma_e in sigma_e_all:
            for mu_d in mu_d_values:
                series = []
                x_vals = None
                for mu_e in mu_e_all:
                    key = (s, mu_e, sigma_e)
                    path = lookup.get(key)
                    if path is None:
                        continue
                    sigma_d_vals, data = cache[key]
                    x_vals = sigma_d_vals
                    means, ses = data[mu_d]
                    series.append((f"mu_e={mu_e}", means, ses if show_se else None))
                if not series:
                    continue
                outname = f"compare_mu_e_s_{s}_sigmae_{sigma_e}_mu_d_{mu_d}.png"
                outpath = os.path.join(out_dir, outname)
                if (not overwrite) and os.path.exists(outpath):
                    print("Already exists, skip:", outpath)
                    generated.append(outpath)
                else:
                    title = f"Varying mu_e | s={s}, sigma_e={sigma_e}, mu_d={mu_d}"
                    plot_series_with_error(x_vals, series, xlabel="sigma_d", title=title, outpath=outpath, show_se=show_se, fill_se=fill_se)
                    print("Saved:", outpath)
                    generated.append(outpath)

    # 3) 比较不同 sigma_e：对每对 (s, mu_e) 生成一张图，曲线为不同 sigma_e
    for s in s_all:
        for mu_e in mu_e_all:
            for mu_d in mu_d_values:
                series = []
                x_vals = None
                for sigma_e in sigma_e_all:
                    key = (s, mu_e, sigma_e)
                    path = lookup.get(key)
                    if path is None:
                        continue
                    sigma_d_vals, data = cache[key]
                    x_vals = sigma_d_vals
                    means, ses = data[mu_d]
                    series.append((f"sigma_e={sigma_e}", means, ses if show_se else None))
                if not series:
                    continue
                outname = f"compare_sigma_e_s_{s}_mue_{mu_e}_mu_d_{mu_d}.png"
                outpath = os.path.join(out_dir, outname)
                if (not overwrite) and os.path.exists(outpath):
                    print("Already exists, skip:", outpath)
                    generated.append(outpath)
                else:
                    title = f"Varying sigma_e | s={s}, mu_e={mu_e}, mu_d={mu_d}"
                    plot_series_with_error(x_vals, series, xlabel="sigma_d", title=title, outpath=outpath, show_se=show_se, fill_se=fill_se)
                    print("Saved:", outpath)
                    generated.append(outpath)

    print("Finished generating plots. Total:", len(generated))
    return generated

def main():
    parser = argparse.ArgumentParser(description="Bulk plot comparisons from simulation CSVs (single-variable changes)")
    parser.add_argument("--csv_dir", default=CSV_DIR_DEFAULT, help="Directory containing CSVs")
    parser.add_argument("--out_dir", default=OUT_DIR_DEFAULT, help="Output directory for generated PNGs")
    parser.add_argument("--no_se", action="store_true", help="Do not plot SE")
    parser.add_argument("--no_fill", action="store_true", help="Do not fill SE bands; use errorbars instead")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNGs")
    args = parser.parse_args()

    csv_dir = args.csv_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    show_se = not args.no_se
    fill_se = not args.no_fill

    generated = generate_all_comparisons(csv_dir, out_dir, show_se=show_se, fill_se=fill_se, overwrite=args.overwrite)
    print("Generated files:", len(generated))
    for p in generated:
        print("  ", p)

if __name__ == "__main__":
    main()