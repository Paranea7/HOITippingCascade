#!/usr/bin/env python3
"""
plot_and_compare_all_pdf.py

自动读取 outputcsvd 中的仿真 CSV 文件，并生成“单变量变化”的比较图（PDF 格式）：
 - 比较不同 s（固定 mu_e、sigma_e）
 - 比较不同 mu_e（固定 s、sigma_e）
 - 比较不同 sigma_e（固定 s、mu_e）

默认行为：遍历所有可用的 (s, mu_e, sigma_e) 组合并为每个可用 mu_d 生成比较图。
输出图像保存在 compare_plotsrho0 目录中（自动创建）。

脚本无需命令行输入，所有路径和参数均在程序中配置。
文件命名规则：s_{s}_mue_{mu_e}_sigmae_{sigma_e}.csv
若命名不同，请修改 FNAME_RE 正则表达式。
"""

import os
import re
import glob
import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# ========================
# 🚀 默认配置（无需命令行输入）
# ========================
CSV_DIR_DEFAULT = "csv_output"
OUT_DIR_DEFAULT = "compare_plotsrandom0"
FNAME_RE = re.compile(r"s_(?P<s>\d+)_mue_(?P<mu_e>[\d\.]+)_sigmae_(?P<sigma_e>[\d\.]+)\.csv")

# 图像设置
SHOW_SE = True      # 是否显示标准误
FILL_SE = True      # 是否填充误差带（半透明）
OVERWRITE = False   # 是否覆盖已有文件（默认不覆盖）

def find_csv_files(csv_dir):
    """查找目录中所有匹配命名规则的 CSV 文件"""
    files = glob.glob(os.path.join(csv_dir, "*.csv"))
    parsed = []
    for f in files:
        name = os.path.basename(f)
        m = FNAME_RE.match(name)
        if not m:
            continue
        s = int(m.group("s"))
        mu_e = float(m.group("mu_e"))
        sigma_e = float(m.group("sigma_e"))
        parsed.append((f, s, mu_e, sigma_e))
    return parsed

def load_csv_data(path):
    """
    读取 CSV，返回 sigma_d_values 和 dict mapping mu_d -> (means, ses)
    CSV 格式：
      header: sigma_d, mean_mu_d_0.2, se_mu_d_0.2, mean_mu_d_0.3, se_mu_d_0.3, ...
      rows: 每个 sigma_d 一行
    """
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    sigma_d = np.array([float(r[0]) for r in rows], dtype=float)

    # 解析 mu_d 列
    mu_d_list = []
    for col in header[1::2]:
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

def plot_series_with_error(x, series, xlabel, title, outpath, show_se=SHOW_SE, fill_se=FILL_SE):
    """
    绘制多条曲线，支持标准误显示与误差带填充
    """
    plt.figure(figsize=(8, 6))
    for label, y, yerr in series:
        if show_se and (yerr is not None):
            if fill_se:
                plt.plot(x, y, '-o', label=label)
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
    plt.savefig(outpath, format='pdf', bbox_inches='tight')  # ✅ PDF 输出
    plt.close()

def build_lookup(parsed_files):
    """
    构建参数到路径的映射，同时收集所有 s, mu_e, sigma_e 值
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

def generate_all_comparisons():
    csv_dir = CSV_DIR_DEFAULT
    out_dir = OUT_DIR_DEFAULT

    # 主输出目录
    os.makedirs(out_dir, exist_ok=True)

    # 创建三个子目录
    out_dir_s = os.path.join(out_dir, "compare_s")
    out_dir_mu_e = os.path.join(out_dir, "compare_mu_e")
    out_dir_sigma_e = os.path.join(out_dir, "compare_sigma_e")
    os.makedirs(out_dir_s, exist_ok=True)
    os.makedirs(out_dir_mu_e, exist_ok=True)
    os.makedirs(out_dir_sigma_e, exist_ok=True)

    print(f"🔍 开始读取 CSV 文件：{csv_dir}")
    parsed = find_csv_files(csv_dir)
    if not parsed:
        print("❌ 未找到任何匹配的 CSV 文件。")
        return []

    lookup, s_all, mu_e_all, sigma_e_all = build_lookup(parsed)
    print(f"✅ 参数空间：s={s_all}, mu_e={mu_e_all}, sigma_e={sigma_e_all}")

    # 缓存数据
    cache = {}
    for key, path in lookup.items():
        sigma_d_vals, data = load_csv_data(path)
        cache[key] = (sigma_d_vals, data)

    any_key = next(iter(cache))
    sigma_d_master, data_master = cache[any_key]
    mu_d_values = sorted(list(data_master.keys()))
    print("✅ mu_d 值:", mu_d_values)

    generated = []

    # ========================
    # 1️⃣ 比较不同 s
    # ========================
    for mu_e in mu_e_all:
        for sigma_e in sigma_e_all:
            for mu_d in mu_d_values:
                series = []
                x_vals = None
                for s in s_all:
                    key = (s, mu_e, sigma_e)
                    if key not in cache:
                        continue
                    sigma_d_vals, data = cache[key]
                    x_vals = sigma_d_vals
                    means, ses = data[mu_d]
                    series.append((f"s={s}", means, ses if SHOW_SE else None))
                if not series:
                    continue

                outname = f"compare_s_mu_e_{mu_e}_sigmae_{sigma_e}_mu_d_{mu_d}.pdf"
                outpath = os.path.join(out_dir_s, outname)

                if not OVERWRITE and os.path.exists(outpath):
                    generated.append(outpath)
                else:
                    title = f"Varying s | mu_e={mu_e}, sigma_e={sigma_e}, mu_d={mu_d}"
                    plot_series_with_error(x_vals, series, "sigma_d", title, outpath)
                    generated.append(outpath)

    # ========================
    # 2️⃣ 比较不同 mu_e
    # ========================
    for s in s_all:
        for sigma_e in sigma_e_all:
            for mu_d in mu_d_values:
                series = []
                x_vals = None
                for mu_e in mu_e_all:
                    key = (s, mu_e, sigma_e)
                    if key not in cache:
                        continue
                    sigma_d_vals, data = cache[key]
                    x_vals = sigma_d_vals
                    means, ses = data[mu_d]
                    series.append((f"mu_e={mu_e}", means, ses if SHOW_SE else None))
                if not series:
                    continue

                outname = f"compare_mu_e_s_{s}_sigmae_{sigma_e}_mu_d_{mu_d}.pdf"
                outpath = os.path.join(out_dir_mu_e, outname)

                if not OVERWRITE and os.path.exists(outpath):
                    generated.append(outpath)
                else:
                    title = f"Varying mu_e | s={s}, sigma_e={sigma_e}, mu_d={mu_d}"
                    plot_series_with_error(x_vals, series, "sigma_d", title, outpath)
                    generated.append(outpath)

    # ========================
    # 3️⃣ 比较不同 sigma_e
    # ========================
    for s in s_all:
        for mu_e in mu_e_all:
            for mu_d in mu_d_values:
                series = []
                x_vals = None
                for sigma_e in sigma_e_all:
                    key = (s, mu_e, sigma_e)
                    if key not in cache:
                        continue
                    sigma_d_vals, data = cache[key]
                    x_vals = sigma_d_vals
                    means, ses = data[mu_d]
                    series.append((f"sigma_e={sigma_e}", means, ses if SHOW_SE else None))
                if not series:
                    continue

                outname = f"compare_sigma_e_s_{s}_mue_{mu_e}_mu_d_{mu_d}.pdf"
                outpath = os.path.join(out_dir_sigma_e, outname)

                if not OVERWRITE and os.path.exists(outpath):
                    generated.append(outpath)
                else:
                    title = f"Varying sigma_e | s={s}, mu_e={mu_e}, mu_d={mu_d}"
                    plot_series_with_error(x_vals, series, "sigma_d", title, outpath)
                    generated.append(outpath)

    print("🎉 输出完成，共生成:", len(generated))
    return generated

# ========================
# 🚀 主程序入口
# ========================
if __name__ == "__main__":
    generate_all_comparisons()
