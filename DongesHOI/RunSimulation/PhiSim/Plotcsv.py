#!/usr/bin/env python3
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import cycle


# ==========================================
# 1. PRL 风格全局设置 (Global Style Settings)
# ==========================================
def set_prl_style():
    """配置 Matplotlib 以符合 Physical Review Letters 风格"""
    mpl.rcParams.update({
        # 字体设置
        "text.usetex": False,  # 如果系统有 LaTeX 可设为 True，否则用 mathtext
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",  # 类似 LaTeX 的数学字体

        # 字号设置 (适配单栏宽度)
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,

        # 刻度设置 (朝内，精细)
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,  # 顶部显示刻度线
        "ytick.right": True,  # 右侧显示刻度线
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,

        # 线条与布局
        "axes.linewidth": 0.8,
        "axes.grid": False,  # PRL 通常不加网格
        "lines.linewidth": 1.2,
        "lines.markersize": 4,

        # 保存设置
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "figure.dpi": 300  # 预览用高 DPI
    })


# ==========================================
# 2. 数据处理逻辑 (保持不变)
# ==========================================
def load_csv(path):
    mu_d_vals = []
    sigma_d_vals = []
    means = {}
    ses = {}

    with open(path) as f:
        r = csv.reader(f)
        header = next(r)

        # 解析 header 提取 mu_d 值
        mu_d_vals = sorted(list(set(float(h.split("_")[-1])
                                    for h in header if h.startswith("mean"))))

        for mu in mu_d_vals:
            means[mu] = []
            ses[mu] = []

        for row in r:
            if not row: continue  # 跳过空行
            sigma_d_vals.append(float(row[0]))
            idx = 1
            for mu in mu_d_vals:
                means[mu].append(float(row[idx]));
                idx += 1
                ses[mu].append(float(row[idx]));
                idx += 1

    return np.array(sigma_d_vals), mu_d_vals, means, ses


# ==========================================
# 3. 绘图逻辑 (PRL 适配版)
# ==========================================
def plot_csv_prl(csv_path, out_dir):
    sigma_d_vals, mu_d_vals, means, ses = load_csv(csv_path)

    os.makedirs(out_dir, exist_ok=True)
    # 推荐保存为 PDF (矢量图) 用于投稿，PNG 用于预览
    fname_base = os.path.splitext(os.path.basename(csv_path))[0]
    out_path_pdf = os.path.join(out_dir, fname_base + ".pdf")

    # PRL 单栏宽度约 3.375 英寸 (8.6 cm)
    # 宽高比通常设为 4:3 或 黄金分割
    fig_width = 3.375
    fig_height = fig_width * 0.75

    plt.figure(figsize=(fig_width, fig_height))

    # 定义样式循环，确保黑白打印也能区分 (线型 + 标记)
    # 颜色使用高对比度色系
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', '<']
    linestyles = ['-', '--', '-.', ':', '-', '--']

    style_cycler = cycle(zip(colors, markers, linestyles))

    for mu in mu_d_vals:
        c, m, l = next(style_cycler)

        # 使用 LaTeX 格式的 label
        label_str = r"$\mu_d = {}$".format(mu)

        plt.errorbar(sigma_d_vals, means[mu], yerr=ses[mu],
                     fmt=l,  # 线型
                     marker=m,  # 标记
                     color=c,  # 颜色
                     markerfacecolor='none',  # 空心标记更显清爽 (可选)
                     markeredgewidth=0.8,
                     capsize=2,  # 误差棒帽子小一点
                     label=label_str,
                     alpha=0.9)

    # 轴标签使用 LaTeX 格式
    plt.xlabel(r"$\sigma_d$")
    plt.ylabel(r"Tipping Rate  $\phi$")  # 或 r"$P_{\mathrm{surv}}$"

    # 移除标题 (PRL 图表标题通常在 Caption 中，不在图内)
    # 如果必须区分不同文件，建议在图内加个小标签 (text box)
    # plt.title(fname_base)

    # 图例设置：去边框，紧凑，自动寻找最佳位置
    plt.legend(frameon=False, loc='best', handlelength=2.5, labelspacing=0.3)

    # 调整布局
    plt.tight_layout(pad=0.2)

    # 保存
    plt.savefig(out_path_pdf, format='pdf')
    print(f"Saved PRL plot: {out_path_pdf}")
    plt.close()


def main():
    # ------- 配置 -------
    csv_dir = "csv_output_fixed_c_no_self"
    out_dir = "plots_prl"  # 修改输出目录以区分
    # -------------------

    # 应用样式
    set_prl_style()

    if not os.path.exists(csv_dir):
        print(f"Error: Directory '{csv_dir}' not found.")
        return

    files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]

    if not files:
        print(f"No csv files found in {csv_dir}")
        return

    print(f"Found {len(files)} files. Processing...")
    for f in files:
        try:
            plot_csv_prl(os.path.join(csv_dir, f), out_dir)
        except Exception as e:
            print(f"Failed to plot {f}: {e}")


if __name__ == "__main__":
    main()