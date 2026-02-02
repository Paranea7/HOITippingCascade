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
import matplotlib as mpl
from itertools import cycle  # 引入 cycle 用于循环样式

# ==========================================
# 🚀 PRL 风格全局设置 (已整合所有修改)
# ==========================================
mpl.rcParams.update({
    "text.usetex": False,  # 设为 False 以避免 LaTeX 环境问题
    "font.family": "serif",  # 使用衬线字体
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],  # 优先使用 Times New Roman
    "mathtext.fontset": "stix",  # 使用 STIX 字体渲染公式，效果最像 LaTeX
    "font.size": 10,  # 全局基础字号 (PRL 正文约 10pt)
    "axes.labelsize": 10,  # 轴标签字号
    "xtick.labelsize": 8,  # x轴刻度字号
    "ytick.labelsize": 8,  # y轴刻度字号
    "legend.fontsize": 8,  # 图例字号
    "xtick.direction": "in",  # 刻度向内
    "ytick.direction": "in",  # 刻度向内

    # --- [关键修改] 刻度只显示在左侧和底部 ---
    "xtick.top": False,  # 关闭顶部刻度线
    "ytick.right": False,  # 关闭右侧刻度线
    "xtick.bottom": True,  # 开启底部刻度线
    "ytick.left": True,  # 开启左侧刻度线
    # ----------------------------------------

    "axes.linewidth": 0.8,  # 轴线宽度
    "xtick.major.width": 0.8,  # 主要刻度线宽度
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,  # 主要刻度线长度
    "ytick.major.size": 3,
    "axes.grid": False,  # 默认不显示网格线

    "figure.dpi": 300,  # 高 DPI 预览
    "savefig.bbox": "tight",  # 保存时去除多余白边
    "savefig.pad_inches": 0.05  # 紧凑保存，留一点点边距
})

# ========================
# 🚀 默认配置（无需命令行输入）
# ========================
CSV_DIR_DEFAULT = "outputcsvd0"
OUT_DIR_DEFAULT = "compare_plotsrho0_prl_style"  # 修改输出目录，避免覆盖
FNAME_RE = re.compile(r"s_(?P<s>\d+)_mue_(?P<mu_e>[\d\.]+)_sigmae_(?P<sigma_e>[\d\.]+)\.csv")

# 图像设置
SHOW_SE = True  # 是否显示标准误
FILL_SE = True  # 是否填充误差带（半透明）
OVERWRITE = False  # 是否覆盖已有文件（默认不覆盖）

# PRL 风格的线型和标记循环
# 确保即使颜色相同，也能通过线型和标记区分
prl_styles = [
    {'marker': 'o', 'linestyle': '-', 'markersize': 4},
    {'marker': 's', 'linestyle': '--', 'markersize': 4},
    {'marker': '^', 'linestyle': ':', 'markersize': 4},
    {'marker': 'D', 'linestyle': '-.', 'markersize': 4},
    {'marker': 'v', 'linestyle': '-', 'markersize': 4},
    {'marker': 'p', 'linestyle': '--', 'markersize': 4},
    {'marker': '*', 'linestyle': ':', 'markersize': 5},  # 星号稍大
    {'marker': 'h', 'linestyle': '-.', 'markersize': 4},
]


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
        means = np.array([float(r[1 + 2 * i]) for r in rows], dtype=float)
        ses = np.array([float(r[1 + 2 * i + 1]) for r in rows], dtype=float)
        data[mu_d] = (means, ses)

    return sigma_d, data


def plot_series_with_error(x, series, xlabel, title, outpath, show_se=SHOW_SE, fill_se=FILL_SE):
    """
    绘制多条曲线，支持标准误显示与误差带填充
    已根据PRL风格调整
    """
    # PRL 单栏图宽度通常约为 3.375 英寸 (约 8.57 cm)
    # 保持一个合理的宽高比，例如 3:2 或 4:3
    fig_width_prl = 3.375
    fig_height_prl = fig_width_prl * (2 / 3)  # 保持 3:2 的宽高比

    plt.figure(figsize=(fig_width_prl, fig_height_prl))

    style_cycler = cycle(prl_styles)  # 创建样式循环器

    for label, y, yerr in series:
        current_style = next(style_cycler)  # 获取下一个样式

        if show_se and (yerr is not None):
            if fill_se:
                # 绘制主线
                plt.plot(x, y, label=label, **current_style)
                # 填充误差带
                lower = y - yerr
                upper = y + yerr
                # 使用当前线的颜色，但透明度较低
                current_color = plt.gca().lines[-1].get_color()  # 获取刚刚绘制的线的颜色
                plt.fill_between(x, lower, upper, color=current_color, alpha=0.2)
            else:
                # 绘制误差棒
                # fmt 参数需要一个字符串，结合线型和标记
                fmt_str = f"{current_style['linestyle']}{current_style['marker']}"
                plt.errorbar(x, y, yerr=yerr, fmt=fmt_str, capsize=3, label=label,
                             markersize=current_style['markersize'])
        else:
            # 只绘制线
            plt.plot(x, y, label=label, **current_style)

    plt.xlabel(xlabel)
    plt.ylabel("Survival Rate")
    plt.title(title, fontsize=10)  # 标题字号可以稍大
    # plt.grid(True) # PRL 风格通常不显示网格线，已在 rcParams 中设置为 False
    plt.legend(loc='best', frameon=False)  # 图例放置在最佳位置，不带边框
    plt.tight_layout()  # 紧凑布局
    plt.savefig(outpath, format='pdf', bbox_inches='tight')  # PDF 输出
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
    return lookup, sorted(list(s_set)), sorted(list(mu_e_set)), sorted(list(sigma_e_set))  # 确保返回的是列表


def generate_all_comparisons():
    """
    核心函数：生成三类单变量变化的对比图（PDF 格式）
    """
    csv_dir = CSV_DIR_DEFAULT
    out_dir = OUT_DIR_DEFAULT

    # 创建输出目录
    os.makedirs(out_dir, exist_ok=True)

    print(f"🔍 开始读取 CSV 文件：{csv_dir}")
    parsed = find_csv_files(csv_dir)
    if not parsed:
        print("❌ 未找到任何匹配的 CSV 文件。请检查文件命名格式或目录。")
        return []

    lookup, s_all, mu_e_all, sigma_e_all = build_lookup(parsed)
    print(f"✅ 找到文件，s in {s_all}, mu_e in {mu_e_all}, sigma_e in {sigma_e_all}")

    # 缓存数据（避免重复读取）
    cache = {}
    for key, path in lookup.items():
        try:
            sigma_d_vals, data = load_csv_data(path)
        except Exception as e:
            print(f"❌ 读取失败: {path} -> {e}")
            continue
        cache[key] = (sigma_d_vals, data)

    # 获取所有 mu_d 值（统一）
    # 确保 cache 不为空
    if not cache:
        print("❌ 未能成功读取任何 CSV 数据，无法继续。")
        return []

    any_key = next(iter(cache))
    sigma_d_master, data_master = cache[any_key]
    mu_d_values = sorted(list(data_master.keys()))
    print("✅ 检测到 mu_d 值:", mu_d_values)

    generated = []

    # ========================
    # 1️⃣ 比较不同 s（固定 mu_e, sigma_e）
    # ========================
    print("\n📈 生成 '比较不同 s' 的图表...")
    for mu_e in mu_e_all:
        for sigma_e in sigma_e_all:
            for mu_d in mu_d_values:
                series = []
                x_vals = None
                for s in s_all:
                    key = (s, mu_e, sigma_e)
                    # 检查 key 是否在 cache 中，因为有些文件可能读取失败
                    if key not in cache:
                        continue
                    sigma_d_vals, data = cache[key]
                    x_vals = sigma_d_vals
                    means, ses = data[mu_d]
                    series.append((f"s={s}", means, ses if SHOW_SE else None))
                if not series:
                    continue
                outname = f"compare_s_mu_e_{mu_e}_sigmae_{sigma_e}_mu_d_{mu_d}.pdf"
                outpath = os.path.join(out_dir, outname)
                if not OVERWRITE and os.path.exists(outpath):
                    print("📄 已存在，跳过：", outpath)
                    generated.append(outpath)
                else:
                    title = rf"Varying $s$ | $\mu_e={mu_e}$, $\sigma_e={sigma_e}$, $\mu_d={mu_d}$"  # 使用 LaTeX 格式
                    plot_series_with_error(
                        x_vals, series, xlabel=r"$\sigma_d$", title=title,  # 使用 LaTeX 格式
                        outpath=outpath, show_se=SHOW_SE, fill_se=FILL_SE
                    )
                    print("✅ 保存：", outpath)
                    generated.append(outpath)

    # ========================
    # 2️⃣ 比较不同 mu_e（固定 s, sigma_e）
    # ========================
    print("\n📈 生成 '比较不同 mu_e' 的图表...")
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
                    series.append((rf"$\mu_e={mu_e}$", means, ses if SHOW_SE else None))  # 使用 LaTeX 格式
                if not series:
                    continue
                outname = f"compare_mu_e_s_{s}_sigmae_{sigma_e}_mu_d_{mu_d}.pdf"
                outpath = os.path.join(out_dir, outname)
                if not OVERWRITE and os.path.exists(outpath):
                    print("📄 已存在，跳过：", outpath)
                    generated.append(outpath)
                else:
                    title = rf"Varying $\mu_e$ | $s={s}$, $\sigma_e={sigma_e}$, $\mu_d={mu_d}$"  # 使用 LaTeX 格式
                    plot_series_with_error(
                        x_vals, series, xlabel=r"$\sigma_d$", title=title,  # 使用 LaTeX 格式
                        outpath=outpath, show_se=SHOW_SE, fill_se=FILL_SE
                    )
                    print("✅ 保存：", outpath)
                    generated.append(outpath)

    # ========================
    # 3️⃣ 比较不同 sigma_e（固定 s, mu_e）
    # ========================
    print("\n📈 生成 '比较不同 sigma_e' 的图表...")
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
                    series.append((rf"$\sigma_e={sigma_e}$", means, ses if SHOW_SE else None))  # 使用 LaTeX 格式
                if not series:
                    continue
                outname = f"compare_sigma_e_s_{s}_mue_{mu_e}_mu_d_{mu_d}.pdf"
                outpath = os.path.join(out_dir, outname)
                if not OVERWRITE and os.path.exists(outpath):
                    print("📄 已存在，跳过：", outpath)
                    generated.append(outpath)
                else:
                    title = rf"Varying $\sigma_e$ | $s={s}$, $\mu_e={mu_e}$, $\mu_d={mu_d}$"  # 使用 LaTeX 格式
                    plot_series_with_error(
                        x_vals, series, xlabel=r"$\sigma_d$", title=title,  # 使用 LaTeX 格式
                        outpath=outpath, show_se=SHOW_SE, fill_se=FILL_SE
                    )
                    print("✅ 保存：", outpath)
                    generated.append(outpath)

    print("\n🎉 所有图表生成完成！共生成:", len(generated))
    return generated


# ========================
# 🚀 主程序入口
# ========================
if __name__ == "__main__":
    # 建议在运行前，先确认你的 'outputcsvd0' 目录存在，
    # 并且里面的 .csv 文件名格式与 FNAME_RE 正则表达式匹配。
    # 例如: s_1_mue_0.5_sigmae_0.1.csv
    try:
        generate_all_comparisons()
    except Exception as e:
        print("An error occurred during plotting. Please check the error message:")
        print(e)
        print("\nCommon issues:")
        print(
            "1. If 'Failed to process string with tex because latex could not be found' appears, ensure mpl.rcParams['text.usetex'] = False.")
        print(
            "2. If 'findfont: Font family ['Times New Roman'] not found' appears, ensure Times New Roman is installed on your system or choose another serif font.")