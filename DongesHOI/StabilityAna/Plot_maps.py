#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import BoundaryNorm
import matplotlib as mpl
import matplotlib.ticker as ticker  # 引入 ticker 用于控制刻度数量

# ==========================================
# 1. PRL 风格全局设置 (已整合所有修改)
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
    "legend.fontsize": 8,  # 图例字号 (如果需要)
    "xtick.direction": "in",  # 刻度向内
    "ytick.direction": "in",  # 刻度向内

    # --- [关键修改] 刻度只显示在左侧和底部 ---
    "xtick.top": False,  # 关闭顶部刻度线
    "ytick.right": False,  # 关闭右侧刻度线
    "xtick.bottom": True,  # 开启底部刻度线
    "ytick.left": True,  # 开启左侧刻度线
    # ----------------------------------------

    "figure.dpi": 300,  # 高 DPI 预览
    "savefig.bbox": "tight",  # 保存时去除多余白边
    "savefig.pad_inches": 0.05  # 紧凑保存，留一点点边距
})

# --- 参数：d12 升序；d21 负号保留 ---
d12_vals = [0.0, 0.2, 0.3, 0.5, 0.7, 0.9]
d21_vals = [0.0, -0.2, -0.3, -0.5, -0.7, -0.9]

c1_min, c1_max = -0.8, 0.8
c2_min, c2_max = -0.8, 0.8
MAX_STABLE_DISPLAY = 4

cmap = plt.get_cmap("viridis", MAX_STABLE_DISPLAY + 1)
bounds = np.arange(-0.5, MAX_STABLE_DISPLAY + 1.5, 1.0)
norm = BoundaryNorm(bounds, cmap.N, clip=True)

OUTDIR = "stability_results_prl_style2"  # 修改输出目录，避免覆盖


def load_map(d12, d21):
    """
    加载稳定性数据文件。
    如果文件不存在，则生成随机数据用于演示。
    """
    # 构造文件名，确保 d21 的负号在文件名中是 '-'
    fn_original_format = os.path.join("stability_results_3sys_threebody_fixE", f"stabmap_d12_{d12:.1f}_d21_{d21:.1f}.npy")

    if os.path.exists(fn_original_format):
        return np.load(fn_original_format)
    else:
        print(f"Warning: {fn_original_format} not found, using random data for demo.")
        # 生成随机数据，确保数据范围符合 MAX_STABLE_DISPLAY
        return np.random.randint(0, MAX_STABLE_DISPLAY + 1, (200, 200))


def make_combined_prl_pdf():
    """
    生成符合PRL风格的组合稳定性图PDF。
    """
    rows = len(d12_vals)
    cols = len(d21_vals)

    # PRL 双栏图宽度通常约为 7 英寸 (约 17.8 cm)
    # 对于 6x6 的网格，高度也应接近宽度，以保持子图的合理比例
    fig_width = 7.0
    # 经验值调整，确保所有标签和 colorbar 有足够空间
    fig_height = fig_width * (rows / cols) * 1.05 + 0.5  # 额外高度给 colorbar 和上下边距

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(fig_width, fig_height),
        sharex=True, sharey=True,
        gridspec_kw={'wspace': 0.05, 'hspace': 0.05}  # 极小的间距
    )
    # 确保 axes 是二维数组，即使只有一行或一列
    axes = np.atleast_2d(axes)

    # 定义需要加红框的索引 (行索引, 列索引)
    highlight_indices = [(0, 0), (1, 1), (4, 4), (5, 5)]

    # ========== 绘制所有子图 ==========
    for i, d12 in enumerate(d12_vals):
        for j, d21 in enumerate(d21_vals):
            ax = axes[i, j]
            sm = load_map(d12, d21)

            ax.imshow(
                sm, extent=(c1_min, c1_max, c2_min, c2_max),
                origin="lower", cmap=cmap, norm=norm, aspect='auto'
            )

            # --- [关键修改] 解决刻度冲突：强制减少刻度数量 ---
            # nbins=3 表示最多显示 3 个主要刻度，防止太密挤在一起
            # prune='both' 会去掉两端的刻度（例如最小值和最大值），这在紧凑布局中非常有用，
            # 可以防止第一个子图的右刻度和第二个子图的左刻度打架。
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune='both'))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune='both'))
            # --------------------------------------------------

            # --- [修改] 添加红框逻辑 ---
            if (i, j) in highlight_indices:
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(1.8)  # 调整线宽，使其显眼但不夸张
                    spine.set_zorder(10)  # 确保红框绘制在最上层
            # -------------------------

            # --- 轴标签和刻度标签控制 ---
            # 只有最底行显示 x 轴标签和刻度值
            if i == rows - 1:
                ax.set_xlabel(r"$c_1$", fontsize=10)
                ax.tick_params(axis='x', labelbottom=True)  # 确保底部刻度标签显示
            else:
                ax.tick_params(axis='x', labelbottom=False)  # 隐藏内部子图的 x 刻度值

            # 只有最左列显示 y 轴标签和刻度值
            if j == 0:
                ax.set_ylabel(r"$c_2$", fontsize=10)
                ax.tick_params(axis='y', labelleft=True)  # 确保左侧刻度标签显示
            else:
                ax.tick_params(axis='y', labelleft=False)  # 隐藏内部子图的 y 刻度值

            # 顶部列标题 (d21)
            if i == 0:  # 仅在顶行设置列标题
                ax.set_title(rf"$d_{{21}}={d21:.1f}$", fontsize=9, pad=5)

            # 统一刻度线长度和确保只显示左下刻度
            ax.tick_params(axis='both', which='major', length=4, top=False, right=False, left=True, bottom=True)

    # ========== 整体布局调整 ==========
    # 调整整个图的边距，为 colorbar 和行/列标签留出空间
    fig.subplots_adjust(
        left=0.10, right=0.98,  # 增加左侧空间给 d12 标签
        bottom=0.15, top=0.90,  # 增加底部空间给 colorbar，顶部空间给 d21 标题
        wspace=0.05, hspace=0.05  # 保持子图间距小
    )

    # ========== 左侧行标签 (d12) ==========
    # 计算每行子图的中心 Y 坐标 (在 figure 坐标系中)
    for i, d12_val in enumerate(d12_vals):
        # 获取该行第一个子图的 bounding box
        bbox = axes[i, 0].get_position()
        # 计算该行子图的中心 Y 坐标
        mid_y = bbox.y0 + bbox.height / 2

        fig.text(
            0.01,  # 距离左边缘的距离，可以根据需要调整
            mid_y,
            rf"$d_{{12}}={d12_val:.1f}$",
            ha="left", va="center",  # 左对齐，垂直居中
            rotation="vertical",  # 垂直显示
            fontsize=9,  # 字体大小与列标题一致
            transform=fig.transFigure  # 使用 figure 坐标系
        )

    # ========== Colorbar ==========
    # 精确计算 colorbar 的位置和宽度，使其与子图网格对齐
    # 获取最底行子图的左边界和右边界
    left_bound = axes[-1, 0].get_position().x0
    right_bound = axes[-1, -1].get_position().x1

    cax_width = right_bound - left_bound
    cax_height = 0.02  # colorbar 高度
    cax_y_pos = 0.06  # colorbar 底部位置

    cax = fig.add_axes([left_bound, cax_y_pos, cax_width, cax_height])

    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax, orientation='horizontal',
        boundaries=bounds,
        ticks=np.arange(0, MAX_STABLE_DISPLAY + 1)
    )
    cb.ax.tick_params(labelsize=9)  # colorbar 刻度字号
    cb.set_label(r'Number of Stable Fixed Points ($N_{st}$)', fontsize=10)  # colorbar 标签字号

    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)

    out_pdf = os.path.join(OUTDIR, "combined_stabmap_prl_style_clean_ticks.pdf")
    plt.savefig(out_pdf)  # savefig.bbox='tight' 已经在 rcParams 设置
    plt.close(fig)
    print("Saved:", out_pdf)


if __name__ == "__main__":
    # 建议在运行前，先确认你的 'stability_results' 目录存在，
    # 并且里面的 .npy 文件名格式与 load_map 函数中构造的 fn_original_format 匹配。
    # 例如: stabmap_d12_0.0_d21_0.0.npy
    # 如果文件不存在，load_map 会生成随机数据。
    try:
        make_combined_prl_pdf()
    except Exception as e:
        print("An error occurred during plotting. Please check the error message:")
        print(e)
        print("\nCommon issues:")
        print(
            "1. If 'Failed to process string with tex because latex could not be found' appears, set mpl.rcParams['text.usetex'] = False.")
        print(
            "2. If 'findfont: Font family ['Times New Roman'] not found' appears, ensure Times New Roman is installed on your system or choose another serif font.")