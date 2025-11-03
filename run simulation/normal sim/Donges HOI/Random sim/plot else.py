#!/usr/bin/env python3
"""
plot_multi_axes.py

功能：
- 遍历指定目录或处理单个 CSV 文件。
- 从每个 CSV 中读取 survival 数据（列为：sigma_d, mean_mu_d_<...>, se_mu_d_<...>, ...）
- 根据文件名或 header 提取参数 mu_d / mu_e / sigma_e（用于横轴）。
- 为每个 CSV 生成三类图（横轴分别为 mu_d、mu_e、sigma_e），保存到 CSV 所在目录下的子目录 out_subdir。

使用：
- 编辑脚本顶部设置 csv_path（文件或目录）、recursive（是否递归）、out_subdir、合并选项等。
- 运行： python plot_multi_axes.py
"""

import os
import re
import csv
import sys
import fnmatch
import math
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# ========== 配置区 ==========
# csv_path 可以是单个 CSV 文件或目录
csv_path = "outputcsvdrandom1"

# 是否递归搜索目录
recursive = True

# 输出子目录（相对于每个 CSV 文件所在目录）
out_subdir = "output_plotselse"

# 图像尺寸与分辨率
figsize = (10, 6)
dpi = 150

# 是否把所有 CSV 的同类图合并成一张大图（按横轴变量合并）
make_combined_plots = False

# 解析文件名参数的正则（根据你的文件命名调整）
# 例子能匹配: ..._mud_0.2_mue_0.5_sigmae_0.1.csv 或 mud0.2-mue0.5-sigmae0.1.csv 等
FILENAME_PARAM_RE = re.compile(
    r"(?:mud[_\-]?|mu_d[_\-]?|muD[_\-]?)(?P<mud>-?\d+(\.\d+)?([eE][-+]?\d+)?)|"
    r"(?:mue[_\-]?|mu_e[_\-]?|muE[_\-]?)(?P<mue>-?\d+(\.\d+)?([eE][-+]?\d+)?)|"
    r"(?:sigmae[_\-]?|sigma_e[_\-]?)(?P<sigmae>-?\d+(\.\d+)?([eE][-+]?\d+)?)",
    re.IGNORECASE
)
# ===========================

def find_csv_files(path, recursive=True, pattern="*.csv"):
    path = os.path.abspath(path)
    if os.path.isfile(path):
        if fnmatch.fnmatch(os.path.basename(path), pattern):
            return [path]
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
            if fnmatch.fnmatch(name, pattern):
                matches.append(os.path.join(path, name))
    return matches

def read_csv(path):
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError("CSV empty or missing header: " + path)
        data = [row for row in reader if row and not all(cell.strip()=='' for cell in row)]
    return header, data

def parse_csv_data(header, data):
    # header: list of column names
    # data: list of rows (strings)
    ncols = len(header)
    if ncols < 2:
        raise ValueError("CSV must have at least two columns (sigma_d and one mean)")
    # parse sigma_d (first column)
    sigma_d = np.array([float(row[0]) for row in data], dtype=float)

    # find mean/se column pairs
    means = []   # list of arrays
    ses = []
    labels = []  # label for each pair, e.g. mu_d=0.2  or simply mu_d_0.2
    col = 1
    while col < ncols:
        mean_label = header[col].strip()
        se_label = header[col+1].strip() if (col+1) < ncols else None

        # label extraction
        if mean_label.startswith("mean_mu_d_"):
            lbl = mean_label.replace("mean_mu_d_", "")
        elif mean_label.startswith("mean_mu_e_"):
            lbl = mean_label.replace("mean_mu_e_", "")
        else:
            lbl = mean_label

        # parse mean values
        mean_vals = []
        for r,row in enumerate(data, start=1):
            try:
                mean_vals.append(float(row[col]))
            except Exception as e:
                raise ValueError(f"Failed to parse mean at row {r}, col {col}: {e}")
        mean_arr = np.array(mean_vals, dtype=float)

        # parse se values if exists
        if se_label is not None and (col+1) < ncols:
            se_vals = []
            for r,row in enumerate(data, start=1):
                try:
                    se_vals.append(float(row[col+1]))
                except Exception:
                    se_vals.append(0.0)
            se_arr = np.array(se_vals, dtype=float)
        else:
            se_arr = np.zeros_like(mean_arr)

        labels.append(lbl)
        means.append(mean_arr)
        ses.append(se_arr)
        col += 2

    return sigma_d, labels, means, ses

def extract_params_from_filename(fn):
    # 返回字典，可能包含 keys: mud, mue, sigmae （字符串形式）
    base = os.path.basename(fn)
    name = os.path.splitext(base)[0]
    params = {}
    # 找所有匹配
    for m in FILENAME_PARAM_RE.finditer(name):
        gd = m.groupdict()
        if gd.get("mud"):
            params["mud"] = gd["mud"]
        if gd.get("mue"):
            params["mue"] = gd["mue"]
        if gd.get("sigmae"):
            params["sigmae"] = gd["sigmae"]
    return params

def try_extract_from_labels(labels):
    # 如果 labels 中形如 "0.2" 或 "mu_d=0.2" 等，尝试提取
    res = {}
    for lbl in labels:
        s = str(lbl)
        # 数字出现则可能是 mu 值
        m = re.search(r"(-?\d+(\.\d+)?([eE][-+]?\d+)?)", s)
        if m:
            val = m.group(1)
            # 根据 label 中是否含 mu_e 或 mu_d 做区分
            if "mu_e" in s or "mue" in s:
                res.setdefault("mue", []).append(val)
            elif "mu_d" in s or "mud" in s:
                res.setdefault("mud", []).append(val)
            else:
                # 不确定的话放到 generic
                res.setdefault("generic", []).append(val)
    return res

def plot_single(csv_file, out_dir, figsize=figsize, dpi=dpi):
    header, data = read_csv(csv_file)
    sigma_d, labels, means, ses = parse_csv_data(header, data)
    # try to get params from filename first
    params = extract_params_from_filename(csv_file)
    label_params = try_extract_from_labels(labels)

    # build a metadata dict for this CSV
    meta = {
        "file": csv_file,
        "params_from_fname": params,
        "labels": labels,
        "labels_params": label_params
    }

    # We will produce three kinds of plots:
    #  A: x axis = mu_d (if we can extract mu_d values)
    #  B: x axis = mu_e (if we can extract mu_e values)
    #  C: x axis = sigma_e (from filename or labels if available)
    # For each plot, if we cannot determine a numeric x for this CSV, skip that plot.

    # Determine candidate x values:
    # 1) from filename params (single value)
    # 2) from labels (list)
    # 3) If none available, skip

    # Helper to convert strings list to floats if possible
    def to_floats(lst):
        if not lst:
            return []
        out = []
        for v in lst:
            try:
                out.append(float(v))
            except Exception:
                continue
        return out

    mud_from_fname = to_floats([params["mud"]]) if params.get("mud") else []
    mue_from_fname = to_floats([params["mue"]]) if params.get("mue") else []
    sigmae_from_fname = to_floats([params["sigmae"]]) if params.get("sigmae") else []

    mud_from_labels = to_floats(label_params.get("mud", []))
    mue_from_labels = to_floats(label_params.get("mue", []))
    sigmae_from_labels = to_floats(label_params.get("sigmae", []))  # rarely present

    # For CSVs that actually encode multiple mu values across columns, labels list may represent them.
    # We'll treat labels[] as the x values for one-dimension plotting if they are numeric.
    labels_numeric = to_floats(labels)

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # Prepare color cycle
    color_cycle = plt.rcParams.get('axes.prop_cycle').by_key().get('color', None)

    outputs = []

    # Helper to draw a plot where xvals is array-like and y is one mean (with se)
    def draw_plot(xvals, yvals_list, yerrs_list, series_labels, xlabel, outname):
        if len(xvals) == 0:
            return None
        x = np.array(xvals, dtype=float)
        # sort by x for nicer lines
        order = np.argsort(x)
        x_sorted = x[order]
        fig, ax = plt.subplots(figsize=figsize)
        for i, (yvals, yerrs, s_lbl) in enumerate(zip(yvals_list, yerrs_list, series_labels)):
            y = np.array(yvals, dtype=float)
            e = np.array(yerrs, dtype=float)
            # if y has same length as x? if not, try to broadcast or skip
            if y.shape != x.shape:
                # if y is a curve over sigma_d, take mean across sigma_d to get single point per file
                # but in many cases user's CSV stores survival across sigma_d: then we cannot map to single x.
                # We'll attempt to take mean of y as single scalar per series.
                if y.size == 1:
                    y_pts = np.full_like(x_sorted, float(y))
                    e_pts = np.full_like(x_sorted, float(e)) if e.size==1 else np.zeros_like(x_sorted)
                else:
                    # Attempt to compute mean and use that single y for plotting single x point
                    mean_y = float(np.nanmean(y))
                    mean_e = float(np.nanmean(e)) if e.size>0 else 0.0
                    # produce single point per x value? No — better to plot a single marker at x position(s).
                    y_pts = np.array([mean_y])
                    e_pts = np.array([mean_e])
                    x_plot = np.array([x_sorted[0]]) if x_sorted.size>0 else np.array([np.nan])
                    ax.errorbar(x_plot, y_pts, yerr=e_pts, fmt='o', capsize=4, label=s_lbl, color=color_cycle[i%len(color_cycle)] if color_cycle else None)
                    continue
            else:
                y_pts = y[order]
                e_pts = e[order]
            ax.errorbar(x_sorted, y_pts, yerr=e_pts, fmt='o-' if len(x_sorted)>1 else 'o', capsize=4, label=s_lbl, color=color_cycle[i%len(color_cycle)] if color_cycle else None, markersize=4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Survival Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best', fontsize='small')
        ax.set_title(f"{xlabel} vs Survival ({os.path.basename(csv_file)})")
        outpath = os.path.join(out_dir, outname)
        fig.savefig(outpath, bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        return outpath

    # 1) 横轴 = mu_d
    # Try: if mud_from_labels (list) has values and its length matches number of series, use that.
    if len(mud_from_labels) == len(means):
        xvals = mud_from_labels
        yvals_list = [np.mean(m) if np.ndim(m)>0 else float(m) for m in means]
        # but better: if each mean is array over sigma_d, take mean across sigma_d to collapse to single value
        yvals_list = [float(np.nanmean(m)) for m in means]
        yerrs_list = [float(np.nanmean(s)) if s.size>0 else 0.0 for s in ses]
        series_labels = labels
        outname = os.path.splitext(os.path.basename(csv_file))[0] + "_vs_mud.png"
        p = draw_plot(xvals, [yvals_list], [yerrs_list], series_labels, "mu_d", outname)
        if p: outputs.append(p)
    elif mud_from_fname:
        # if one mud value in filename, we can't make a mu_d sweep from single file; skip or save info
        pass

    # 2) 横轴 = mu_e
    if len(mue_from_labels) == len(means):
        xvals = mue_from_labels
        yvals_list = [float(np.nanmean(m)) for m in means]
        yerrs_list = [float(np.nanmean(s)) if s.size>0 else 0.0 for s in ses]
        series_labels = labels
        outname = os.path.splitext(os.path.basename(csv_file))[0] + "_vs_mue.png"
        p = draw_plot(xvals, [yvals_list], [yerrs_list], series_labels, "mu_e", outname)
        if p: outputs.append(p)

    # 3) 横轴 = sigma_e (from filename)
    if sigmae_from_fname:
        # single sigmae value per file -> cannot sweep within this file
        # If you have multiple files with different sigmae and want a combined plot, set make_combined_plots=True
        pass

    # 4) 如果 labels 本身是数值并可以作为横轴（例如 labels = [0.1,0.2,0.3]）
    if labels_numeric and len(labels_numeric) == len(means):
        xvals = labels_numeric
        yvals_list = [float(np.nanmean(m)) for m in means]
        yerrs_list = [float(np.nanmean(s)) if s.size>0 else 0.0 for s in ses]
        series_labels = labels
        outname = os.path.splitext(os.path.basename(csv_file))[0] + "_vs_labelvals.png"
        p = draw_plot(xvals, [yvals_list], [yerrs_list], series_labels, "label_value", outname)
        if p: outputs.append(p)

    # Additionally: if the CSV contains survival curves across sigma_d (i.e., x axis = sigma_d),
    # keep original plot: sigma_d vs each series
    outname = os.path.splitext(os.path.basename(csv_file))[0] + "_vs_sigma_d.png"
    p = draw_plot(sigma_d, means, ses, labels, "sigma_d", outname)
    if p: outputs.append(p)

    return outputs, meta

def process_all(csv_path, recursive=True, out_subdir=out_subdir):
    files = find_csv_files(csv_path, recursive=recursive)
    if not files:
        print("No CSV files found under", csv_path)
        return []
    all_outputs = []
    metas = []
    for f in files:
        csv_dir = os.path.dirname(f) or "."
        out_dir = os.path.join(csv_dir, out_subdir)
        try:
            outs, meta = plot_single(f, out_dir)
            all_outputs.extend(outs)
            metas.append(meta)
            print(f"Processed {f}: produced {len(outs)} files")
        except Exception as e:
            print(f"Error processing {f}: {e}", file=sys.stderr)
    # OPTIONAL: make combined plots across files if requested (e.g. sigmae sweep)
    if make_combined_plots:
        try:
            make_combined(across_files=files, out_root=os.path.abspath(csv_path))
        except Exception as e:
            print("Combined plotting failed:", e, file=sys.stderr)
    return all_outputs, metas

# OPTIONAL: combine multiple files into plots along a parameter (e.g. sigmae)
def make_combined(across_files, out_root):
    """
    示例：读取所有文件名中的 sigmae 值，把每个文件的（sigma_d vs mean）曲线在一张图中叠加，
    横轴显示 sigma_d，图例显示 sigmae 值。
    """
    grouped = defaultdict(list)  # key: sigmae value, value: list of file paths
    for f in across_files:
        params = extract_params_from_filename(f)
        sigmae = None
        if params.get("sigmae"):
            try:
                sigmae = float(params["sigmae"])
            except Exception:
                sigmae = None
        if sigmae is None:
            continue
        grouped[sigmae].append(f)
    if not grouped:
        print("No sigmae values found in filenames for combined plotting.")
        return
    out_dir = os.path.join(out_root, "combined_plots")
    os.makedirs(out_dir, exist_ok=True)
    # for each group (sigmae), average curves if multiple files, then plot
    fig, ax = plt.subplots(figsize=figsize)
    color_cycle = plt.rcParams.get('axes.prop_cycle').by_key().get('color', None)
    for i,(sigmae, flist) in enumerate(sorted(grouped.items())):
        # take first file's curve (or average)
        header, data = read_csv(flist[0])
        sigma_d, labels, means, ses = parse_csv_data(header, data)
        # average across series into a single curve (mean of means)
        curve = np.nanmean(np.vstack([m for m in means]), axis=0)
        ax.plot(sigma_d, curve, '-o', label=f"sigmae={sigmae}", color=color_cycle[i%len(color_cycle)] if color_cycle else None)
    ax.set_xlabel("sigma_d")
    ax.set_ylabel("Survival Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='best', fontsize='small')
    outpath = os.path.join(out_dir, "combined_sigmae_vs_sigma_d.png")
    fig.savefig(outpath, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    print("Saved combined plot:", outpath)

def main():
    outs, metas = process_all(csv_path, recursive=recursive, out_subdir=out_subdir)
    print("Finished. Generated files:", len(outs))

if __name__ == '__main__':
    main()