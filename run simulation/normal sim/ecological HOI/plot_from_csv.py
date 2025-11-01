#!/usr/bin/env python3
# plot_from_csv.py
# 在本地读取 simd_no_plot.py 生成的 CSV 并绘图

import os
import csv
import matplotlib.pyplot as plt
import numpy as np

def read_csv(path):
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = list(reader)
    return header, data

def plot_csv(csv_path, out_png_path=None):
    header, data = read_csv(csv_path)
    # header = ['sigma_d', 'mean_mu_d_0.1', 'se_mu_d_0.1', 'mean_mu_d_0.2', 'se_mu_d_0.2', ...]
    sigma_d = np.array([float(row[0]) for row in data])
    # parse mu_d labels
    mu_d_labels = []
    for i in range(1, len(header), 2):
        label = header[i]
        if label.startswith("mean_mu_d_"):
            mu_d_labels.append(label.replace("mean_mu_d_", ""))
        else:
            mu_d_labels.append(f"col{i}")
    means = []
    ses = []
    for j in range(len(mu_d_labels)):
        mean_col = 1 + 2*j
        se_col = mean_col + 1
        means.append(np.array([float(row[mean_col]) for row in data]))
        ses.append(np.array([float(row[se_col]) for row in data]))

    fig, ax = plt.subplots(figsize=(10,6))
    for j, mu_label in enumerate(mu_d_labels):
        ax.errorbar(sigma_d, means[j], yerr=ses[j], fmt='o-', capsize=4, label=f"mu_d={mu_label}")
    ax.set_title(f"Survival Rate vs Sigma_d for {os.path.basename(csv_path)}")
    ax.set_xlabel("Sigma_d")
    ax.set_ylabel("Survival Rate")
    ax.grid(True)
    ax.legend()
    if out_png_path is None:
        out_png_path = csv_path.replace('.csv', '.png')
    fig.savefig(out_png_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return out_png_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: plot_from_csv.py <csv_path> [out_png_path]")
        sys.exit(1)
    csv_path = sys.argv[1]
    out_png = None if len(sys.argv) < 3 else sys.argv[2]
    out = plot_csv(csv_path, out_png)
    print("Saved plot to:", out)