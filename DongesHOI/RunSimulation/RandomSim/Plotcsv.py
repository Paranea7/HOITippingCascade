#!/usr/bin/env python3
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def load_csv(path):
    mu_d_vals = []
    sigma_d_vals = []
    means = {}
    ses = {}

    with open(path) as f:
        r = csv.reader(f)
        header = next(r)

        mu_d_vals = sorted(list(set(float(h.split("_")[-1])
                                    for h in header if h.startswith("mean"))))

        for mu in mu_d_vals:
            means[mu] = []
            ses[mu] = []

        for row in r:
            sigma_d_vals.append(float(row[0]))
            idx = 1
            for mu in mu_d_vals:
                means[mu].append(float(row[idx])); idx += 1
                ses[mu].append(float(row[idx])); idx += 1

    return np.array(sigma_d_vals), mu_d_vals, means, ses


def plot_csv(csv_path, out_dir):
    sigma_d_vals, mu_d_vals, means, ses = load_csv(csv_path)

    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.basename(csv_path).replace(".csv", ".png")
    out_path = os.path.join(out_dir, fname)

    plt.figure(figsize=(10, 6))

    for mu in mu_d_vals:
        plt.errorbar(sigma_d_vals, means[mu], yerr=ses[mu],
                     fmt='o-', capsize=4, label=f"mu_d={mu}")

    plt.xlabel("sigma_d")
    plt.ylabel("Tipping rate")
    plt.title(fname)
    plt.grid(True)
    plt.legend()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print("saved plot:", out_path)


def main():
    # ------- 手动指定目录 -------
    csv_dir = "csv_output"   # 输入 CSV 目录
    out_dir = "plotsraw"        # 输出绘图目录
    # --------------------------

    files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]

    for f in files:
        plot_csv(os.path.join(csv_dir, f), out_dir)


if __name__ == "__main__":
    main()