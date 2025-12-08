import csv
import numpy as np
import matplotlib.pyplot as plt

def load_csv(csv_file):
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)

    mu_d = np.array([float(x) for x in rows[0][1:]])
    mu_e = []
    grid = []

    for r in rows[1:]:
        mu_e.append(float(r[0]))
        grid.append([float(x) for x in r[1:]])

    return mu_d, np.array(mu_e), np.array(grid)


def plot_heatmap_from_csv(csv_file, out_png):
    mu_d, mu_e, grid = load_csv(csv_file)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        grid,
        origin='lower',
        aspect='auto',
        extent=[mu_d[0], mu_d[-1], mu_e[0], mu_e[-1]],
        cmap='viridis',
        vmin=0,
        vmax=1,
    )

    ax.set_xlabel("mu_d")
    ax.set_ylabel("mu_e")
    ax.set_title("Survival rate")
    plt.colorbar(im, ax=ax, label="Survival rate")

    # ---------------- 辅助虚线接口（你自己填坐标） ----------------
    # 示例：在 mu_d = 0 画一条竖虚线
    # ax.axvline(x=0.0, color='white', linestyle='--', linewidth=1)

    # 示例：在 mu_e = 0 画一条横虚线
    # ax.axhline(y=0.0, color='white', linestyle='--', linewidth=1)
    # ------------------------------------------------------------

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# 简单测试方式：
# plot_heatmap_from_csv("your.csv", "heatmap.png")