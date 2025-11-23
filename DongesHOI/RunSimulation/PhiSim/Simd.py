#!/usr/bin/env python3
import os
import csv
import numpy as np
from multiprocessing import Pool, cpu_count

# ---------------- 固定比例 c_i 设置 ----------------
phi0 = 0.05    # 有 5% 的节点使用 c = 0.4
c_high = 0.4
c_low = 0.0

# ---------------- 参数生成（删除 mu_c, sigma_c） ----------------
def generate_parameters(s, mu_d, sigma_d, rho_d, mu_e, sigma_e):

    # ----------- 固定比例生成 c_i -----------
    c_i = np.full(s, c_low)
    n_high = int(phi0 * s)
    idx = np.random.choice(s, n_high, replace=False)
    c_i[idx] = c_high

    # two-body scaling
    mean_d = mu_d / s
    std_d = sigma_d / np.sqrt(s)
    d_ij = np.random.normal(mean_d, std_d, (s, s))
    d_ji = rho_d * d_ij + np.sqrt(1 - rho_d**2) * \
           np.random.normal(mean_d, std_d, (s, s))

    # three-body scaling
    mean_e = mu_e / s
    std_e = sigma_e / (s ** 1.5)
    e_ijk = np.random.normal(mean_e, std_e, (s, s, s))

    return c_i, d_ij, d_ji, e_ijk


# ---------------- 动力学模拟 ----------------
def dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps):
    x = x_init.copy()
    dt = 0.01
    XMAX = 50.0

    def safe(xx):
        return np.clip(xx, -XMAX, XMAX)

    def f(xl):
        xl = safe(xl)
        dx = -xl**3 + xl + c_i
        dx = dx + d_ji @ xl
        dx = dx + np.einsum('ijk,j,k->i', e_ijk, xl, xl)
        return safe(dx)

    for _ in range(t_steps):
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x += (k1 + 2*k2 + 2*k3 + k4) * dt / 6.
        x = safe(x)

    return x


def calculate_survival_rate(final_states):
    return np.mean(final_states > 0)


def single_simulation(s, mu_d, sigma_d, rho_d, mu_e, sigma_e, t_steps):
    c_i, _, d_ji, e_ijk = generate_parameters(
        s, mu_d, sigma_d, rho_d, mu_e, sigma_e
    )
    x0 = np.full(s, -0.6)
    final = dynamics_simulation(s, c_i, d_ji, e_ijk, x0, t_steps)
    return calculate_survival_rate(final)


# ---------------- CSV 输出 ----------------
def save_results_csv(sigma_d_values, mu_d_values, survival_means, survival_ses, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{prefix}.csv")

    header = ["sigma_d"]
    for mu_d in mu_d_values:
        header += [f"mean_mu_d_{mu_d}", f"se_mu_d_{mu_d}"]

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        for i, s_d in enumerate(sigma_d_values):
            row = [float(s_d)]
            for j in range(len(mu_d_values)):
                row.append(float(survival_means[j][i]))
                row.append(float(survival_ses[j][i]))
            w.writerow(row)
    return path


# ---------------- 主计算逻辑 ----------------
def run_one_combo(s, mu_e, sigma_e,
                  mu_d_values, sigma_d_values,
                  rho_d,
                  t_steps, sims,
                  out_csv_dir,
                  n_workers):

    survival_means = []
    survival_ses = []

    for mu_d in mu_d_values:
        means = []
        ses = []

        for sigma_d in sigma_d_values:

            args = [(s, mu_d, float(sigma_d),
                     rho_d, mu_e, sigma_e, t_steps)
                    for _ in range(sims)]

            with Pool(n_workers) as pool:
                results = pool.starmap(single_simulation, args)

            mean_ = np.mean(results)
            se_ = np.std(results, ddof=1) / np.sqrt(len(results)) if len(results) > 1 else 0

            means.append(mean_)
            ses.append(se_)

        survival_means.append(means)
        survival_ses.append(ses)

    prefix = f"s_{s}_mue_{mu_e}_sigmae_{sigma_e}_phi0_{phi0}"
    csv_path = save_results_csv(sigma_d_values, mu_d_values,
                                survival_means, survival_ses,
                                out_csv_dir, prefix)
    return csv_path


# ---------------- main ----------------
def main():
    rho_d = 0.0

    s_values = [30, 50]
    mu_e_values = [0.5, 1.0]
    sigma_e_values = [0.0, 0.1, 0.3, 0.5, 1.0]

    mu_d_values = [0.0, 0.1, 0.3, 0.5, 1.0]
    sigma_d_values = np.linspace(0, 1, 21)

    t_steps = 2000
    simulations_per_sigma = 50

    out_csv_dir = "csv_output_fixed_c"
    n_workers = max(1, cpu_count())

    total = len(s_values) * len(mu_e_values) * len(sigma_e_values)
    k = 0

    for s in s_values:
        for mu_e in mu_e_values:
            for sigma_e in sigma_e_values:
                k += 1
                print(f"[{k}/{total}] running s={s}, mu_e={mu_e}, sigma_e={sigma_e}")

                csv_path = run_one_combo(
                    s, mu_e, sigma_e,
                    mu_d_values, sigma_d_values,
                    rho_d,
                    t_steps, simulations_per_sigma,
                    out_csv_dir,
                    n_workers
                )

                print(" saved:", csv_path)


if __name__ == "__main__":
    main()