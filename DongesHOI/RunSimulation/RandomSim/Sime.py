#!/usr/bin/env python3
import os
import csv
import numpy as np
from multiprocessing import Pool, cpu_count

# ---------------- 参数生成 ----------------
def generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e):
    c_i = np.random.normal(mu_c, sigma_c, s)

    # two-body scaling
    mean_d = mu_d / s
    std_d = sigma_d / np.sqrt(s)
    d_ij = np.random.normal(mean_d, std_d, (s, s))
    d_ji = rho_d * d_ij + np.sqrt(1 - rho_d**2) * \
           np.random.normal(mean_d, std_d, (s, s))

    # three-body scaling
    mean_e = mu_e / s
    std_e = sigma_e / (s**2)
    e_ijk = np.random.normal(mean_e, std_e, (s, s, s))

    return c_i, d_ij, d_ji, e_ijk


# ---------------- 动力学模拟 ----------------
def dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps):
    x = x_init.copy()
    dt = 0.01
    XMAX = 50.0

    def safe(x):
        return np.clip(x, -XMAX, XMAX)

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


def single_simulation(s, mu_c, sigma_c, mu_d, sigma_d, rho_d,
                      mu_e, sigma_e, t_steps):

    c_i, _, d_ji, e_ijk = generate_parameters(
        s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e
    )

    x0 = np.full(s, -0.6)
    final = dynamics_simulation(s, c_i, d_ji, e_ijk, x0, t_steps)
    return calculate_survival_rate(final)


# ---------------- CSV 输出 ----------------
def save_results_csv(sigma_e_values, means, ses, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_csv_dir, f"{prefix}.csv")

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma_e", "mean_survival", "se"])

        for i, s_e in enumerate(sigma_e_values):
            w.writerow([float(s_e), float(means[i]), float(ses[i])])

    return path


# ---------------- 主扫描：只扫描 sigma_e ----------------
def run_sigma_e_scan(s, mu_e,
                     mu_c, sigma_c,
                     mu_d, sigma_d, rho_d,
                     sigma_e_values,
                     t_steps, sims,
                     out_csv_dir, n_workers):

    means = []
    ses = []

    for sigma_e in sigma_e_values:
        args = [
            (s, mu_c, sigma_c, mu_d, sigma_d, rho_d,
             mu_e, float(sigma_e), t_steps)
            for _ in range(sims)
        ]

        with Pool(n_workers) as pool:
            results = pool.starmap(single_simulation, args)

        mean_ = np.mean(results)
        se_ = np.std(results, ddof=1) / np.sqrt(len(results)) if len(results) > 1 else 0

        means.append(mean_)
        ses.append(se_)

    prefix = f"s_{s}_muE_{mu_e}"
    return save_results_csv(sigma_e_values, means, ses, out_csv_dir, prefix)


# ---------------- main ----------------
def main():
    mu_c = 0.0
    sigma_c = 2*np.sqrt(3)/27.0
    rho_d = 0.0

    s = 50           # 固定 s
    mu_e = 0.0       # 固定 mu_e
    mu_d = 0.1       # 固定 mu_d
    sigma_d = 0.5    # 固定 sigma_d

    sigma_e_values = [0.0, 0.1, 0.3, 0.5, 1.0]

    t_steps = 2400
    simulations_per_sigma = 50

    global out_csv_dir
    out_csv_dir = "csv_output_sigmaE_only"
    n_workers = max(1, cpu_count())

    print("Running σ_e scan...")

    csv_path = run_sigma_e_scan(
        s, mu_e,
        mu_c, sigma_c,
        mu_d, sigma_d, rho_d,
        sigma_e_values,
        t_steps, simulations_per_sigma,
        out_csv_dir,
        n_workers
    )

    print("Saved:", csv_path)


if __name__ == "__main__":
    main()