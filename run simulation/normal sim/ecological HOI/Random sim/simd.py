#!/usr/bin/env python3
# simd_no_plot.py
# 服务器上运行：只生成 CSV，不依赖 matplotlib

import os
import csv
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count

def generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e):
    c_i = np.random.normal(mu_c, sigma_c, s)
    d_ij = np.random.normal(mu_d / s, sigma_d / s, (s, s))
    d_ji = rho_d * d_ij + np.sqrt(max(0.0, 1 - rho_d**2)) * np.random.normal(mu_d / s, sigma_d / s, (s, s))
    e_ijk = np.random.normal(mu_e / s**2, sigma_e / s**2, (s, s, s))
    return c_i, d_ij, d_ji, e_ijk

def dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps):
    x = x_init.copy()
    dt = 0.01
    for _ in range(t_steps):
        def compute_dx(x_local):
            dx = -x_local**3 + x_local + c_i
            dx = dx + np.dot(d_ji, x_local)
            dx = dx + np.einsum('ijk,j,k->i', e_ijk, x_local, x_local)
            return dx
        k1 = compute_dx(x)
        k2 = compute_dx(x + 0.5 * dt * k1)
        k3 = compute_dx(x + 0.5 * dt * k2)
        k4 = compute_dx(x + dt * k3)
        x += (k1 + 2*k2 + 2*k3 + k4) * dt / 6.0
    return x

def calculate_survival_rate(final_states):
    survival = np.sum(final_states > 0)
    total = len(final_states)
    return float(survival) / float(total)

def single_simulation(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, t_steps):
    c_i, _, d_ji, e_ijk = generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e)
    x_init = np.full(s, -0.6, dtype=float)
    final_states = dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps)
    return calculate_survival_rate(final_states)

def save_results_csv(sigma_d_values, mu_d_values, survival_means, survival_ses, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{prefix}.csv")
    header = ['sigma_d']
    for mu_d in mu_d_values:
        header.append(f"mean_mu_d_{mu_d}")
        header.append(f"se_mu_d_{mu_d}")
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx, sigma_d in enumerate(sigma_d_values):
            row = [float(sigma_d)]
            for j in range(len(mu_d_values)):
                row.append(float(survival_means[j][idx]))
                row.append(float(survival_ses[j][idx]))
            writer.writerow(row)
    return out_path

def run_one_parameter_combo(s, mu_e, sigma_e,
                            mu_d_values, sigma_d_values,
                            mu_c, sigma_c, rho_d,
                            t_steps, simulations_per_sigma,
                            out_csv_dir,
                            n_workers=None):

    if n_workers is None:
        n_workers = max(1, cpu_count())

    survival_means = []
    survival_ses = []

    for mu_d in mu_d_values:
        means_for_mu_d = []
        ses_for_mu_d = []
        for sigma_d in sigma_d_values:
            args = [(s, mu_c, sigma_c, mu_d, float(sigma_d), rho_d, mu_e, sigma_e, t_steps)
                    for _ in range(simulations_per_sigma)]
            with Pool(processes=n_workers) as pool:
                results = pool.starmap(single_simulation, args)
            mean_surv = float(np.mean(results))
            se_surv = float(np.std(results, ddof=1) / np.sqrt(len(results))) if len(results) > 1 else 0.0
            means_for_mu_d.append(mean_surv)
            ses_for_mu_d.append(se_surv)
            del results
        survival_means.append(means_for_mu_d)
        survival_ses.append(ses_for_mu_d)

    prefix = f"s_{s}_mue_{mu_e}_sigmae_{sigma_e}"
    csv_path = save_results_csv(sigma_d_values, mu_d_values, survival_means, survival_ses, out_csv_dir, prefix)

    del survival_means, survival_ses

    return csv_path

def main():
    mu_c = 0.0
    sigma_c = 2.0 * np.sqrt(3.0) / 9.0
    rho_d = 0.0

    s_values = [10,30,50,100]
    mu_e_values = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    sigma_e_values = [0.0,0.1,0.2,0.3,0.5,0.7,1.0]

    mu_d_values = [0.1,0.2, 0.3, 0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    sigma_d_values = np.linspace(0.0, 1.0, 21)

    t_steps = 2000
    simulations_per_sigma = 500

    n_workers = max(1, cpu_count())

    out_csv_dir = "outputcsvdrandom0"
    os.makedirs(out_csv_dir, exist_ok=True)

    generated_files = []

    total = len(s_values) * len(mu_e_values) * len(sigma_e_values)
    cnt = 0
    for s in s_values:
        for mu_e in mu_e_values:
            for sigma_e in sigma_e_values:
                cnt += 1
                print(f"[{cnt}/{total}] Starting run for s={s}, mu_e={mu_e}, sigma_e={sigma_e}")
                csv_path = run_one_parameter_combo(
                    s, mu_e, sigma_e,
                    mu_d_values, sigma_d_values,
                    mu_c, sigma_c, rho_d,
                    t_steps, simulations_per_sigma,
                    out_csv_dir,
                    n_workers=n_workers
                )
                generated_files.append(csv_path)
                print(f"  Saved CSV: {csv_path}")

    print("All parameter combinations finished.")
    print("Generated files:")
    for c in generated_files:
        print("  ", c)

if __name__ == "__main__":
    main()