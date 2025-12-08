from main_code import sweep_phase_mu_d_mu_e, plot_phase_mu, sim_params
from common_utils import save_csv, ensure_dir
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    ensure_dir("results_img")

    num_batches = 10
    initial_mean = -0.6
    initial_std = 0.0

    mu_d_list = np.linspace(-0.5, 0.5, 11)
    mu_e_list = np.linspace(-0.5, 0.5, 11)

    sim_grid, th_grid = sweep_phase_mu_d_mu_e(
        sim_params, mu_d_list, mu_e_list,
        num_batches, initial_mean, initial_std)

    save_csv("result_phase_mu.csv",
             mu_d=np.repeat(mu_d_list, len(mu_e_list)),
             mu_e=np.tile(mu_e_list, len(mu_d_list)),
             sim_phi=sim_grid.flatten(),
             th_phi=th_grid.flatten())

    plot_phase_mu(mu_d_list, mu_e_list, sim_grid, th_grid)
    plt.savefig("results_img/phase_mu.png", dpi=300)