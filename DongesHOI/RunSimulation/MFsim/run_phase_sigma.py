from main_code import sweep_phase_sigma_d_sigma_e, plot_phase_sigma, sim_params
from common_utils import save_csv, ensure_dir
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    ensure_dir("results_img")

    num_batches = 10
    initial_mean = -0.6
    initial_std = 0.0

    sigma_d_list = np.linspace(0, 0.9, 10)
    sigma_e_list = np.linspace(0, 0.9, 10)

    sim_grid, th_grid = sweep_phase_sigma_d_sigma_e(
        sim_params, sigma_d_list, sigma_e_list,
        num_batches, initial_mean, initial_std)

    save_csv("result_phase_sigma.csv",
             sigma_d=np.repeat(sigma_d_list, len(sigma_e_list)),
             sigma_e=np.tile(sigma_e_list, len(sigma_d_list)),
             sim_phi=sim_grid.flatten(),
             th_phi=th_grid.flatten())

    plot_phase_sigma(sigma_d_list, sigma_e_list, sim_grid, th_grid)
    plt.savefig("results_img/phase_sigma.png", dpi=300)