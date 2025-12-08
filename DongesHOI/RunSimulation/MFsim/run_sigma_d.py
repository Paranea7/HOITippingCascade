from main_code import sweep_sigma_d, plot_phi_vs_sigma_d, sim_params
from common_utils import save_csv, ensure_dir
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    ensure_dir("results_img")

    num_batches = 10
    initial_mean = -0.6
    initial_std = 0.0

    sigma_d_list = np.linspace(0, 0.9, 10)

    sim_phi_sd, th_phi_sd = sweep_sigma_d(sim_params, sigma_d_list,
                                          num_batches, initial_mean, initial_std)

    save_csv("result_sigma_d.csv",
             sigma_d=sigma_d_list,
             sim_phi=sim_phi_sd,
             th_phi=th_phi_sd)

    plot_phi_vs_sigma_d(sigma_d_list, sim_phi_sd, th_phi_sd)
    plt.savefig("results_img/phi_vs_sigma_d.png", dpi=300)