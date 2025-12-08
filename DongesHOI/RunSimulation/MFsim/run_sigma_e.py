from main_code import sweep_sigma_e, plot_phi_vs_sigma_e, sim_params
from common_utils import save_csv, ensure_dir
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    ensure_dir("results_img")

    num_batches = 10
    initial_mean = -0.6
    initial_std = 0.0

    sigma_e_list = np.linspace(0, 0.9, 10)

    sim_phi_se, th_phi_se = sweep_sigma_e(sim_params, sigma_e_list,
                                          num_batches, initial_mean, initial_std)

    save_csv("result_sigma_e.csv",
             sigma_e=sigma_e_list,
             sim_phi=sim_phi_se,
             th_phi=th_phi_se)

    plot_phi_vs_sigma_e(sigma_e_list, sim_phi_se, th_phi_se)
    plt.savefig("results_img/phi_vs_sigma_e.png", dpi=300)