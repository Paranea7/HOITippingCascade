import numpy as np
import matplotlib.pyplot as plt
from theoretical_solver import calculate_theoretical_properties

def compute_phase_diagram(param1, param2, grid1, grid2, base_params, initial_guess):
    Z = np.zeros((len(grid1), len(grid2)))

    for i, v1 in enumerate(grid1):
        for j, v2 in enumerate(grid2):
            params = base_params.copy()
            params[param1] = float(v1)
            params[param2] = float(v2)

            m_th, q_th, phi_th, _ = calculate_theoretical_properties(params, initial_guess)
            Z[i, j] = phi_th

    return Z

if __name__ == "__main__":
    base = {
        'mu_c': 0.0, 'sigma_c': 0.2,
        'mu_d': 0.0, 'sigma_d': 0.2,
        'mu_e': 0.0, 'sigma_e': 0.2
    }

    grid1 = np.linspace(-1, 1, 40)
    grid2 = np.linspace(-1, 1, 40)

    Z = compute_phase_diagram("mu_d", "mu_e", grid1, grid2, base, initial_guess=0.1)

    plt.contour(grid1, grid2, Z.T, levels=[0], colors='black')
    plt.imshow(Z.T, extent=[grid1.min(),grid1.max(),grid2.min(),grid2.max()],
               origin='lower', aspect='auto', cmap='coolwarm')
    plt.colorbar(label="phi")
    plt.xlabel("mu_d")
    plt.ylabel("mu_e")
    plt.title("Phase boundary phi=0 in mu_d - mu_e plane")
    plt.show()
