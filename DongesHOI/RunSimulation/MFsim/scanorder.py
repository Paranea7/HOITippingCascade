import numpy as np
import matplotlib.pyplot as plt
from theoretical_solver import calculate_theoretical_properties

def scan_parameter(param_name, values, base_params, initial_guess):
    phis = []
    ms = []
    qs = []

    for val in values:
        params = base_params.copy()
        params[param_name] = float(val)

        m_th, q_th, phi_th, _ = calculate_theoretical_properties(
            params,
            initial_guess
        )
        ms.append(m_th)
        qs.append(q_th)
        phis.append(phi_th)

    return np.array(ms), np.array(qs), np.array(phis)

if __name__ == "__main__":
    base = {
        'mu_c': 0.0, 'sigma_c': 0.2,
        'mu_d': 0.3, 'sigma_d': 0.2,
        'mu_e': 0.1, 'sigma_e': 0.2
    }

    values = np.linspace(-1, 1, 50)

    ms, qs, phis = scan_parameter("mu_e", values, base, initial_guess=0.1)

    plt.plot(values, phis, '-o')
    plt.xlabel("mu_e")
    plt.ylabel("phi")
    plt.grid()
    plt.title("phi vs mu_e (mean-field theory)")
    plt.show()
