import numpy as np
import matplotlib.pyplot as plt

def generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e):
    c_i = np.random.normal(mu_c, sigma_c, s)
    d_ij = np.random.normal(mu_d / s, sigma_d / s, (s, s))
    d_ji = rho_d * d_ij + np.sqrt(1 - rho_d ** 2) * np.random.normal(mu_d / s, sigma_d / s, s)
    e_ijk = np.random.normal(mu_e / s ** 2, sigma_e / s ** 2, (s, s, s))
    return c_i, d_ij, d_ji, e_ijk


def compute_dynamics(x, c_i, d_ji, e_ijk):
    dx = -x ** 3 + x + c_i
    dx += np.dot(d_ji, x)

    x_matrix = x[:, None, None] * x[None, :, None] * x[None, None, :]
    e_contribution = np.einsum('ijk->i', e_ijk * x_matrix)
    dx += e_contribution

    return dx


def rk4_step(x, c_i, d_ji, e_ijk, dt):
    """ 实现龙格-库塔四阶方法 """
    k1 = compute_dynamics(x, c_i, d_ji, e_ijk)
    k2 = compute_dynamics(x + 0.5 * dt * k1, c_i, d_ji, e_ijk)
    k3 = compute_dynamics(x + 0.5 * dt * k2, c_i, d_ji, e_ijk)
    k4 = compute_dynamics(x + dt * k3, c_i, d_ji, e_ijk)

    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps):
    x = x_init.copy()
    dt = 0.01
    survival_counts = []

    for _ in range(t_steps):
        x = rk4_step(x, c_i, d_ji, e_ijk, dt)
        survival_counts.append(np.sum(x > 0))

    return x, survival_counts


def plot_final_state_distribution(final_states):
    plt.figure(figsize=(10, 6))
    plt.hist(final_states, bins=100, density=True, alpha=0.6, color='b', edgecolor='black')
    plt.title('Final State Distribution')
    plt.xlabel('System State (x)')
    plt.ylabel('Density')
    plt.xlim(-1.6, 1.6)
    plt.grid()
    plt.show()


def main():
    s = 500
    mu_c = 0.0
    sigma_c = 0.5
    mu_d = 0.0
    sigma_d = 0.5
    rho_d = 1.0
    mu_e = 0.0
    sigma_e = 0.0

    c_i, d_ij, d_ji, e_ijk = generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e)
    x_init = np.full(s, -0.8)
    t_steps = 500

    final_states, survival_counts = dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps)

    plot_final_state_distribution(final_states)


if __name__ == "__main__":
    main()