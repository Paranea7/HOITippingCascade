import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import math
@njit
def rk4_step(x, dt, c, d, e, m):
    h = c + d*m + e*m*m
    k1 = dt * (-x**3 + x + h)
    x2 = x + 0.5*k1
    k2 = dt * (-x2**3 + x2 + h)
    x3 = x + 0.5*k2
    k3 = dt * (-x3**3 + x3 + h)
    x4 = x + k3
    k4 = dt * (-x4**3 + x4 + h)
    return x + (k1 + 2*k2 + 2*k3 + k4) / 6

@njit
def simulate_x(m, initial_x, dt, c_arr, d_arr, e_arr):
    x = initial_x
    for t in range(len(c_arr)):
        x = rk4_step(x, dt, c_arr[t], d_arr[t], e_arr[t], m)
    return x

def compute_simulated_phi(mu_d, mu_e, N=100, T=3000, dt=0.01):
    mu_c = 0
    mu_m = 0
    sigma = 0.3
    pos = 0
    t_steps = int(T/dt)

    for _ in range(N):
        c = np.random.normal(mu_c, sigma)
        d = np.random.normal(mu_d, sigma)
        e = np.random.normal(mu_e, sigma)
        m = np.random.normal(mu_m, sigma)

        c_arr = np.full(t_steps, c)
        d_arr = np.full(t_steps, d)
        e_arr = np.full(t_steps, e)

        x_end = simulate_x(m, 0.1, dt, c_arr, d_arr, e_arr)
        if x_end > 0:
            pos += 1

    return pos / N

def phi_theory(mu_d, mu_e):
    mu_c = 0
    mu_m = 0
    sigma_c = sigma_d = sigma_e = sigma_m = 1

    mu_h = mu_c + mu_d * mu_m + mu_e * (mu_m**2 + sigma_m**2)

    var1 = sigma_c**2
    var2 = mu_m**2 * sigma_d**2 + sigma_m**2 * (mu_d**2 + sigma_d**2)
    var3 = sigma_e**2 * (mu_m**4 + 6*mu_m**2*sigma_m**2 + 3*sigma_m**4)

    sigma_h = np.sqrt(var1 + var2 + var3)
    return 0.5 * (1 + math.erf(mu_h / (np.sqrt(2) * sigma_h)))

def main():
    mu_d_list = np.linspace(-1, 1, 20)
    mu_e_list = np.linspace(-0.5, 0.5, 20)

    phi_sim = np.zeros((len(mu_d_list), len(mu_e_list)))
    phi_th = np.zeros((len(mu_d_list), len(mu_e_list)))

    for i, mu_d in enumerate(mu_d_list):
        for j, mu_e in enumerate(mu_e_list):
            phi_sim[i, j] = compute_simulated_phi(mu_d, mu_e)
            phi_th[i, j] = phi_theory(mu_d, mu_e)

    X, Y = np.meshgrid(mu_e_list, mu_d_list)

    plt.figure(figsize=(6,5))
    plt.contour(X, Y, phi_sim, levels=8, colors='blue')
    plt.contour(X, Y, phi_th, levels=8, colors='red', linestyles='dashed')
    plt.xlabel("mu_e")
    plt.ylabel("mu_d")
    plt.title("Blue = simulated phi; Red dashed = theoretical phi")
    plt.show()

if __name__ == "__main__":
    main()