import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import time
import random
import os


# 1. Utility Functions
@njit
def _jit_sigma(x, theta):
    return 1.0 / (1.0 + np.exp(-x / theta))

def sigma(x, theta):
    return _jit_sigma(x, theta)

@njit
def compute_m_q_phi(x, theta):
    S = x.shape[0]
    s = _jit_sigma(x, theta)
    m = np.mean(s)
    q = np.mean(s * s)
    phi = np.mean(_jit_sigma(x, theta) * (1 - _jit_sigma(x, theta)) / theta)
    return m, q, phi

@njit
def compute_m_q_phi_time_series(x_series, theta):
    t_steps = x_series.shape[0]
    m_series = np.zeros(t_steps)
    q_series = np.zeros(t_steps)
    for t in range(t_steps):
        m, q, _ = compute_m_q_phi(x_series[t], theta)
        m_series[t] = m
        q_series[t] = q
    return m_series, q_series


# 2. JIT version of compute_rhs
@njit
def compute_rhs_jit(x, J, K, terms, theta):
    S = x.shape[0]
    h = np.zeros(S)
    sigma_vals = _jit_sigma(x, theta)

    for m in range(terms):
        sum_m = np.zeros(S)
        for k in range(K):
            idx = m * K + k
            if idx >= J.shape[2]:
                break
            for i in range(S):
                for j in range(S):
                    sum_m[i] += J[i, j, idx] * sigma_vals[j]
        h += sum_m

    return h


# 3. RK4 integrator (JIT-accelerated)
@njit
def runge_kutta_step(x, dt, J, K, terms, theta):
    k1 = compute_rhs_jit(x, J, K, terms, theta)
    k2 = compute_rhs_jit(x + 0.5 * dt * k1, J, K, terms, theta)
    k3 = compute_rhs_jit(x + 0.5 * dt * k2, J, K, terms, theta)
    k4 = compute_rhs_jit(x + dt * k3, J, K, terms, theta)
    return x + dt / 6.0 * (k1 + 2*k2 + 2*k3 + k4)


# 4. Single-batch simulation (still JIT)
@njit
def single_simulation_jit(S, t_steps, dt, J, K, terms, theta,
                          initial_x_mean, initial_x_std,
                          record_mq):
    x = np.random.normal(initial_x_mean, initial_x_std, S)
    if record_mq:
        m_hist = np.zeros(t_steps)
        q_hist = np.zeros(t_steps)
    else:
        m_hist = np.zeros(1)
        q_hist = np.zeros(1)

    for t in range(t_steps):
        x = runge_kutta_step(x, dt, J, K, terms, theta)
        if record_mq:
            m_val, q_val, _ = compute_m_q_phi(x, theta)
            m_hist[t] = m_val
            q_hist[t] = q_val

    m, q, phi = compute_m_q_phi(x, theta)
    return x, m, q, phi, m_hist, q_hist


# 5. Wrapper to call single simulation
def run_single_simulation(batch_id, params, initial_x_mean, initial_x_std):
    S = params["S"]
    t_steps = params["t_steps"]
    dt = params["dt"]
    J = params["J"]
    K = params["K"]
    terms = params["terms"]
    theta = params["theta"]
    record_mq = params["record_mq"]

    x_fin, m, q, phi, m_hist, q_hist = single_simulation_jit(
        S, t_steps, dt, J, K, terms, theta,
        initial_x_mean, initial_x_std,
        record_mq
    )
    return x_fin, m, q, phi, m_hist, q_hist


# 6. Serial version (“方案 A”)
def run_multiple_batches_serial(num_batches, params, initial_x_mean, initial_x_std):
    all_x = []
    all_m, all_q, all_phi = [], [], []
    last_m_hist = None
    last_q_hist = None

    for i in range(num_batches):
        result = run_single_simulation(i, params, initial_x_mean, initial_x_std)
        x_fin, m, q, phi, m_hist, q_hist = result
        all_x.extend(x_fin)
        all_m.append(m)
        all_q.append(q)
        all_phi.append(phi)
        last_m_hist = m_hist
        last_q_hist = q_hist

    return np.array(all_x), np.mean(all_m), np.mean(all_q), np.mean(all_phi), last_m_hist, last_q_hist


# 7. Sweep sigma_d
def sweep_sigma_d(sigma_d_values):
    global J
    sim_phi = []
    th_phi = []

    for sigma_d in sigma_d_values:
        print("Simulating σ_d =", sigma_d)

        J = np.random.normal(0, sigma_d, (S, S, comb_size)) / np.sqrt(S)

        params = {
            "S": S,
            "t_steps": t_steps,
            "dt": dt,
            "J": J,
            "K": K,
            "terms": terms,
            "theta": theta,
            "record_mq": record_mq
        }

        x_fin, m_avg, q_avg, phi_avg, m_hist, q_hist = \
            run_multiple_batches_serial(num_batches, params, initial_x_mean, initial_x_std)

        sim_phi.append(phi_avg)
        th_phi.append(q_avg - m_avg*m_avg)

    return sim_phi, th_phi


# 8. Global configuration
theta = 1.0
terms = 3
K = 3
comb_size = terms * K

t_steps = 3000
dt = 0.01
record_mq = True

S = 50
num_batches = 10

initial_x_mean = 5
initial_x_std  = 2

sigma_d_values = np.linspace(0.1, 1.0, 10)


# 9. Main
if __name__ == "__main__":
    start = time.time()
    sim_phi, th_phi = sweep_sigma_d(sigma_d_values)
    end = time.time()

    print("Total time required:", end-start, "seconds")

    plt.plot(sigma_d_values, sim_phi, label="Simulation φ")
    plt.plot(sigma_d_values, th_phi, label="Theory φ")
    plt.xlabel("σ_d")
    plt.ylabel("φ")
    plt.legend()
    plt.title("Comparison of Theory and Simulation (Serial JIT)")
    plt.show()