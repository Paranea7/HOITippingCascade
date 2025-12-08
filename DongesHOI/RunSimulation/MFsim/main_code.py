import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from joblib import Parallel, delayed
import time
from scipy.stats import gaussian_kde

sim_params = {
        'S': 50,
        'dt': 0.01,
        't_steps': 3000,
        'mu_c': 0.0, 'sigma_c': 0.1283,
        'mu_d': 0.3, 'sigma_d': 0.2,
        'mu_e': 0.1, 'sigma_e': 0.2
    }


# ==========================================================
# 1. 动力学部分
# ==========================================================

@jit(nopython=True, cache=True)
def compute_rhs(x, c, d_matrix, e_tensor):
    nonlinear_term = -x**3 + x
    d_term = d_matrix @ x

    S = x.shape[0]
    e_term = np.zeros(S)

    if e_tensor.shape[0] > 0:
        for i in range(S):
            val = 0.0
            for j in range(S):
                for k in range(S):
                    val += e_tensor[i, j, k] * x[j] * x[k]
            e_term[i] = val

    return nonlinear_term + c + d_term + e_term


@jit(nopython=True, cache=True)
def rk4_step(x, dt, c, d_matrix, e_tensor):
    k1 = compute_rhs(x, c, d_matrix, e_tensor)
    k2 = compute_rhs(x + 0.5 * dt * k1, c, d_matrix, e_tensor)
    k3 = compute_rhs(x + 0.5 * dt * k2, c, d_matrix, e_tensor)
    k4 = compute_rhs(x + dt * k3, c, d_matrix, e_tensor)
    return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)


@jit(nopython=True, cache=True)
def simulate_dynamics_jit(x_init, dt, t_steps, c, d_matrix, e_tensor):
    x = x_init.copy()
    S = x.shape[0]
    m_hist = np.zeros(t_steps)
    q_hist = np.zeros(t_steps)

    for t in range(t_steps):
        x = rk4_step(x, dt, c, d_matrix, e_tensor)
        m_hist[t] = np.mean(x)
        q_hist[t] = np.mean(x**2)

    return x, m_hist, q_hist


# ==========================================================
# 2. 单次/多次并行模拟
# ==========================================================

def run_single_simulation(seed, params, initial_x_mean, initial_x_std):
    np.random.seed(seed)
    S = params['S']

    c = np.random.normal(params['mu_c'], params['sigma_c'], size=S)
    d_matrix = np.random.normal(params['mu_d']/S, params['sigma_d']/np.sqrt(S), size=(S,S))
    np.fill_diagonal(d_matrix, 0)

    if params['sigma_e']==0 and params['mu_e']==0:
        e_tensor = np.zeros((0,0,0))
    else:
        e_tensor = np.random.normal(params['mu_e']/S**2, params['sigma_e']/S, size=(S,S,S))

    x_init = np.random.normal(initial_x_mean, initial_x_std, size=S)

    final_x, m_hist, q_hist = simulate_dynamics_jit(
        x_init, params['dt'], params['t_steps'], c, d_matrix, e_tensor
    )

    final_m = np.mean(final_x)
    final_q = np.mean(final_x**2)

    # 统一定义 φ = 最终状态落在 x>0 的比例
    phi = np.mean(final_x > 0)

    return final_x, final_m, final_q, phi, m_hist, q_hist


def run_multiple_batches_parallel(num_batches, params, initial_x_mean, initial_x_std):
    print("\n--- Running {} simulations parallel ---".format(num_batches))

    results = Parallel(n_jobs=-1)(
        delayed(run_single_simulation)(i, params, initial_x_mean, initial_x_std)
        for i in range(num_batches)
    )

    all_x = []
    all_m, all_q, all_phi = [], [], []

    last_m_hist = None
    last_q_hist = None

    for (x_fin, m, q, phi, m_hist, q_hist) in results:
        all_x.extend(x_fin)
        all_m.append(m)
        all_q.append(q)
        all_phi.append(phi)
        last_m_hist = m_hist
        last_q_hist = q_hist

    return np.array(all_x), np.mean(all_m), np.mean(all_q), np.mean(all_phi), last_m_hist, last_q_hist


# ==========================================================
# 3. 理论
# ==========================================================

def solve_for_x_stable(h, initial_guess):
    coeffs = [1, 0, -1, -h]
    roots = np.roots(coeffs)
    real_roots = roots[np.abs(roots.imag)<1e-6].real
    if len(real_roots)==0:
        return np.nan
    return np.max(real_roots) if initial_guess>0 else np.min(real_roots)


def calculate_theoretical_properties(params, initial_x_mean_guess,
                                     num_h_samples=50000, max_iter=2000, tol=1e-5,
                                     damping=0.2):

    mu_c = params['mu_c']
    sigma_c = params['sigma_c']
    mu_d = params['mu_d']
    sigma_d = params['sigma_d']
    mu_e = params['mu_e']
    sigma_e = params['sigma_e']

    m_th = initial_x_mean_guess
    q_th = initial_x_mean_guess**2

    z_samples = np.random.normal(0,1,num_h_samples)

    for _ in range(max_iter):
        mu_h = mu_c + mu_d*m_th + mu_e*(m_th**2)
        sigma_h2 = sigma_c**2 + sigma_d**2*q_th + sigma_e**2*(q_th**2)
        sigma_h = np.sqrt(max(0,sigma_h2))

        h_samples = mu_h + sigma_h*z_samples

        x_samples = np.array([solve_for_x_stable(h, initial_x_mean_guess)
                              for h in h_samples])
        x_samples = x_samples[~np.isnan(x_samples)]
        if len(x_samples)==0:
            return np.nan, np.nan, np.nan, np.array([])

        m_new = np.mean(x_samples)
        q_new = np.mean(x_samples**2)

        if abs(m_new-m_th)+abs(q_new-q_th) < tol:
            break

        m_th = (1-damping)*m_th + damping*m_new
        q_th = (1-damping)*q_th + damping*q_new

    # 理论 φ 也改为 proportion(x>0)
    phi_th = np.mean(x_samples > 0)

    return m_th, q_th, phi_th, x_samples


# ==========================================================
# 4. 辅助统一函数（减少重复）
# ==========================================================

def get_sim_theory_phi(params, num_batches, initial_mean, initial_std):
    _, _, _, sim_phi, _, _ = run_multiple_batches_parallel(
        num_batches, params, initial_mean, initial_std
    )
    _, _, th_phi, _ = calculate_theoretical_properties(params, initial_mean)
    return sim_phi, th_phi


# ==========================================================
# 5. 扫描函数
# ==========================================================

def sweep_sigma_d(params, sigma_d_list, num_batches, initial_mean, initial_std):
    sim_phi, th_phi = [], []
    for sd in sigma_d_list:
        newp = params.copy()
        newp['sigma_d'] = sd
        sim_p, th_p = get_sim_theory_phi(newp, num_batches, initial_mean, initial_std)
        sim_phi.append(sim_p)
        th_phi.append(th_p)
    return np.array(sim_phi), np.array(th_phi)


def sweep_sigma_e(params, sigma_e_list, num_batches, initial_mean, initial_std):
    sim_phi, th_phi = [], []
    for se in sigma_e_list:
        newp = params.copy()
        newp['sigma_e'] = se
        sim_p, th_p = get_sim_theory_phi(newp, num_batches, initial_mean, initial_std)
        sim_phi.append(sim_p)
        th_phi.append(th_p)
    return np.array(sim_phi), np.array(th_phi)


def sweep_phase_sigma_d_sigma_e(params, sigma_d_list, sigma_e_list, num_batches, initial_mean, initial_std):
    sim_grid = np.zeros((len(sigma_d_list), len(sigma_e_list)))
    th_grid = np.zeros((len(sigma_d_list), len(sigma_e_list)))

    for i, sd in enumerate(sigma_d_list):
        for j, se in enumerate(sigma_e_list):
            newp = params.copy()
            newp['sigma_d'] = sd
            newp['sigma_e'] = se
            sim_p, th_p = get_sim_theory_phi(newp, num_batches, initial_mean, initial_std)
            sim_grid[i,j] = sim_p
            th_grid[i,j] = th_p

    return sim_grid, th_grid


def sweep_phase_mu_d_mu_e(params, mu_d_list, mu_e_list, num_batches, initial_mean, initial_std):
    sim_grid = np.zeros((len(mu_d_list), len(mu_e_list)))
    th_grid = np.zeros((len(mu_d_list), len(mu_e_list)))

    for i, md in enumerate(mu_d_list):
        for j, me in enumerate(mu_e_list):
            newp = params.copy()
            newp['mu_d'] = md
            newp['mu_e'] = me
            sim_p, th_p = get_sim_theory_phi(newp, num_batches, initial_mean, initial_std)
            sim_grid[i,j] = sim_p
            th_grid[i,j] = th_p

    return sim_grid, th_grid


# ==========================================================
# 6. 绘图
# ==========================================================

def plot_phi_vs_sigma_d(sigma_d_list, sim_phi, th_phi):
    plt.figure()
    plt.plot(sigma_d_list, sim_phi, 'o-', label='Simulation')
    plt.plot(sigma_d_list, th_phi, '--', label='Theory')
    plt.xlabel("sigma_d")
    plt.ylabel("phi")
    plt.grid()
    plt.legend()
    plt.title("phi vs sigma_d")
    plt.show()


def plot_phi_vs_sigma_e(sigma_e_list, sim_phi, th_phi):
    plt.figure()
    plt.plot(sigma_e_list, sim_phi, 'o-', label='Simulation')
    plt.plot(sigma_e_list, th_phi, '--', label='Theory')
    plt.xlabel("sigma_e")
    plt.ylabel("phi")
    plt.grid()
    plt.legend()
    plt.title("phi vs sigma_e")
    plt.show()


def plot_phase_sigma(sigma_d_list, sigma_e_list, sim_grid, th_grid):
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(sim_grid, origin='lower',
               extent=[sigma_e_list[0], sigma_e_list[-1], sigma_d_list[0], sigma_d_list[-1]],
               aspect='auto')
    plt.colorbar()
    plt.title("Sim phi(sigma_d, sigma_e)")

    plt.subplot(1,2,2)
    plt.imshow(th_grid, origin='lower',
               extent=[sigma_e_list[0], sigma_e_list[-1], sigma_d_list[0], sigma_d_list[-1]],
               aspect='auto')
    plt.colorbar()
    plt.title("Theory phi(sigma_d, sigma_e)")

    plt.show()


def plot_phase_mu(mu_d_list, mu_e_list, sim_grid, th_grid):
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(sim_grid, origin='lower',
               extent=[mu_e_list[0], mu_e_list[-1], mu_d_list[0], mu_d_list[-1]],
               aspect='auto')
    plt.colorbar()
    plt.title("Sim phi(mu_d, mu_e)")

    plt.subplot(1,2,2)
    plt.imshow(th_grid, origin='lower',
               extent=[mu_e_list[0], mu_e_list[-1], mu_d_list[0], mu_d_list[-1]],
               aspect='auto')
    plt.colorbar()
    plt.title("Theory phi(mu_d, mu_e)")

    plt.show()


# ==========================================================
# 7. 主程序
# ==========================================================

if __name__ == "__main__":

    num_batches = 10
    initial_mean = -0.6
    initial_std = 0.0

    sigma_d_list = np.linspace(0.0, 0.9, 10)
    sim_phi_sd, th_phi_sd = sweep_sigma_d(sim_params, sigma_d_list, num_batches, initial_mean, initial_std)
    plot_phi_vs_sigma_d(sigma_d_list, sim_phi_sd, th_phi_sd)

    sigma_e_list = np.linspace(0.0, 0.9, 10)
    sim_phi_se, th_phi_se = sweep_sigma_e(sim_params, sigma_e_list, num_batches, initial_mean, initial_std)
    plot_phi_vs_sigma_e(sigma_e_list, sim_phi_se, th_phi_se)

    sim_grid_sdse, th_grid_sdse = sweep_phase_sigma_d_sigma_e(
        sim_params, sigma_d_list, sigma_e_list, num_batches, initial_mean, initial_std
    )
    plot_phase_sigma(sigma_d_list, sigma_e_list, sim_grid_sdse, th_grid_sdse)

    mu_d_list = np.linspace(-0.5, 0.5, 11)
    mu_e_list = np.linspace(-0.5, 0.5, 11)
    sim_grid_mdme, th_grid_mdme = sweep_phase_mu_d_mu_e(
        sim_params, mu_d_list, mu_e_list, num_batches, initial_mean, initial_std
    )
    plot_phase_mu(mu_d_list, mu_e_list, sim_grid_mdme, th_grid_mdme)