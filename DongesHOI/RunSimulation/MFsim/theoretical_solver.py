import numpy as np

def solve_for_x_stable(h, initial_guess):
    coeffs = [1, 0, -1, -h]
    roots = np.roots(coeffs)
    real_roots = roots[np.abs(roots.imag) < 1e-6].real

    if len(real_roots) == 0:
        return np.nan

    if initial_guess > 0:
        return np.max(real_roots)
    else:
        return np.min(real_roots)

def calculate_theoretical_properties(params, initial_x_mean_guess,
                                     num_h_samples=40000, max_iter=2000,
                                     tol=1e-5, damping=0.2):

    mu_c = params['mu_c']
    sigma_c = params['sigma_c']
    mu_d = params['mu_d']
    sigma_d = params['sigma_d']
    mu_e = params['mu_e']
    sigma_e = params['sigma_e']

    m_th = initial_x_mean_guess
    q_th = initial_x_mean_guess**2

    z_samples = np.random.normal(0, 1, num_h_samples)

    for _ in range(max_iter):
        mu_h = mu_c + mu_d*m_th + mu_e*(m_th**2)
        var_h = sigma_c**2 + sigma_d**2*q_th + sigma_e**2*(q_th**2)
        sigma_h = np.sqrt(max(0, var_h))

        h_samples = mu_h + sigma_h*z_samples

        x_samples = np.array([solve_for_x_stable(h, initial_x_mean_guess) for h in h_samples])
        x_samples = x_samples[np.isfinite(x_samples)]

        if len(x_samples) == 0:
            return np.nan, np.nan, np.nan, np.array([])

        m_new = np.mean(x_samples)
        q_new = np.mean(x_samples**2)

        if abs(m_new - m_th) + abs(q_new - q_th) < tol:
            break

        m_th = (1 - damping)*m_th + damping*m_new
        q_th = (1 - damping)*q_th + damping*q_new

    if initial_x_mean_guess > 0:
        phi_th = np.mean(x_samples > 0)
    else:
        phi_th = np.mean(x_samples < 0)

    return m_th, q_th, phi_th, x_samples
