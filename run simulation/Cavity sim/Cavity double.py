#!/usr/bin/env python3
# coding: utf-8
"""
修改版程序：在自洽求解后显式寻找有效单体势的所有稳定点（局部最小处），
并在找到两个稳定点时从理论玻尔兹曼密度中为每个 basin 估计峰的均值、方差与权重，
将这两个高斯分量与数值拟合结果在同一张图中比较。

保存为 bistable_compare_with_basins.py 并运行。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
import time

from scipy import optimize

matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# ---------------- 数值仿真模块 ----------------

def generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    c_i = rng.normal(mu_c, sigma_c, s)
    d_ij = np.zeros((s, s))
    for i in range(s):
        for j in range(i + 1, s):
            z1, z2 = rng.normal(0, 1, 2)
            d_ij[i, j] = mu_d / s + sigma_d / np.sqrt(s) * z1
            d_ij[j, i] = mu_d / s + sigma_d / np.sqrt(s) * (rho_d * z1 + np.sqrt(max(0.0, 1 - rho_d ** 2)) * z2)
    np.fill_diagonal(d_ij, 0.0)
    e_ijk = None
    return c_i, d_ij, d_ij.T, e_ijk

def dynamics_equation(x, t, c_i, d_ji, e_ijk_func, S):
    dxdt = -x ** 3 + x + c_i
    dxdt += d_ji.dot(x)
    mean_x = np.mean(x)
    dxdt += 0.5 * mean_x ** 2
    return dxdt

def dynamics_simulation(S, c_i, d_ji, e_ijk, x_init, t_steps, dt=0.01, record_every=10):
    x_current = x_init.copy()
    survival_counts = []
    for step in range(t_steps):
        k1 = dt * dynamics_equation(x_current, 0, c_i, d_ji, None, S)
        k2 = dt * dynamics_equation(x_current + 0.5 * k1, 0, c_i, d_ji, None, S)
        k3 = dt * dynamics_equation(x_current + 0.5 * k2, 0, c_i, d_ji, None, S)
        k4 = dt * dynamics_equation(x_current + k3, 0, c_i, d_ji, None, S)
        x_current += (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        if step % record_every == 0:
            survival_counts.append(np.sum(np.abs(x_current) > 0.5))
    return x_current, survival_counts

def plot_final_state_distribution(x_final, bins=50):
    plt.figure(figsize=(10, 6))
    plt.hist(x_final, bins=bins, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='势垒')
    plt.xlabel('最终状态 x')
    plt.ylabel('概率密度')
    plt.title('数值模拟最终状态分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def estimate_phi_from_final_states(x_final, threshold=0.5):
    return np.sum(x_final > threshold) / len(x_final)

# ---------------- 理论自洽求解器 ----------------

class BistableCavitySolver:
    def __init__(self, params):
        """
        params 包含 μ_c, μ_d, μ_e, σ_c, σ_d, σ_e, ρ_d, S, max_iter, tol, relaxation, denom_eps
        """
        self.params = params
        self.convergence_history = []
        # 以下属性在 solve_iterative 之后可能被设置：
        self.μ1 = None; self.μ2 = None
        self.σ1_sq = None; self.σ2_sq = None
        self.phi_comp = None  # weight for μ2 (right peak)
        self.basin_edges = None

    def single_particle_potential(self, x, mu_eff):
        return x ** 4 / 4 - x ** 2 / 2 - mu_eff * x

    def effective_drift(self, x, mu_eff):
        return -x ** 3 + x + mu_eff

    def boltzmann_distribution(self, x, mu_eff, sigma_eff):
        if sigma_eff < 1e-12:
            # 退化情况
            pdf = np.zeros_like(x)
            st = self.find_stable_points(mu_eff)
            if len(st) > 0:
                idx = np.argmin(np.abs(x - st[0]))
                pdf[idx] = 1.0
            else:
                pdf[:] = 1.0 / x.size
            return pdf
        V_eff = self.single_particle_potential(x, mu_eff)
        D_eff = sigma_eff ** 2 / 2.0
        pdf = np.exp(-V_eff / D_eff)
        dx = x[1] - x[0]
        pdf = pdf / (np.sum(pdf) * dx)
        return pdf

    def find_stable_points(self, mu_eff, x_min=-3.0, x_max=3.0, n_points=2000):
        """
        返回所有稳定点（即 drift=0 且 second derivative > 0）
        """
        x_grid = np.linspace(x_min, x_max, n_points)
        drift = self.effective_drift(x_grid, mu_eff)
        zeros = []
        for i in range(1, n_points):
            if drift[i - 1] == 0:
                zeros.append(x_grid[i - 1])
            elif drift[i - 1] * drift[i] < 0:
                try:
                    root = optimize.brentq(lambda xx: self.effective_drift(xx, mu_eff), x_grid[i - 1], x_grid[i])
                    zeros.append(root)
                except Exception:
                    continue
        # 去重（合并近似相同的根）
        zeros = np.array(sorted(set([round(float(z), 8) for z in zeros])))
        stable_pts = []
        for z in zeros:
            # second derivative of potential V'' = 3 x^2 - 1
            if 3 * z * z - 1 > 0:  # V'' > 0 => 稳定
                stable_pts.append(float(z))
        return sorted(stable_pts)

    def find_all_stable_points_and_basins(self, mu_eff, sigma_eff, x_min=-3.0, x_max=3.0):
        """
        找稳定点并用玻尔兹曼密度在每个 basin 上计算权重和局部方差（以估计高斯分量）。
        返回 (peaks, weights, variances, basin_edges)
        - peaks: list of μ_i (升序)
        - weights: list sum to 1
        - variances: estimated local variance around each peak (σ_i^2)
        - basin_edges: list of interval boundaries [x0, x1, x2, ...] where intervals are [x0,x1],[x1,x2],...
        """
        x = np.linspace(x_min, x_max, 10001)
        pdf = self.boltzmann_distribution(x, mu_eff, sigma_eff)
        peaks = self.find_stable_points(mu_eff, x_min=x_min, x_max=x_max, n_points=2000)
        if len(peaks) == 0:
            return [], [], [], None
        if len(peaks) == 1:
            # 只有一个稳定点：整体质量为1，方差由 pdf
            mean = np.sum(x * pdf) * (x[1] - x[0])
            var = np.sum((x - mean) ** 2 * pdf) * (x[1] - x[0])
            return [peaks[0]], [1.0], [max(var, 1e-8)], [x_min, x_max]
        # len(peaks) >= 2 : 选择其中两个最主要的（通常为两侧的第一个和最后一个）
        # basins: 由两个最近的鞍点（drift零交叉的中点）划分
        # 我们通过找到 pdf 的局部最小点作为分界
        from scipy.signal import argrelextrema
        # 找局部最小的索引
        minima_idx = argrelextrema(pdf, np.less)[0]
        # 如果找到了 minima，选择位于两个主要峰之间的那个最小值作为分界
        # 否则用中点分割
        peak_positions = peaks
        # 对于多个峰，找到相邻峰之间的最小值以划分 basin
        basin_edges = [x_min]
        for i in range(len(peak_positions) - 1):
            left = peak_positions[i]; right = peak_positions[i + 1]
            # 在 x 网格上找到位于(left,right) 的 pdf 最小值
            mask = (x > left) & (x < right)
            if np.any(mask):
                xi = x[mask]; yi = pdf[mask]
                min_idx = np.argmin(yi)
                cut = xi[min_idx]
            else:
                cut = 0.5 * (left + right)
            basin_edges.append(float(cut))
        basin_edges.append(x_max)
        # 现在对每个 basin 计算权重和局部均值方差
        weights = []
        variances = []
        peaks_used = []
        dx = x[1] - x[0]
        for i in range(len(basin_edges) - 1):
            L, R = basin_edges[i], basin_edges[i + 1]
            mask = (x >= L) & (x <= R)
            mass = np.trapz(pdf[mask], x[mask])
            if mass <= 0:
                weights.append(0.0)
                variances.append(1e-8)
                peaks_used.append((L + R) / 2.0)
                continue
            mean_loc = np.trapz(x[mask] * pdf[mask], x[mask]) / mass
            var_loc = np.trapz((x[mask] - mean_loc) ** 2 * pdf[mask], x[mask]) / mass
            weights.append(float(mass))
            variances.append(float(max(var_loc, 1e-8)))
            peaks_used.append(mean_loc)
        # 归一化 weights
        wsum = sum(weights)
        if wsum > 0:
            weights = [w / wsum for w in weights]
        return peaks_used, weights, variances, basin_edges

    def cavity_equations(self, variables):
        p = self.params
        m, q, phi_plus, G = variables
        S = p['S']
        mu_eff = (p['μ_c'] + p['μ_d'] * m +
                  p['μ_e'] * (m ** 2 + (1.21 - m ** 2) / S) +
                  G * p['ρ_d'] * p['σ_d'] ** 2 * (phi_plus + (1 - phi_plus)) * m)
        sigma_eff2 = (p['σ_c'] ** 2 +
                      p['σ_d'] ** 2 * (1.21 - m ** 2) +
                      1.4641 * p['σ_e'] ** 2)
        sigma_eff = np.sqrt(max(sigma_eff2, 1e-12))
        x_range = np.linspace(-2.5, 2.5, 1000)
        pdf = self.boltzmann_distribution(x_range, mu_eff, sigma_eff)
        dx = x_range[1] - x_range[0]
        m_new = np.sum(x_range * pdf) * dx
        q_new = np.sum(x_range ** 2 * pdf) * dx
        phi_plus_new = np.sum(pdf[x_range > 0]) * dx
        # G update (保守版本)
        stable_points = self.find_stable_points(mu_eff)
        if len(stable_points) > 0:
            G_vals = []
            for x_st in stable_points:
                denom = 3 * x_st ** 2 - 1
                if abs(denom) < 1e-8:
                    G_vals.append(0.0)
                else:
                    G_vals.append(1.0 / denom)
            G_new = np.mean(G_vals)
        else:
            G_new = G
        denominator = 2.63 - G * p['ρ_d'] * p['σ_d'] ** 2 * (phi_plus + (1 - phi_plus)) - 2 * G * p['σ_e'] ** 2 * q
        if abs(denominator) < p.get('denom_eps', 1e-8):
            denominator = np.sign(denominator) * p.get('denom_eps', 1e-8)
        G_new = 1.0 / denominator
        residuals = np.array([m_new - m, q_new - q, phi_plus_new - phi_plus, G_new - G])
        return residuals, np.array([m_new, q_new, phi_plus_new, G_new]), mu_eff, sigma_eff

    def solve_iterative(self, initial_guess=None, verbose=False):
        p = self.params
        if initial_guess is None:
            initial_guess = np.array([0.0, 1.21, 0.5, 1.0 / 2.63])
        variables = initial_guess.copy()
        self.convergence_history = [variables.copy()]
        for i in range(p.get('max_iter', 2000)):
            old = variables.copy()
            residuals, new_vars, mu_eff, sigma_eff = self.cavity_equations(variables)
            if np.any(np.isnan(new_vars)) or np.any(np.isinf(new_vars)):
                variables = old
                continue
            variables = (1 - p.get('relaxation', 0.3)) * old + p.get('relaxation', 0.3) * new_vars
            variables[0] = np.clip(variables[0], -2.5, 2.5)
            variables[1] = max(variables[1], 1e-6)
            variables[2] = np.clip(variables[2], 1e-6, 1.0 - 1e-6)
            variables[3] = np.clip(variables[3], -1e6, 1e6)
            self.convergence_history.append(variables.copy())
            if np.linalg.norm(residuals) < p.get('tol', 1e-6) and np.linalg.norm(variables - old) < p.get('tol', 1e-6):
                break
        # 最终结果
        residuals, vars_final, mu_eff_final, sigma_eff_final = self.cavity_equations(variables)
        result = {
            'm': vars_final[0],
            'q': vars_final[1],
            'phi': vars_final[2],
            'G': vars_final[3],
            'μ_eff': mu_eff_final,
            'σ_eff': sigma_eff_final,
            'σ_sq': sigma_eff_final ** 2,
            'converged': np.linalg.norm(residuals) < p.get('tol', 1e-6)
        }
        # 在求解结束后，基于 μ_eff 与 σ_eff 找稳定点并分解 basin -> 若存在两个以上稳定点则设定 μ1/μ2/σ1_sq/σ2_sq/phi_comp
        peaks, weights, variances, basin_edges = self.find_all_stable_points_and_basins(result['μ_eff'], result['σ_eff'])
        if len(peaks) >= 2:
            # 选择左右两个主要峰（按 mu 排序）
            peaks_sorted = sorted(peaks)
            # 若权重对应个数>2，我们合并左右两端主要两个（极端情况）
            # 我们挑选最左与最右作为两峰
            μ1 = peaks_sorted[0]
            μ2 = peaks_sorted[-1]
            # 对应 weights/variances 也采取左右对应项
            # peaks_used 返回的是每个 basin 的 mean (升序)， basin_edges 则显示边界
            # 找索引
            idx_left = 0
            idx_right = len(peaks) - 1
            w_left = weights[idx_left]
            w_right = weights[idx_right]
            var_left = variances[idx_left]
            var_right = variances[idx_right]
            # 存到 solver 上
            self.μ1 = μ1; self.μ2 = μ2
            self.σ1_sq = float(var_left); self.σ2_sq = float(var_right)
            self.phi_comp = float(w_right)  # 右峰权重
            self.basin_edges = basin_edges
        elif len(peaks) == 1:
            self.μ1 = peaks[0]; self.μ2 = None
            self.σ1_sq = variances[0] if variances else None
            self.σ2_sq = None
            self.phi_comp = 1.0
            self.basin_edges = basin_edges
        else:
            self.μ1 = None; self.μ2 = None
            self.σ1_sq = None; self.σ2_sq = None; self.phi_comp = None
            self.basin_edges = None
        return result

    def plot_convergence(self):
        if not self.convergence_history:
            print("没有收敛历史")
            return
        history = np.array(self.convergence_history)
        iters = np.arange(history.shape[0])
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        labels = ['m', 'q', 'φ_+', 'G']
        for i, ax in enumerate(axes.flat):
            ax.plot(iters, history[:, i], '-b')
            ax.set_title(labels[i])
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_bimodal_distribution(self, x_range=(-3, 3), n_points=10000):
        res = self.solve_iterative(verbose=False)
        x = np.linspace(x_range[0], x_range[1], n_points)
        pdf = self.boltzmann_distribution(x, res['μ_eff'], res['σ_eff'])
        plt.figure(figsize=(8, 5))
        plt.plot(x, pdf, 'k-', lw=2, label='Theory (boltzmann)')
        plt.axvline(res['m'], color='purple', linestyle='--', label=f'm={res["m"]:.3f}')
        # 若 solver 中有 μ1/μ2，则画出两个高斯近似
        if (self.μ1 is not None) and (self.μ2 is not None) and (self.σ1_sq is not None) and (self.σ2_sq is not None):
            xg = np.linspace(x_range[0], x_range[1], 2000)
            g1 = (1.0 - self.phi_comp) * np.exp(-(xg - self.μ1) ** 2 / (2 * self.σ1_sq)) / np.sqrt(2 * np.pi * self.σ1_sq)
            g2 = (self.phi_comp) * np.exp(-(xg - self.μ2) ** 2 / (2 * self.σ2_sq)) / np.sqrt(2 * np.pi * self.σ2_sq)
            total = g1 + g2
            # 归一化（理论玻尔兹曼已归一，二高斯也归一）
            total = total / np.trapz(total, xg)
            plt.plot(xg, total, 'r--', lw=1.6, label='Theory two-Gauss approx')
            plt.plot(xg, g1, 'b:', lw=1.0, label='mode1 (gauss approx)')
            plt.plot(xg, g2, 'g:', lw=1.0, label='mode2 (gauss approx)')
            plt.axvline(self.μ1, color='b', linestyle=':', alpha=0.8)
            plt.axvline(self.μ2, color='g', linestyle=':', alpha=0.8)
            plt.title(f'理论双峰 (μ1={self.μ1:.3f}, μ2={self.μ2:.3f}, phi_right={self.phi_comp:.3f})')
        else:
            plt.title('理论分布（单峰或无法分解为双峰）')
        plt.xlabel('x'); plt.ylabel('P(x)')
        plt.legend(); plt.grid(alpha=0.3); plt.show()

# ---------------- 绘图比较与拟合 ----------------

def plot_theory_and_fit(solver, x_final, x_range=(-3, 3), bins=100,
                        method='kde', kde_bandwidth=None, gmm_n_components=2,
                        show_hist=True, hist_alpha=0.4, figsize=(10, 6)):
    x_final = np.asarray(x_final)
    res = solver.solve_iterative(verbose=False)
    print("Diagnostics after solve_iterative:", res)
    print("Solver peaks:", solver.μ1, solver.μ2, "sigmas:", solver.σ1_sq, solver.σ2_sq, "phi_right:", solver.phi_comp)
    x_theory = np.linspace(x_range[0], x_range[1], 2000)
    theory_density = solver.boltzmann_distribution(x_theory, res['μ_eff'], res['σ_eff'])
    counts, bin_edges = np.histogram(x_final, bins=bins, range=x_range, density=False)
    N = x_final.size
    widths = np.diff(bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    hist_mass = counts.astype(float) / N
    hist_density = np.zeros_like(hist_mass)
    nz = widths > 0
    hist_density[nz] = hist_mass[nz] / widths[nz]
    x_fit = np.linspace(x_range[0], x_range[1], 2000)
    fit_density = np.zeros_like(x_fit)
    if method == 'kde':
        try:
            from sklearn.neighbors import KernelDensity
            bw = kde_bandwidth
            if bw is None:
                std = np.std(x_final)
                n = max(1, x_final.size)
                bw = 1.06 * std * (n ** (-1/5)) if std > 0 else 0.1
            kde = KernelDensity(kernel='gaussian', bandwidth=bw)
            kde.fit(x_final.reshape(-1, 1))
            log_d = kde.score_samples(x_fit.reshape(-1, 1))
            fit_density = np.exp(log_d)
        except Exception as e:
            # 退化 KDE
            std = np.std(x_final)
            n = max(1, x_final.size)
            bw = kde_bandwidth if kde_bandwidth is not None else (1.06 * std * (n ** (-1/5)) if std > 0 else 0.1)
            bw = max(bw, 1e-6)
            from math import sqrt, pi, exp
            def g(u): return exp(-0.5 * u * u) / sqrt(2 * pi)
            for i, xi in enumerate(x_fit):
                u = (xi - x_final) / bw
                fit_density[i] = np.sum(np.array([g(ui) for ui in u])) / (n * bw)
    elif method == 'gmm':
        try:
            from sklearn.mixture import GaussianMixture
            X = x_final.reshape(-1, 1)
            gmm = GaussianMixture(n_components=gmm_n_components, covariance_type='full', random_state=0)
            gmm.fit(X)
            logprob = gmm.score_samples(x_fit.reshape(-1, 1))
            fit_density = np.exp(logprob)
            print("GMM found means:", gmm.means_.ravel(), "weights:", gmm.weights_)
        except Exception as e:
            # fallback: use two largest histogram bins
            idx_sorted = np.argsort(counts)[-2:]
            mu_est = bin_centers[idx_sorted]
            sigma_est = np.array([np.std(x_final[(x_final >= (bc - widths[0]/2)) & (x_final < (bc + widths[0]/2))]) + 1e-3 for bc in mu_est])
            amp = counts[idx_sorted].astype(float) / N
            from numpy import exp, sqrt, pi
            fit_density = np.zeros_like(x_fit)
            for i in range(len(mu_est)):
                fit_density += (amp[i] * exp(-(x_fit - mu_est[i])**2 / (2 * sigma_est[i]**2)) / (sqrt(2*pi)*sigma_est[i]))
    else:
        raise ValueError("method must be 'kde' or 'gmm'")
    # 归一化 fit_density
    area = np.trapz(fit_density, x_fit)
    if area > 0:
        fit_density = fit_density / area
    plt.figure(figsize=figsize)
    if show_hist:
        plt.bar(bin_centers, hist_density, width=widths, alpha=hist_alpha, color='lightgray', edgecolor='black',
                label='Numerical histogram (density)')
    plt.plot(x_theory, theory_density, 'k-', lw=2, label='Theory density (boltzmann)')
    plt.plot(x_fit, fit_density, 'r--', lw=2, label=f'Fit ({method})')
    # 若 solver 已经分解出两峰参数，则绘制两个高斯近似
    if (solver.μ1 is not None) and (solver.μ2 is not None) and (solver.σ1_sq is not None) and (solver.σ2_sq is not None):
        xg = np.linspace(x_range[0], x_range[1], 2000)
        g1 = (1.0 - solver.phi_comp) * np.exp(-(xg - solver.μ1) ** 2 / (2 * solver.σ1_sq)) / np.sqrt(2 * np.pi * solver.σ1_sq)
        g2 = (solver.phi_comp) * np.exp(-(xg - solver.μ2) ** 2 / (2 * solver.σ2_sq)) / np.sqrt(2 * np.pi * solver.σ2_sq)
        total = g1 + g2
        total = total / np.trapz(total, xg)
        plt.plot(xg, total, 'g:', lw=1.8, label='Theory two-Gauss approx')
        plt.plot(xg, g1, 'b:', lw=1.0, label='mode1 (gauss approx)')
        plt.plot(xg, g2, 'g:', lw=1.0, label='mode2 (gauss approx)')
        plt.axvline(solver.μ1, color='b', linestyle=':', alpha=0.8)
        plt.axvline(solver.μ2, color='g', linestyle=':', alpha=0.8)
    plt.xlabel('x'); plt.ylabel('Density')
    plt.xlim(x_range)
    plt.title('Theory density and numerical fit comparison')
    plt.legend(); plt.grid(alpha=0.25); plt.show()
    return {
        'x_theory': x_theory, 'theory_density': theory_density,
        'x_fit': x_fit, 'fit_density': fit_density,
        'bin_centers': bin_centers, 'hist_density': hist_density
    }

def compute_hist_probabilities(data, bins=100, range=None):
    counts, bin_edges = np.histogram(data, bins=bins, range=range, density=False)
    N = float(data.size)
    widths = np.diff(bin_edges)
    mass = counts.astype(float) / N
    density = np.zeros_like(mass)
    nonzero = widths > 0
    density[nonzero] = mass[nonzero] / widths[nonzero]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_edges, bin_centers, mass, density

def plot_compare_distribution_two_panel(solver, final_states, x_range=(-3, 3), bins=100, alpha_hist=0.6):
    final_states = np.asarray(final_states)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(final_states, bins=bins, density=True,
                 alpha=alpha_hist, color='tab:blue', edgecolor='black', label='Numerical (final, density)')
    x = np.linspace(x_range[0], x_range[1], 2000)
    # 使用 solver 的双高斯近似（若存在）
    if (solver.μ1 is not None) and (solver.μ2 is not None):
        σ1_sq = max(solver.σ1_sq, 1e-12)
        σ2_sq = max(solver.σ2_sq, 1e-12)
        phi_attr = solver.phi_comp if solver.phi_comp is not None else 0.5
        gaussian1 = ((1 - phi_attr) * np.exp(-(x - solver.μ1) ** 2 / (2 * σ1_sq)) / np.sqrt(2 * np.pi * σ1_sq))
        gaussian2 = (phi_attr * np.exp(-(x - solver.μ2) ** 2 / (2 * σ2_sq)) / np.sqrt(2 * np.pi * σ2_sq))
        total = gaussian1 + gaussian2
        axes[0].plot(x, total, 'k-', lw=2, label='Theory total (two-gauss)')
        axes[0].plot(x, gaussian1, 'b--', lw=1.2, label='Theory mode1 (gauss)')
        axes[0].plot(x, gaussian2, 'r--', lw=1.2, label='Theory mode2 (gauss)')
        axes[0].axvline(solver.μ1, color='b', linestyle=':', alpha=0.8)
        axes[0].axvline(solver.μ2, color='r', linestyle=':', alpha=0.8)
    else:
        res = solver.solve_iterative(verbose=False)
        total = solver.boltzmann_distribution(x, res['μ_eff'], res['σ_eff'])
        axes[0].plot(x, total, 'k-', lw=2, label='Theory (boltzmann)')
    axes[0].set_xlim(x_range); axes[0].set_xlabel('x'); axes[0].set_ylabel('Density')
    axes[0].set_title('Density: Numerical vs Theory'); axes[0].legend(); axes[0].grid(alpha=0.25)
    # 右面板：每-bin mass 对比
    bin_edges = np.linspace(x_range[0], x_range[1], bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths = np.diff(bin_edges)
    counts, _ = np.histogram(final_states, bins=bin_edges, density=False)
    N = final_states.size
    mass = counts.astype(float) / N
    total_prob = mass.sum()
    # 计算理论 mass（通过细网格积分）
    finer = 5
    x_fine = np.linspace(x_range[0], x_range[1], int(bins * finer) + 1)
    if (solver.μ1 is not None) and (solver.μ2 is not None):
        gaussian1_f = ((1 - solver.phi_comp) * np.exp(-(x_fine - solver.μ1) ** 2 / (2 * max(solver.σ1_sq, 1e-12))) / np.sqrt(2 * np.pi * max(solver.σ1_sq, 1e-12)))
        gaussian2_f = (solver.phi_comp * np.exp(-(x_fine - solver.μ2) ** 2 / (2 * max(solver.σ2_sq, 1e-12))) / np.sqrt(2 * np.pi * max(solver.σ2_sq, 1e-12)))
        theory_density_fine = gaussian1_f + gaussian2_f
    else:
        res = solver.solve_iterative(verbose=False)
        theory_density_fine = solver.boltzmann_distribution(x_fine, res['μ_eff'], res['σ_eff'])
    theory_mass = np.zeros_like(mass)
    for i in range(len(bin_edges) - 1):
        left, right = bin_edges[i], bin_edges[i + 1]
        mask = (x_fine >= left) & (x_fine <= right)
        xs = x_fine[mask]; ys = theory_density_fine[mask]
        if xs.size == 0:
            xc = 0.5 * (left + right)
            yc = np.interp(xc, x_fine, theory_density_fine)
            theory_mass[i] = yc * (right - left)
        else:
            theory_mass[i] = np.trapz(ys, xs)
    theory_total = theory_mass.sum()
    axes[1].bar(bin_centers, mass, width=widths, alpha=0.6, color='tab:blue', edgecolor='black',
                label='Numerical: per-bin probability (mass)')
    axes[1].plot(bin_centers, theory_mass, 'r.-', lw=1.6, ms=6, label='Theory: per-bin probability (integrated)')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('Probability per bin (mass)')
    axes[1].set_title('Per-bin probability: Numerical vs Theory')
    axes[1].legend(); axes[1].grid(alpha=0.25)
    axes[1].text(0.98, 0.95, f'sum(numerical)={total_prob:.6f}\nsum(theory)={theory_total:.6f}',
                 transform=axes[1].transAxes, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    plt.tight_layout(); plt.show()
    return {'bin_edges': bin_edges, 'bin_centers': bin_centers, 'mass': mass, 'theory_mass': theory_mass,
            'total_prob': total_prob, 'theory_total': theory_total}

# ---------------- 主程序示例 ----------------

if __name__ == "__main__":
    # 参数（示例）
    s = 2000
    mu_c = 0.0
    sigma_c = 0.4
    mu_d = 0.2
    sigma_d = 0.3
    rho_d = 1.0
    mu_e = 0.2
    sigma_e = 0.1

    params = {
        'μ_c': mu_c, 'μ_d': mu_d, 'μ_e': mu_e,
        'σ_c': sigma_c, 'σ_d': sigma_d, 'σ_e': sigma_e,
        'ρ_d': rho_d, 'S': s,
        'max_iter': 2000, 'tol': 1e-6, 'relaxation': 0.3,
        'denom_eps': 1e-6, 'denom_warn_limit': 20
    }

    solver = BistableCavitySolver(params)
    print("求解理论自洽方程（可能需要数十到数百次迭代）...")
    t0 = time.time()
    res = solver.solve_iterative(verbose=True)
    t1 = time.time()
    print(f"理论求解完成（{t1 - t0:.2f} s）\n res: {res}")
    print("识别到的理论峰（μ1, μ2）:", solver.μ1, solver.μ2)
    print("对应方差 σ1_sq, σ2_sq:", solver.σ1_sq, solver.σ2_sq)
    print("右峰权重 phi_comp:", solver.phi_comp)
    # 展示收敛和理论分布（含双高斯近似，如可行）
    solver.plot_convergence()
    solver.plot_bimodal_distribution(x_range=(-3, 3))

    # 数值仿真（RK4）
    rng = np.random.default_rng(42)
    c_i, d_ij, d_ji, e_ijk = generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, rng=rng)
    x_init = np.full(s, 0.6)
    t_steps = 3000
    dt = 0.01

    print("\n开始数值仿真（RK4）...")
    t0 = time.time()
    x_final, survival_counts = dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps, dt=dt, record_every=10)
    t1 = time.time()
    print(f"数值仿真完成，耗时 {(t1 - t0):.2f} 秒")

    # 绘制最终分布
    plot_final_state_distribution(x_final)
    phi_num = estimate_phi_from_final_states(x_final)
    print(f"数值仿真 φ_num(final) = {phi_num:.4f}")

    # 在同一张图中绘制理论密度与数值拟合曲线（KDE）
    try:
        plot_theory_and_fit(solver, x_final, x_range=(-3, 3), bins=100, method='kde')
    except Exception as e:
        print("绘制理论与拟合时出现异常：", e)
        print("改用 method='gmm' 作为回退。")
        plot_theory_and_fit(solver, x_final, x_range=(-3, 3), bins=100, method='gmm')

    # 也可调用两面板比较
    plot_compare_distribution_two_panel(solver, x_final, x_range=(-3, 3), bins=100)

    print("程序结束。")