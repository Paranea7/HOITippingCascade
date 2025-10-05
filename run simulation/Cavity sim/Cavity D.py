#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-consistent solver for the cavity-method equations you provided.

Equations implemented (refer to your labels):
 - Noise variance D:       eq:noise_variance
 - Self-consistent m:      eq:self_consistent_m
 - Self-consistent sigma2: eq:self_consistent_sigma
 - Self-consistent phi:    eq:self_consistent_phi
 - Self-consistent v:      eq:self_consistent_v
 - Bimodal mixture P(x):   eq:bimodal_distribution (mu1=-1, mu2=1)

Usage:
    Save as solver_cavity.py and run:
        python solver_cavity.py

Requires: numpy, scipy, matplotlib
"""
import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt


class CavitySolver:
    def __init__(self, params):
        """
        params: dict with keys (all floats):
            mu_c, mu_d, mu_e,
            sigma_c, sigma_d, sigma_e,
            rho_d (correlation), S (species count, unused directly but kept),
            bimodal (bool): whether to use bimodal ansatz (mu1=-1, mu2=1) or unimodal Gaussian
            mu1, mu2 (optional): positions for bimodal peaks (defaults -1, +1)
            alpha: relaxation factor (0<alpha<=1)
            eps: numerical floor for variances/denominators
        """
        # model params (means and noise scales)
        self.mu_c = params.get('mu_c', 0.0)
        self.mu_d = params.get('mu_d', 0.0)
        self.mu_e = params.get('mu_e', 0.0)

        self.sigma_c = params.get('sigma_c', 0.1)
        self.sigma_d = params.get('sigma_d', 0.0)
        self.sigma_e = params.get('sigma_e', 0.0)

        self.rho_d = params.get('rho_d', 0.0)
        self.S = params.get('S', 1e6)  # S enters scaling of distributions but not directly in SC eqs here

        # bimodal switch and peak locations
        self.bimodal = bool(params.get('bimodal', True))
        self.mu1 = params.get('mu1', -1.0)
        self.mu2 = params.get('mu2', 1.0)

        # numerical params
        self.alpha = params.get('alpha', 0.3)
        self.eps = params.get('eps', 1e-12)
        self.max_iter = int(params.get('max_iter', 2000))
        self.tol = params.get('tol', 1e-9)
        # store history?
        self.history = {'m': [], 'var': [], 'phi': [], 'v': []}

    def initialize(self, m0=0.0, var0=0.5, phi0=0.5, v0=1.0):
        # state variables
        self.m = float(m0)
        self.var = float(max(var0, self.eps))
        self.phi = float(np.clip(phi0, 0.0, 1.0))
        self.v = float(max(v0, self.eps))

    def compute_D(self):
        # eq:noise_variance
        # D = sigma_c^2 + sigma_d^2 * phi * (m^2 + var) + sigma_e^2 * phi^2 * (m^2 + var)^2
        msq_plus_var = self.m ** 2 + self.var
        D = (self.sigma_c ** 2 +
             self.sigma_d ** 2 * self.phi * msq_plus_var +
             self.sigma_e ** 2 * (self.phi ** 2) * (msq_plus_var ** 2))
        return max(D, self.eps)

    def effective_drive(self):
        # eq: step 3/4: mu_eff = mu_c + mu_d * phi * m + mu_e * phi^2 * m^2
        return self.mu_c + self.mu_d * self.phi * self.m + self.mu_e * (self.phi ** 2) * (self.m ** 2)

    def denominator_term(self):
        # the term that appears inside denominator 1 - v*( ... )
        # using what's in your equations: (rho_d sigma_d^2 phi + sigma_e^2 phi^2 (m^2 + sigma^2))
        msq_plus_var = self.m ** 2 + self.var
        return (self.rho_d * (self.sigma_d ** 2) * self.phi +
                (self.sigma_e ** 2) * (self.phi ** 2) * msq_plus_var)

    def update_once(self):
        """
        Perform one self-consistent update using your eqns:
         - eq:self_consistent_m
         - eq:self_consistent_sigma
         - eq:self_consistent_phi
         - eq:self_consistent_v

        Use relaxation (alpha) to improve stability.
        """
        D = self.compute_D()
        mu_eff = self.effective_drive()
        denom_term = self.denominator_term()

        # denominator: 1 - v * denom_term
        denom = 1.0 - self.v * denom_term
        # protect small denom
        if abs(denom) < self.eps:
            denom = np.sign(denom) * self.eps

        # eq:self_consistent_m
        m_prop = mu_eff / denom

        # eq:self_consistent_sigma (sigma^2)
        sigma2_prop = D / (2.0 * denom)
        # ensure positive
        sigma2_prop = max(sigma2_prop, self.eps)

        # eq:self_consistent_phi:
        # phi = 0.5 * [1 + erf( mu_eff / sqrt(2 D) ) ]
        arg = mu_eff / np.sqrt(2.0 * D)
        # clip arg to avoid overflow in erf (erf is safe but keep numerically stable)
        # scipy.special.erf handles large args, but still we can bound
        if arg > 1e3:
            phi_prop = 1.0
        elif arg < -1e3:
            phi_prop = 0.0
        else:
            phi_prop = 0.5 * (1.0 + erf(arg))

        # eq:self_consistent_v:
        # v = 1 / (1 - v * denom_term)  => rearranged, this is fixed point for v
        # This equation implies v solves v = 1/(1 - v*D0)  => v(1 - v*D0) = 1 => -D0 v^2 + v - 1 = 0
        # Solve quadratic: D0 v^2 - v + 1 = 0  (careful sign). But using your original form directly:
        # we treat v_prop = 1 / (1 - v * denom_term)  (as iterative update, not algebraic exact solve)
        # protect denominator again:
        denom_v = 1.0 - self.v * denom_term
        if abs(denom_v) < self.eps:
            denom_v = np.sign(denom_v) * self.eps
        v_prop = 1.0 / denom_v

        # relax updates
        a = float(self.alpha)
        m_new = (1 - a) * self.m + a * m_prop
        var_new = (1 - a) * self.var + a * sigma2_prop
        phi_new = np.clip((1 - a) * self.phi + a * phi_prop, 0.0, 1.0)
        v_new = max((1 - a) * self.v + a * v_prop, self.eps)

        return m_new, var_new, phi_new, v_new, D, mu_eff, denom

    def solve(self, max_iter=None, tol=None, verbose=True):
        if max_iter is None:
            max_iter = self.max_iter
        if tol is None:
            tol = self.tol

        for it in range(max_iter):
            m_new, var_new, phi_new, v_new, D, mu_eff, denom = self.update_once()

            # compute max change
            maxchg = max(abs(m_new - self.m), abs(var_new - self.var),
                         abs(phi_new - self.phi), abs(v_new - self.v))

            # commit
            self.m, self.var, self.phi, self.v = m_new, var_new, phi_new, v_new

            # record history
            self.history['m'].append(self.m)
            self.history['var'].append(self.var)
            self.history['phi'].append(self.phi)
            self.history['v'].append(self.v)

            if verbose and (it % 50 == 0 or maxchg < tol):
                print(f"Iter {it:4d}: m={self.m:.6f}, var={self.var:.6e}, phi={self.phi:.6f}, v={self.v:.6f}, D={D:.6e}, mu_eff={mu_eff:.6e}, denom={denom:.6e}, maxchg={maxchg:.2e}")

            if maxchg < tol:
                if verbose:
                    print(f"Converged at iter {it}, maxchg={maxchg:.2e}")
                break
        else:
            if verbose:
                print(f"Warning: reached max_iter={max_iter} without full convergence (last maxchg={maxchg:.2e})")

        # final D, mu_eff recompute for reporting
        D_final = self.compute_D()
        mu_eff_final = self.effective_drive()
        return {'m': self.m, 'var': self.var, 'phi': self.phi, 'v': self.v,
                'D': D_final, 'mu_eff': mu_eff_final, 'history': self.history}

    # Utilities for bimodal mixture plotting (eq:bimodal_distribution)
    def mixture_pdf(self, x_grid, show_components=True):
        """
        Return mixture pdf values on x_grid using current phi, var.
        For bimodal ansatz we use mu1, mu2 as provided and state variances equal to var (or allow different).
        For unimodal, return single Gaussian with mean m and variance var.
        """
        x = np.asarray(x_grid)
        if not self.bimodal:
            # single Gaussian
            var = max(self.var, self.eps)
            pdf = np.exp(-(x - self.m) ** 2 / (2.0 * var)) / np.sqrt(2.0 * np.pi * var)
            return pdf
        else:
            # bimodal: states with means mu1, mu2; assign component variances s1^2 and s2^2 such that
            # total var matches self.var: compute s1^2 = s2^2 = s_comp (simple choice)
            # From your eq:bimodal_variance: var = (1-phi)(mu1^2+s1) + phi(mu2^2+s2) - m^2
            # For simplicity choose s1 = s2 = s_comp and solve:
            m_mix = (1 - self.phi) * self.mu1 + self.phi * self.mu2
            s_comp = (self.var + m_mix ** 2 - (1 - self.phi) * (self.mu1 ** 2) - self.phi * (self.mu2 ** 2))
            # ensure positive
            s_comp = max(s_comp, self.eps)
            comp1 = (1.0 - self.phi) * np.exp(-(x - self.mu1) ** 2 / (2.0 * s_comp)) / np.sqrt(2.0 * np.pi * s_comp)
            comp2 = self.phi * np.exp(-(x - self.mu2) ** 2 / (2.0 * s_comp)) / np.sqrt(2.0 * np.pi * s_comp)
            return comp1 + comp2

    def plot_mixture(self, xlims=(-3, 3), npts=2000):
        x = np.linspace(xlims[0], xlims[1], npts)
        pdf = self.mixture_pdf(x)
        plt.figure(figsize=(7, 4))
        plt.plot(x, pdf, 'k-', lw=2, label='mixture pdf' if self.bimodal else 'pdf (unimodal)')
        if self.bimodal:
            # also plot components
            # recompute comp variances and plot
            m_mix = (1 - self.phi) * self.mu1 + self.phi * self.mu2
            s_comp = (self.var + m_mix ** 2 - (1 - self.phi) * (self.mu1 ** 2) - self.phi * (self.mu2 ** 2))
            s_comp = max(s_comp, self.eps)
            comp1 = (1.0 - self.phi) * np.exp(-(x - self.mu1) ** 2 / (2.0 * s_comp)) / np.sqrt(2.0 * np.pi * s_comp)
            comp2 = self.phi * np.exp(-(x - self.mu2) ** 2 / (2.0 * s_comp)) / np.sqrt(2.0 * np.pi * s_comp)
            plt.plot(x, comp1, 'b--', label=f'comp1 (1-phi), mu1={self.mu1:.2f}')
            plt.plot(x, comp2, 'r--', label=f'comp2 phi, mu2={self.mu2:.2f}')
        plt.axvline(self.m, color='gray', linestyle=':', label=f'mean m={self.m:.3f}')
        plt.xlabel('x')
        plt.ylabel('pdf')
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()

    def plot_history(self):
        it = np.arange(len(self.history['m']))
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1)
        plt.plot(it, self.history['m'])
        plt.xlabel('iter'); plt.ylabel('m'); plt.grid(alpha=0.3)
        plt.subplot(1, 3, 2)
        plt.plot(it, self.history['var'])
        plt.xlabel('iter'); plt.ylabel('var'); plt.grid(alpha=0.3)
        plt.subplot(1, 3, 3)
        plt.plot(it, self.history['phi'])
        plt.xlabel('iter'); plt.ylabel('phi'); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


# Example run with typical parameters (you can edit these)
if __name__ == "__main__":
    params = {
        'mu_c': 0.0,
        'mu_d': 0.5,
        'mu_e': 0.01,
        'sigma_c': 2*np.sqrt(3)/9.,
        'sigma_d': 0.3,
        'sigma_e': 0.05,
        'rho_d': 1.0,
        'S': 50.0,
        'bimodal': True,
        'mu1': -1.0,
        'mu2': 1.0,
        'alpha': 0.25,
        'eps': 1e-12,
        'max_iter': 5000,
        'tol': 1e-10
    }

    solver = CavitySolver(params)
    # sensible initial guesses: small mean, moderate variance, phi near 0.5, v ~ 1
    solver.initialize(m0=0.0, var0=0.5, phi0=0.5, v0=1.0)
    res = solver.solve(verbose=True)

    print("\nFinal results:")
    print(f"m = {res['m']:.6f}")
    print(f"var = {res['var']:.6e}")
    print(f"phi = {res['phi']:.6f}")
    print(f"v = {res['v']:.6f}")
    print(f"D = {res['D']:.6e}")
    print(f"mu_eff = {res['mu_eff']:.6e}")

    # plot mixture pdf and history
    solver.plot_mixture(xlims=(-3, 3))
    solver.plot_history()