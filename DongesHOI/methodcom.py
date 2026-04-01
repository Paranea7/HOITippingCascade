import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Physics & Range Setup
# ==========================================
# The critical field H_c = 2/(3*sqrt(3)) approx 0.3849
H_c = 0.3849
M_range = np.linspace(0, 0.55, 120) # Extended range to see truncation clearly

def get_exact_roots(M):
    """Numerically solve x^3 - x - M = 0"""
    roots = np.roots([1, 0, -1, -M])
    real_roots = np.sort(roots[np.isreal(roots)].real)
    if len(real_roots) == 3:
        return real_roots[0], real_roots[-1]
    else:
        return np.nan, real_roots[-1]

# ==========================================
# 2. Calculating Branch Positions with Truncation
# ==========================================
x_pos_exact, x_neg_exact = [], []
x_pos_1st, x_neg_1st = [], []
x_pos_2nd, x_neg_2nd = [], []

for M in M_range:
    # 1. Exact Roots
    xn_e, xp_e = get_exact_roots(M)
    x_neg_exact.append(xn_e)
    x_pos_exact.append(xp_e)

    # 2. First-order with Truncation
    # Negative branch disappears if M > H_c
    x_neg_1st.append(-1 + M/2 if M <= H_c else np.nan)
    x_pos_1st.append(1 + M/2)

    # 3. Second-order with Truncation
    # Negative branch disappears if M > H_c
    x_neg_2nd.append(-1 + M/2 + 0.375 * M**2 if M <= H_c else np.nan)
    x_pos_2nd.append(1 + M/2 - 0.375 * M**2)

# ==========================================
# 3. Plotting
# ==========================================
plt.figure(figsize=(8, 6))

# Exact Roots
plt.plot(M_range, x_pos_exact, 'k-', linewidth=2, label='Exact Roots (Cubic)')
plt.plot(M_range, x_neg_exact, 'k-', linewidth=2)

# 1st Order (Truncated)
plt.plot(M_range, x_pos_1st, 'r--', alpha=0.6, label='1st Order (Truncated)')
plt.plot(M_range, x_neg_1st, 'r--', alpha=0.6)

# 2nd Order (Truncated)
plt.plot(M_range, x_pos_2nd, 'b-.', linewidth=1.5, label='2nd Order (Truncated)')
plt.plot(M_range, x_neg_2nd, 'b-.', linewidth=1.5)

# Decoration
plt.axvline(H_c, color='gray', linestyle=':', label=r'Saddle-node $H_c$')
plt.axhline(0, color='black', linewidth=0.5)
plt.title("Equilibrium Positions with Truncation Mechanism")
plt.xlabel("Effective Field $M$")
plt.ylabel("Root Position $x^*$")
plt.legend(frameon=False, loc='upper left')
plt.grid(alpha=0.2)
plt.ylim(-1.2, 1.4)
plt.xlim(0, 0.55)

plt.show()
