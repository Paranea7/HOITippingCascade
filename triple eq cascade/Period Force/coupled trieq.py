import numpy as np
import matplotlib.pyplot as plt


def fixed_point_iteration(c1_values, c2_values, d12):
    # Parameters
    num_c1_values = len(c1_values)
    num_c2_values = len(c2_values)

    fixed_points_system1 = np.zeros_like(c1_values)  # Storage for fixed points of System 1
    fixed_points_system2 = np.zeros_like(c1_values)  # Storage for fixed points of System 2

    # Iterate over c1 values
    for i in range(num_c1_values):
        c1 = c1_values[i]
        x1 = -1.5  # Initial guess for fixed point of System 1
        # Fixed point iteration for System 1
        for _ in range(200):
            x1 = np.cbrt(x1 + c1)  # Iterative update using cube root
        fixed_points_system1[i] = x1

        # Fixed point iteration for System 2 with coupling
        c2 = c2_values[i % num_c2_values]  # Use c2 values cyclically
        x2 = -1.5  # Initial guess for fixed point of System 2
        for _ in range(200):
            x2 = np.cbrt(x2 + c2 + d12 * x1)  # Iterative update using cube root with coupling
        fixed_points_system2[i] = x2

    return fixed_points_system1, fixed_points_system2


def compute_fixed_points(c1_values, d12):
    c2_values = np.linspace(-4, 4, 200)  # Define c2 values
    return fixed_point_iteration(c1_values, c2_values, d12)


def compute_uncoupled_fixed_points(c1_values):
    c2_values = np.linspace(-4, 4, 200)  # Define c2 values
    fixed_points_system1 = np.zeros_like(c1_values)  # Storage for fixed points of System 1
    fixed_points_system2 = np.zeros_like(c1_values)  # Storage for fixed points of System 2

    for i in range(len(c1_values)):
        c1 = c1_values[i]
        x1 = -1.5  # Initial guess for fixed point of System 1
        for _ in range(200):
            x1 = np.cbrt(x1 + c1)  # Iterative update using cube root
        fixed_points_system1[i] = x1

        # Fixed point iteration for System 2 without coupling
        c2 = c2_values[i % len(c2_values)]  # Use c2 values cyclically
        x2 = -1.5  # Initial guess for fixed point of System 2
        for _ in range(200):
            x2 = np.cbrt(x2 + c2)  # Iterative update using cube root without coupling
        fixed_points_system2[i] = x2

    return fixed_points_system1, fixed_points_system2


# Parameters
c1_values = np.linspace(-2, 2, 200)
d12 = -0.5  # Coupling strength

# Compute fixed points for coupled system
fixed_points_system1_coupled, fixed_points_system2_coupled = compute_fixed_points(c1_values, d12)

# Compute fixed points for uncoupled system
fixed_points_system1_uncoupled, fixed_points_system2_uncoupled = compute_uncoupled_fixed_points(c1_values)

# Plot results
plt.figure(figsize=(14, 7))

# Coupled system fixed points
plt.plot(c1_values, fixed_points_system1_coupled, '-o', label='System 1 Fixed Points (Coupled)')
plt.plot(c1_values, fixed_points_system2_coupled, '-s', label='System 2 Fixed Points (Coupled)')

# Uncoupled system fixed points
plt.plot(c1_values, fixed_points_system1_uncoupled, '--o', label='System 1 Fixed Points (Uncoupled)')
plt.plot(c1_values, fixed_points_system2_uncoupled, '--s', label='System 2 Fixed Points (Uncoupled)')

plt.xlabel('c1')
plt.ylabel('Fixed Point')
plt.title(f'Fixed Points x_1 and x_2 vs Parameter c1 with and without Coupling d12={d12}')
plt.legend()
plt.grid(True)
plt.show()