import numpy as np
import matplotlib.pyplot as plt


def fixed_point_iteration_with_coupling_and_external_force(c1_values, d12, amplitude, period):
    num_c1_values = len(c1_values)

    fixed_points_system1 = np.zeros_like(c1_values)
    fixed_points_system2 = np.zeros_like(c1_values)

    # Generate the periodic external force
    time = np.linspace(0, 2 * np.pi, 200)
    external_force = amplitude * np.sin(time / period)

    for i in range(num_c1_values):
        c1 = c1_values[i]
        x1 = -1.5  # Initial guess for System 1
        x2 = -1.5  # Initial guess for System 2

        # Iterate to find the fixed point considering coupling and external force
        for t in range(len(external_force)):
            x1 = np.cbrt(x1 + c1 + external_force[t])
            x2 = np.cbrt(x2 + d12 * x1)  # Adjust x2 to include coupling with x1

        fixed_points_system1[i] = x1
        fixed_points_system2[i] = x2

    return fixed_points_system1, fixed_points_system2


def compute_fixed_points_with_coupling_and_force(c1_values, d12):
    amplitude = np.pi / 2
    period = 40
    return fixed_point_iteration_with_coupling_and_external_force(c1_values, d12, amplitude, period)


def compute_uncoupled_fixed_points(c1_values):
    c2_values = np.linspace(-4, 4, 200)
    fixed_points_system1 = np.zeros_like(c1_values)
    fixed_points_system2 = np.zeros_like(c1_values)

    for i in range(len(c1_values)):
        c1 = c1_values[i]
        x1 = -1.5
        for _ in range(200):
            x1 = np.cbrt(x1 + c1)
        fixed_points_system1[i] = x1

        c2 = c2_values[i % len(c2_values)]
        x2 = -1.5
        for _ in range(200):
            x2 = np.cbrt(x2 + c2)
        fixed_points_system2[i] = x2

    return fixed_points_system1, fixed_points_system2


# Parameters
c1_values = np.linspace(-2, 2, 200)
d12 = 0.5  # Coupling strength

# Compute fixed points for coupled system with external force
fixed_points_system1_with_coupling_and_force, fixed_points_system2_with_coupling_and_force = compute_fixed_points_with_coupling_and_force(
    c1_values, d12)

# Compute fixed points for uncoupled system
fixed_points_system1_uncoupled, fixed_points_system2_uncoupled = compute_uncoupled_fixed_points(c1_values)

# Plot results
plt.figure(figsize=(14, 7))

# Coupled system with external force
plt.plot(c1_values, fixed_points_system1_with_coupling_and_force, '-o',
         label='System 1 Fixed Points (Coupled with External Force)')
plt.plot(c1_values, fixed_points_system2_with_coupling_and_force, '-s',
         label='System 2 Fixed Points (Coupled with External Force)')

# Uncoupled system fixed points
plt.plot(c1_values, fixed_points_system1_uncoupled, '--o', label='System 1 Fixed Points (Uncoupled)')
plt.plot(c1_values, fixed_points_system2_uncoupled, '--s', label='System 2 Fixed Points (Uncoupled)')

plt.xlabel('c1')
plt.ylabel('Fixed Point')
plt.title(f'Fixed Points x_1 and x_2 vs Parameter c1 with and without Coupling and External Force d12={d12}')
plt.legend()
plt.grid(True)
plt.show()