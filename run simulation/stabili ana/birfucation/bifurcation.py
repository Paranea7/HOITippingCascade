import numpy as np
import matplotlib.pyplot as plt

# Function whose roots we want to find
def func(x, c):
    return x**3 - x - c

def fixed_point_iteration():
    # Parameters
    c_start = -2.0  # Initial c value
    x0_start = -1.5  # Initial x0 value for increasing c
    c_end = 2.0  # End c value
    num_c_values = 100  # Number of c values to iterate over
    num_iterations = 200  # Number of iterations per c value

    # Define range of c values for increasing and decreasing c
    c_values_increasing = np.linspace(c_start, c_end, num_c_values)
    c_values_decreasing = np.linspace(c_end, c_start, num_c_values)
    fixed_points_increasing = np.zeros_like(c_values_increasing)  # Storage for fixed points
    fixed_points_decreasing = np.zeros_like(c_values_decreasing)  # Storage for fixed points

    # Iteration for increasing c values
    for i in range(num_c_values):
        c = c_values_increasing[i]
        x0 = x0_start
        for _ in range(num_iterations):
            x0 = np.cbrt(x0 + c)  # Iterative update using cube root
        fixed_points_increasing[i] = x0

    # Iteration for decreasing c values
    for i in range(num_c_values):
        c = c_values_decreasing[i]
        x0 = 1.5  # Initial x0 value for decreasing c
        for _ in range(num_iterations):
            x0 = np.cbrt(x0 + c)  # Iterative update using cube root
        fixed_points_decreasing[i] = x0

    return c_values_increasing, fixed_points_increasing, c_values_decreasing, fixed_points_decreasing

def find_fixed_points():
    # Set range for c values
    c_values = np.linspace(-2, 2, 400)
    roots_np = []

    for c in c_values:
        # Using np.roots method to find real roots
        roots = np.roots([1, 0, -1, -c])
        real_roots = roots[np.isreal(roots)].real
        for root in real_roots:
            roots_np.append((c, root))
    return roots_np

# Run fixed point iteration
c_values_increasing, fixed_points_increasing, c_values_decreasing, fixed_points_decreasing = fixed_point_iteration()

# Find fixed points using both methods
roots_np = find_fixed_points()

# Separate coordinates for plotting
c_points_np, x_points_np = zip(*roots_np) if roots_np else ([], [])

# Plot both results on the same graph
plt.figure(figsize=(12, 6))

# Fixed points from iteration
plt.plot(c_values_increasing, fixed_points_increasing, '-o', label='Fixed Points (c increasing)')
plt.plot(c_values_decreasing, fixed_points_decreasing, '-s', label='Fixed Points (c decreasing)')

# Fixed points from np.roots
plt.scatter(c_points_np, x_points_np, s=10, c='blue', label='Using np.roots')


plt.xlabel('c')
plt.ylabel('Fixed Point x_0')
plt.title('Fixed Points x_0 vs Parameter c')
plt.legend()
plt.grid(True)
plt.show()