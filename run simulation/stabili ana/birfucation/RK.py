import numpy as np
import matplotlib.pyplot as plt


def fixed_point_iteration():
    # Parameters
    c_start = -2.0  # Initial c value
    x0_start = -1.5  # Initial x0 value
    c_end = 2.0  # End c value
    num_c_values = 100  # Number of c values to iterate over
    num_iterations = 200  # Number of iterations per c value

    # Define range of c values
    c_values = np.linspace(c_start, c_end, num_c_values)
    fixed_points = np.zeros_like(c_values)  # Storage for fixed points

    # Iteration for each value of c
    for i in range(num_c_values):
        c = c_values[i]
        x0 = x0_start

        # Perform iterations
        for _ in range(num_iterations):
            x0 = np.cbrt(x0 + c)  # Iterative update using cube root

        # Store the result after the final iteration
        fixed_points[i] = x0

    # Plot x0 vs c
    plt.figure()
    plt.plot(c_values, fixed_points, '-o')
    plt.xlabel('c')
    plt.ylabel('Fixed Point x_0')
    plt.title('Fixed Point x_0 vs Parameter c')
    plt.grid(True)
    plt.show()


# Run the function
fixed_point_iteration()