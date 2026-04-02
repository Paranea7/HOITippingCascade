import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# 1. Physics Parameters
# ==========================================
S = 50  # Number of nodes (keep small for clear visualization)
MU_E = 0.5  # Coupling strength (set high enough to trigger tipping)
SIG_U = 0.5 # Individual noise level
DT = 0.05  # Time step
STEPS = 1600  # Number of frames

# Initial state: all nodes near the negative stable point -1
x = np.full(S, -1.0) + np.random.normal(0, 0.05, S)
# Individual quenched noise (external inputs)
xi = np.random.normal(0, SIG_U, S)

# Node positions for a circular layout (Fully Connected Network)
theta = np.linspace(0, 2 * np.pi, S, endpoint=False)
pos_x = np.cos(theta)
pos_y = np.sin(theta)

# ==========================================
# 2. Setup Figure
# ==========================================
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121)  # Network view
ax2 = fig.add_subplot(122)  # Order parameter m(t)

# Network Plot initialization
# We draw faint lines to represent "Fully Connected" without cluttering
for i in range(0, S, 5):  # Only draw subset of edges for speed/clarity
    for j in range(i + 1, S, 10):
        ax1.plot([pos_x[i], pos_x[j]], [pos_y[i], pos_y[j]], color='gray', alpha=0.05, lw=0.5)

scatter = ax1.scatter(pos_x, pos_y, c=x, cmap='coolwarm', s=50, edgecolors='k', zorder=3, vmin=-1.2, vmax=1.2)
ax1.set_title("Network State Cascade")
ax1.axis('off')

# Time series Plot initialization
m_history = []
line, = ax2.plot([], [], lw=2, color='firebrick')
ax2.set_xlim(0, STEPS)
ax2.set_ylim(-1.2, 1.2)
ax2.set_xlabel("Time Steps")
ax2.set_ylabel("Order Parameter $m(t)$")
ax2.set_title("Global Dynamics")
ax2.grid(ls=':', alpha=0.6)


# ==========================================
# 3. Animation Function
# ==========================================
def update(frame):
    global x
    m = np.mean(x)

    # Dynamics Equation: dx = x - x^3 + mu_e * m^2 + xi
    # Note: mu_e * m^2 provides the collective feedback
    drift = x - x ** 3 + MU_E * (m ** 2) + xi
    x += drift * DT
    x = np.clip(x, -2, 2)

    # Update Scatter colors
    scatter.set_array(x)

    # Update Time series
    m_history.append(m)
    line.set_data(range(len(m_history)), m_history)

    # Visual cue for Tipping
    if m > 0:
        ax2.set_facecolor('#fff5f5')  # Light red background when tipped

    return scatter, line


# Create Animation
ani = animation.FuncAnimation(fig, update, frames=STEPS, interval=50, blit=True)

plt.tight_layout()
plt.show()
