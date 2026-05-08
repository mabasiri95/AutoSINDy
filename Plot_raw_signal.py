# -*- coding: utf-8 -*-
"""
Updated Dashboard: States, Derivatives, and Phase Portrait
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import systems # Assuming this is your local file

# 1. GENERATE DATA
x, xdot, t = systems.generate_damped_pendulum_data(
    b=0.25, c=5.0, x0=[np.pi - 0.1, 0], t_end=10, n_samples=5000, noise_level=0.05
)


# x, xdot, t = systems.generate_duffing_data(
#     delta= 0.3, alpha=-1.0, beta= 1.0, x0=[1, -1], t_end=20, n_samples=5000, noise_level= 0.01

# )
# x, xdot, t = systems.generate_complex_lorenz_data(
#     sigma=10.0, rho=28.0, beta=8./3., gamma=1.5, x0=[-8.0, 8.0, 27.0], t_end=50, n_samples=5000, noise_level=0.00
# )

def generate_complex_lorenz_data_test(
        sigma=10.0, rho=28.0, beta=8./3., gamma=1.5,
        x0=[-8.0, 8.0, 27.0], t_end=50, n_samples=5000,
        noise_level=0, noise_seed=42):
    """
    x' = sigma*(y - x)
    y' = x*(rho - z) - y
    z' = x*y - beta*z + gamma*y*sin(x + z)
    """
    def system(t, x):
        return [
            sigma*(x[1]-x[0])  + gamma*x[1]*np.sin(x[0]+x[2]),
            x[0]*(rho-x[2])-x[1],
            x[0]*x[1] - beta*x[2]
        ]

    t_eval = np.linspace(0, t_end, n_samples)
    X_clean = systems.solve_ivp(system, [0, t_end], x0,
                        t_eval=t_eval, dense_output=True).y.T
    X_noisy, X_dot = systems._add_noise_and_differentiate(
        X_clean, t_eval, noise_level, noise_seed)
    print("✓ Step 1: Generated synthetic Complex Lorenz data.")
    return X_noisy, X_dot, t_eval


# x, xdot, t = generate_complex_lorenz_data_test()




# 2. STYLE SETTINGS
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'lines.linewidth': 2
})

# 3. SETUP LAYOUT (2 Rows, 3 Columns)
# Width ratios: States and Derivatives get 1 unit, Phase Portrait gets 1.2 units (slightly wider)
fig = plt.figure(figsize=(15, 6), dpi=300, layout='constrained')
gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1.2])

# --- COLUMN 1: THE STATES (x) ---
# Top Left: Position / Angle
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(t, x[:, 0], color='#0077b6', label='$x_1$ (Angle)')
ax1.set_title('State Responses ($x$)', fontweight='bold')
ax1.set_ylabel('Position')
ax1.legend(loc='upper right', frameon=True, fontsize=8)
ax1.tick_params(labelbottom=False) # Hide x-ticks for top plot

# Bottom Left: Velocity
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax2.plot(t, x[:, 1], color='#0096c7', label='$x_2$ (Velocity)')
ax2.set_ylabel('Velocity')
ax2.set_xlabel('Time (s)')
ax2.legend(loc='upper right', frameon=True, fontsize=8)


# --- COLUMN 2: THE DERIVATIVES (xdot) ---
# Top Middle: Velocity (Rate of change of Angle)
# Note: In a standard pendulum, this usually looks identical to State 2
ax3 = fig.add_subplot(gs[0, 1], sharex=ax1)
ax3.plot(t, xdot[:, 0], color='#ae2012', linestyle='-', label='$\dot{x}_1$ (Vel)')
ax3.set_title('Derivative Responses ($\dot{x}$)', fontweight='bold')
ax3.legend(loc='upper right', frameon=True, fontsize=8)
ax3.tick_params(labelbottom=False) # Hide x-ticks

# Bottom Middle: Acceleration (Rate of change of Velocity)
ax4 = fig.add_subplot(gs[1, 1], sharex=ax1)
ax4.plot(t, xdot[:, 1], color='#9b2226', linestyle='-', label='$\dot{x}_2$ (Accel)')
ax4.set_xlabel('Time (s)')
ax4.legend(loc='upper right', frameon=True, fontsize=8)


# --- COLUMN 3: PHASE PORTRAIT ---
ax5 = fig.add_subplot(gs[:, 2])
ax5.plot(x[:, 0], x[:, 1], color='#333333', lw=1.5, alpha=0.8)

# Markers
ax5.scatter(x[0, 0], x[0, 1], color='#38b000', s=80, label='Start', zorder=5, edgecolors='white')
ax5.scatter(x[-1, 0], x[-1, 1], color='#d00000', s=80, label='End', zorder=5, edgecolors='white')

# Quiver Arrows (Visualizing the flow field)
arrow_int = int(len(t) / 25) 
ax5.quiver(x[::arrow_int, 0], x[::arrow_int, 1], 
           xdot[::arrow_int, 0], xdot[::arrow_int, 1],
           color='gray', alpha=0.5, width=0.005)

ax5.set_title('Phase Space Trajectory', fontweight='bold')
ax5.set_xlabel('State $x_1$')
ax5.set_ylabel('State $x_2$')
ax5.axis('equal') 
ax5.legend(loc='lower right')
# Save as a vector graphic with bounding box trimmed
plt.savefig('publication_figure.svg', format='svg', bbox_inches='tight', dpi=300)

# Optional: Save a high-res PNG version for quick sharing or presentations
# plt.savefig('publication_figure.png', format='png', bbox_inches='tight', dpi=600)

plt.show()

plt.show()