import numpy as np
import matplotlib.pyplot as plt

# Parameters and constants
pi = np.pi
G = 6.6743e-8  # gravitational constant
day = 8.6400E4  # seconds per day
year = 365.25 * day  # seconds per year
t_final = 0.1 * day  # simulation time
n_points = 501  # number of points
R_min = 1e8  # minimum radius
R_max = 1e12  # maximum radius
sigma_max = 1E5  # maximum surface density
M_sun = 1.989e33  # mass of the Sun
mass = 1.4 * M_sun  # mass of the object
R_Peak = 20 * R_min  # peak radius
width = 0.2 * R_Peak  # width of the peak
C = 441.34  # constant
q = 3 / 7  # constant q for calculations
p = 15 / 14  # constant p for calculations
a1 = 2 * p - q  # constant a1 for calculations

# Define range for X and R
X_min = np.sqrt(R_min)
X_max = np.sqrt(R_max)
dX = (X_max - X_min) / n_points
X = np.linspace(X_min, X_max, n_points + 1)
R = X ** 2

# Initial conditions for surface density
sigma0 = sigma_max * np.exp(-0.5 * ((R - R_Peak) / width) ** 2)
S = X * sigma0
V = (C * X ** a1) * (S ** q)
VS = V * S

dt = 0.4 * dX ** 2 * np.min(X / (V + 1e-50))  # time step made with Courant - Friedrichs- Lewy condition.

# Initialize time and step variables
t = 0.0
nt = 0

# For sigma plotting
n_points_sigma = 24
t_plot_sigma = np.logspace(np.log10(1), np.log10(t_final), n_points_sigma)
sigma_plot = np.zeros((n_points_sigma, len(X)))

# For M_dot plotting
n_points_mdot = 400
t_plot_mdot = np.logspace(np.log10(1), np.log10(t_final), n_points_mdot)
M_dot_plot = np.zeros(n_points_mdot)
time_plot = np.zeros(n_points_mdot)

# Time evolution loop
while t < t_final:
    V[1:-1] = (C * X[1:-1] ** a1) * (S[1:-1] ** q)
    VS[1:-1] = V[1:-1] * S[1:-1]
    S[1:-1] += (0.75 * dt / (dX * X[1:-1]) ** 2) * (VS[2:] + VS[:-2] - 2 * VS[1:-1])

    # Boundary conditions
    S[0] = S[-1] = 0
    VS[0] = VS[-1] = 0

    # Calculate accretion rate
    M_dot = 3 * pi * ((VS[1] - VS[0]) / dX)

    nt += 1
    t += dt

    sigma = S / X

    # Save sigma data for plotting
    for i in range(n_points_sigma):
        if abs(t - t_plot_sigma[i]) < 0.5 * dt:
            sigma_plot[i, :] = sigma

    # Save M_dot data for plotting
    for i in range(n_points_mdot):
        if abs(t - t_plot_mdot[i]) < 0.5 * dt:
            M_dot_plot[i] = M_dot
            time_plot[i] = t

# Plot the surface mass density
plt.figure(figsize=(10, 6))
for i in range(n_points_sigma):
    plt.plot(R, sigma_plot[i, :], label="t = {:.2e}".format(t_plot_sigma[i]))
plt.plot(R, sigma0, label="Initial condition")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.ylim(0.01, 2 * sigma_max)
plt.xlabel("Radius (cm)")
plt.ylabel("Surface mass density (g/cm^2)")
plt.show()

# Plot the mass accretion rate
plt.figure(figsize=(10, 6))
plt.plot(time_plot, M_dot_plot)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Time (s)")
plt.ylabel("Mass accretion rate (g/s)")
plt.show()
