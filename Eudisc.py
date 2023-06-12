import numpy as np
import matplotlib.pyplot as plt

# Parameters and constants
pi = np.pi
G = 6.6743e-8  # gravitational constant
day = 8.6400E4  # seconds per day
year = 365.25 * day  # seconds per year
t_final = 1.0 * day  # simulation time
n_points = 501  # number of points
R_min = 1e8  # minimum radius
R_max = 1e12  # maximum radius
sigma_max = 1E5  # maximum surface density
M_sun = 1.989e33  # mass of the Sun
mass = 1.4 * M_sun  # mass of the object
R_Peak = 20 * R_min  # peak radius
width = 0.2 * R_Peak  # width of the peak
VK = 441.34  # some constant related to velocity
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
V = (VK * X ** a1) * (S ** q)
VS = V * S

dt = 0.4 * dX ** 2 * np.min(X / (V + 1e-50))  # time step

# Initialize time and step variables
t = 0.0
nt = 0

# For sigma plotting
n_points_sigma = 10
t_plot_sigma = np.logspace(np.log10(10), np.log10(t_final), n_points_sigma)
sigma_plot = np.zeros((n_points_sigma, len(X)))

# For M_dot plotting
n_points_mdot = 400
t_plot_mdot = np.logspace(np.log10(10), np.log10(t_final), n_points_mdot)
M_dot_plot = np.zeros(n_points_mdot)
time_plot = np.zeros(n_points_mdot)

index_sigma = 0
index_mdot = 0

# Time evolution loop
while t < t_final:
    V[1:-1] = (VK * X[1:-1] ** a1) * (S[1:-1] ** q)
    VS[1:-1] = V[1:-1] * S[1:-1]
    S[1:-1] += (0.75 * dt / (dX * X[1:-1]) ** 2) * (VS[2:] + VS[:-2] - 2 * VS[1:-1])

    # Boundary conditions
    S[0] = S[-1] = 0
    VS[0] = VS[-1] = 0
    print(t)
    # Save sigma data for plotting
    if index_sigma < n_points_sigma and t >= t_plot_sigma[index_sigma]:
        sigma_plot[index_sigma, :] = S / X
        index_sigma += 1

    # Save M_dot data for plotting
    if index_mdot < n_points_mdot and t >= t_plot_mdot[index_mdot]:
        M_dot_plot[index_mdot] = 3 * pi * ((VS[1] - VS[0]) / dX)
        time_plot[index_mdot] = t
        index_mdot += 1

    nt += 1
    t += dt

# Plot the surface mass density
plt.figure(figsize=(10, 6))
plt.plot(R, sigma0, label="Initial condition")  # initial condition plot
for i in range(n_points_sigma):
    plt.plot(R, sigma_plot[i, :], label="t = {:.2e} * 10^3 s".format(t_plot_sigma[i]/1000))  # show time in x*10^3 s
plt.legend(loc='upper right')  # set legend location
plt.xscale("log")
plt.yscale("log")
plt.ylim(0.01, 2 * sigma_max)
plt.xlabel("Radius (cm)")
plt.ylabel("Surface mass density (g/cm^2)")
plt.show()

# Plot the mass accretion rate
plt.figure(figsize=(10, 6))
plt.plot(time_plot[time_plot >= 10], M_dot_plot[time_plot >= 10])  # start from 10 seconds
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Time (s)")
plt.ylabel("Mass accretion rate (g/s)")
plt.show()
