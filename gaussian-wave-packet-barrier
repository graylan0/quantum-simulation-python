import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
hbar = 1.0
mass = 1.0

# Initial wave packet parameters
x0 = -5.0
k0 = 2.0
sigma = 1.0

# Simulation domain and time steps
x = np.linspace(-10, 10, 800)
dx = x[1] - x[0]
dt = 0.005
time_steps = 300

# Potential barrier parameters
barrier_width = 1.0
barrier_height = 1.0

def gaussian_wave_packet(x, x0, k0, sigma):
    normalization = (1 / (sigma * np.sqrt(np.pi)))**(1/2)
    wave_packet = normalization * np.exp(1j * k0 * (x - x0) - ((x - x0)**2) / (2 * sigma**2))
    return wave_packet

def potential_barrier(x, barrier_width, barrier_height):
    potential = np.zeros_like(x)
    potential[np.abs(x) < barrier_width / 2] = barrier_height
    return potential

def time_evolution_operator(x, dt, hbar=1.0, mass=1.0, potential=None):
    kinetic_operator = np.exp(-1j * (hbar**2 / (2 * mass)) * (np.fft.fftfreq(len(x), x[1] - x[0]) * 2 * np.pi)**2 * dt)
    potential_operator = np.exp(-1j * potential * dt / hbar)
    return potential_operator, kinetic_operator

def evolve_wave_function(wave_function, potential_operator, kinetic_operator):
    wave_function = potential_operator * wave_function
    wave_function = np.fft.ifft(kinetic_operator * np.fft.fft(wave_function))
    wave_function = potential_operator * wave_function
    return wave_function

if __name__ == "__main__":
    psi = gaussian_wave_packet(x, x0, k0, sigma)
    potential = potential_barrier(x, barrier_width, barrier_height)
    potential_operator, kinetic_operator = time_evolution_operator(x, dt, potential=potential)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
    line1, = ax1.plot(x, np.abs(psi)**2, lw=2)
    ax1.set_ylabel("Probability Density")
    ax1.set_ylim(0, 0.5)
    line2, = ax2.plot(x, potential, lw=2)
    ax2.set_ylim(0, barrier_height * 1.1)
    ax2.set_xlabel("Position")
    ax2.set_ylabel("Potential")

    def update(frame):
        global psi
        psi = evolve_wave_function(psi, potential_operator, kinetic_operator)
        line1.set_ydata(np.abs(psi)**2)
        return line1,

    ani = FuncAnimation(fig, update, frames=time_steps, interval=20, blit=True)
    plt.show()
