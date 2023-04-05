import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1.0
mass = 1.0
well_width = 1.0

# Simulation domain
x = np.linspace(0, well_width, 1000)

def wave_function(x, n, well_width):
    normalization = np.sqrt(2 / well_width)
    psi_n = normalization * np.sin(n * np.pi * x / well_width)
    return psi_n

def energy_level(n, well_width, hbar=1.0, mass=1.0):
    return (n**2 * np.pi**2 * hbar**2) / (2 * mass * well_width**2)

def plot_wave_functions(x, well_width, num_states=4):
    plt.figure(figsize=(8, 6))

    for n in range(1, num_states + 1):
        psi_n = wave_function(x, n, well_width)
        energy_n = energy_level(n, well_width)
        
        plt.plot(x, psi_n + energy_n, label=f"n = {n}")
        plt.fill_between(x, psi_n + energy_n, energy_n, alpha=0.1)
        plt.hlines(energy_n, 0, well_width, colors='gray', linewidth=0.5)

    plt.xlabel("Position")
    plt.ylabel("Energy / Wave Functions")
    plt.title("Infinite Square Well")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_wave_functions(x, well_width)
