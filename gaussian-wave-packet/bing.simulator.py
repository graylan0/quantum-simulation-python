import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite

# Constants
hbar = 1.0
mass = 1.0
omega = 1.0

# Simulation domain
x = np.linspace(-5, 5, 1000)

def wave_function(x, n, omega):
    normalization = 1 / np.sqrt(2**n * np.math.factorial(n)) * (mass * omega / (np.pi * hbar))**(1/4)
    psi_n = normalization * hermite(n)(x) * np.exp(-mass * omega * x**2 / (2 * hbar))
    return psi_n

def energy_level(n, omega, hbar=1.0):
    return hbar * omega * (n + 0.5)

def plot_wave_functions(x, omega, num_states=4):
    plt.figure(figsize=(8, 6))

    for n in range(num_states):
        psi_n = wave_function(x, n, omega)
        energy_n = energy_level(n, omega)
        
        plt.plot(x, psi_n + energy_n, label=f"n = {n}")
        plt.fill_between(x, psi_n + energy_n, energy_n, alpha=0.1)
        plt.hlines(energy_n, x[0], x[-1], colors='gray', linewidth=0.5)

    plt.xlabel("Position")
    plt.ylabel("Energy / Wave Functions")
    plt.title("Harmonic Oscillator")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_wave_functions(x, omega)
