import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set plot appearance
plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'green'
plt.rcParams['axes.labelcolor'] = 'green'
plt.rcParams['xtick.color'] = 'green'
plt.rcParams['ytick.color'] = 'green'

# Constants
hbar = 1.0
mass = 1.0
omega = 1.0

# Initial wave packet parameters
x0 = 0.0
k0 = 5.0
sigma = 1.0

# Simulation domain and time steps
x = np.linspace(-10, 10, 400)
dt = 0.01
time_steps = 300

def gaussian_wave_packet(x, x0, k0, sigma):
    normalization = (1/(sigma*np.sqrt(np.pi)))**(1/2)
    wave_packet = normalization * np.exp(1j*k0*(x-x0) - ((x-x0)**2)/(2*sigma**2))
    return wave_packet

def time_evolution_operator(x, dt, hbar=1.0, mass=1.0, omega=1.0):
    kinetic_operator = np.exp(-1j*(hbar**2/(2*mass))*(np.fft.fftfreq(len(x), x[1]-x[0])*2*np.pi)**2*dt)
    potential_operator = np.exp(-1j*0.5*mass*omega**2*(x**2)*dt)
    return potential_operator, kinetic_operator

def evolve_wave_function(wave_function, potential_operator, kinetic_operator):
    wave_function = potential_operator * wave_function
    wave_function = np.fft.ifft(kinetic_operator * np.fft.fft(wave_function))
    wave_function = potential_operator * wave_function
    return wave_function

if __name__ == "__main__":
    psi = gaussian_wave_packet(x, x0, k0, sigma)
    potential_operator, kinetic_operator = time_evolution_operator(x, dt)

    fig, ax = plt.subplots()
    line, = ax.plot(x, np.abs(psi)**2, lw=2)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel('Position')
    ax.set_ylabel('Probability Density')

    def update(frame):
        global psi
        psi = evolve_wave_function(psi, potential_operator, kinetic_operator)
        line.set_ydata(np.abs(psi)**2)
        return line,

    ani = FuncAnimation(fig, update, frames=time_steps, interval=20, blit=True)
    plt.show()
