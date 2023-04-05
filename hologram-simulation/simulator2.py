import numpy as np
import matplotlib.pyplot as plt

def create_2d_lattice(n_points_x, n_points_y):
    x_points = np.linspace(0, 1, n_points_x)
    y_points = np.linspace(0, 1, n_points_y)
    x_grid, y_grid = np.meshgrid(x_points, y_points)
    return x_grid, y_grid

def plot_lattice(x_grid, y_grid):
    plt.figure(figsize=(8, 8))
    for i in range(x_grid.shape[0]):
        plt.plot(x_grid[i, :], y_grid[i, :], 'k')
    for i in range(y_grid.shape[1]):
        plt.plot(x_grid[:, i], y_grid[:, i], 'k')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("2D Lattice representing a lower-dimensional spacetime")
    plt.show()

if __name__ == "__main__":
    x_lattice, y_lattice = create_2d_lattice(10, 10)
    plot_lattice(x_lattice, y_lattice)
