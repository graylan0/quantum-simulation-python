import numpy as np
import matplotlib.pyplot as plt

def create_3d_object():
    x = np.random.random(100)
    y = np.random.random(100)
    z = np.random.random(100)
    return np.array([x, y, z])

def project_onto_2d_plane(points_3d, projection_plane_normal=[0, 0, 1]):
    projection_matrix = np.identity(3) - np.outer(projection_plane_normal, projection_plane_normal)
    points_2d = np.dot(projection_matrix, points_3d)
    return points_2d[:2]

def plot_points(points_2d):
    plt.scatter(points_2d[0], points_2d[1], s=30, alpha=0.7)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("2D Projection of 3D Object")
    plt.show()

if __name__ == "__main__":
    object_3d = create_3d_object()
    object_2d = project_onto_2d_plane(object_3d)
    plot_points(object_2d)
