import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

# Simulation parameters
num_vesicles = 1  # Number of vesicles
box_size = 10  # Size of the cubic simulation box
initial_radius = 1  # Initial radius of the vesicles
kappa = 1.0  # Parameter controlling curvature stiffness
l0 = 1.5    # Maximum allowable distance between points before adding an energy penalty.

# Function to generate a triangulated sphere
# Generate points forming a 3D sphere and compute its triangulation
def generate_sphere(radius, num_points=500):
    # Generate points in spherical coordinates
    # phi, costheta, and u are randomly generated to distribute points on a sphere
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    costheta = np.random.uniform(-1, 1, num_points)
    u = np.random.uniform(0, 1, num_points)

    theta = np.arccos(costheta)
    r = radius * u**(1/3)

    # Transformation to Cartesian coordinates uses standard sphere formulas
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    # Delaunay Triangulation:
    # Creates a triangular mesh over the sphere, useful for calculating areas and curvatures
    points = np.vstack((x, y, z)).T
    tri = Delaunay(points)
    # Returns the points (points) and their triangulation (tri)
    return points, tri

# Function to calculate the bending energy of a vesicle
# Calculate the curvature energy of a vesicle based on stiffness kappa
# Iterates over the triangles in the mesh (tri.simplices) and computes:
# 1) Area of each triangle: Using the cross product
# 2) Mean curvature: Approximated as the average normal magnitude
# 3) Local energy: Scaled by the area and squared mean curvature
def curvature_energy(points, tri, kappa):
    energy = 0
    for simplex in tri.simplices:
        vertices = points[simplex]
        area = 0.5 * np.linalg.norm(np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0]))

        # Local mean curvature approximation
        normals = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        mean_curvature = np.linalg.norm(np.mean(normals, axis=0))

        energy += kappa * (mean_curvature**2) * area
    return energy

# Tethering potential to avoid distances greater than l_0
# Penalize distances greater than l0 between points
# Uses a double loop to calculate pairwise distances between points
# Adds a quadratic penalty if the distance exceeds l_0
def tethering_potential(points, l0):
    potential = 0
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if i < j:
                distance = np.linalg.norm(p1 - p2)
                if distance > l0:
                    potential += 1e6 * (distance - l0)**2
    return potential

# Function to minimize the total energy of a vesicle
# Minimize the total energy of a vesicle by combining curvature and tethering penalties
# Defines an energy_func that computes total energy

def minimize_vesicle_energy(vesicle, kappa, l0, iterations=200):
    points = vesicle["points"].copy()
    tri = vesicle["tri"]

    # The total energy is calculated within the energy_func function, which is used during the minimization process
    # This function combines the curvature energy (curvature_energy) and the tethering potential (tethering_potential) to compute the total energy of the system
    # It is called by the minimize function within minimize_vesicle_energy, where the goal is to find the configuration of points that minimizes this total energy
    def energy_func(flat_points):
        reshaped_points = flat_points.reshape(-1, 3)
        curvature = curvature_energy(reshaped_points, tri, kappa)
        tethering = tethering_potential(reshaped_points, l0)
        return curvature + tethering
    
    # Minimization process
    # The L-BFGS-B method is used to minimize the total energy function, updating the positions of the points (vesicle["points"]) for each vesicle
    # This is a specific optimization algorithm designed for solving unconstrained or bound-constrained optimization problems
    # This algorithm allows you to specify upper and lower bounds for the variables being optimized, ensuring that the solution stays within a desired range.
    result = minimize(energy_func, points.flatten(), method='L-BFGS-B')
    vesicle["points"] = result.x.reshape(-1, 3)
    return vesicle

# Generate initial vesicles
def generate_initial_vesicles(num_vesicles, radius, box_size):
    vesicles = []
    for _ in range(num_vesicles):
        center = np.random.uniform(-box_size / 2, box_size / 2, size=3)
        points, tri = generate_sphere(radius)
        vesicle = {
            "center": center,
            "points": points + center,
            "tri": tri
        }
        vesicles.append(vesicle)
    return vesicles

# Visualization of vesicles
def plot_vesicles(vesicles):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    for vesicle in vesicles:
        points = vesicle["points"]
        tri = vesicle["tri"]
        for simplex in tri.simplices:
            triangle = points[simplex]
            ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], alpha=0.2, color='blue')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-box_size / 2, box_size / 2])
    ax.set_ylim([-box_size / 2, box_size / 2])
    ax.set_zlim([-box_size / 2, box_size / 2])
    plt.show()

# Main workflow
vesicles = generate_initial_vesicles(num_vesicles, initial_radius, box_size)

# Minimize energy for each vesicle
for i, vesicle in enumerate(vesicles):
    print(f"Minimizando energía de vesícula {i + 1}...")
    vesicle = minimize_vesicle_energy(vesicle, kappa, l0)

plot_vesicles(vesicles)
