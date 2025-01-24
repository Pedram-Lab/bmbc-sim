import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

# Simulation parameters
NUM_VESICLES = 15  # Number of vesicles
BOX_SIZE = 10  # Size of the cubic box
INITIAL_RADIUS = 1  # Initial radius of vesicles
KAPPA = 1.0  # Bending rigidity
L0 = 1.5  # Maximum link length

def generate_sphere(radius, num_points=50):
    """
    Function to generate a sphere using ConvexHull
    """
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    costheta = np.random.uniform(-1, 1, num_points)
    theta = np.arccos(costheta)
    r = radius

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    points = np.vstack((x, y, z)).T
    hull = ConvexHull(points)
    return points, hull

def curvature_energy(points, hull, kappa):
    """
    Function to calculate the curvature energy of a vesicle
    """
    # Extract all simplices (triangles)
    simplices = hull.simplices
    vertices = points[simplices]  # Shape: (n_simplices, 3, 3)

    # Compute vectors for the edges of each triangle
    edge1 = vertices[:, 1] - vertices[:, 0]  # Vector from vertex 0 to vertex 1
    edge2 = vertices[:, 2] - vertices[:, 0]  # Vector from vertex 0 to vertex 2

    # Compute normals for each triangle
    normals = np.cross(edge1, edge2)  # Shape: (n_simplices, 3)

    # Compute the area of each triangle
    areas = 0.5 * np.linalg.norm(normals, axis=1)  # Shape: (n_simplices,)

    # Normalize normals (to get unit vectors)
    normalized_normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    # Approximate mean curvature as the norm of the mean normal
    mean_curvatures = np.linalg.norm(np.mean(normalized_normals, axis=0))  # Single value

    # Compute total curvature energy
    energy = kappa * np.sum((mean_curvatures**2) * areas)
    return energy

def penalty(points, hull, l0):
    """
    Tethering potential to prevent distances greater than l0
    """
    # Get the simplices (triangles) from the hull
    simplices = hull.simplices  # Shape: (n_simplices, 3)
    vertices = points[simplices]  # Shape: (n_simplices, 3, 3)

    # Calculate pairwise distances within each triangle
    edge1 = np.linalg.norm(vertices[:, 1] - vertices[:, 0], axis=1)
    edge2 = np.linalg.norm(vertices[:, 2] - vertices[:, 0], axis=1)
    edge3 = np.linalg.norm(vertices[:, 2] - vertices[:, 1], axis=1)

    # Combine all edges into a single array
    edges = np.hstack((edge1, edge2, edge3))  # Shape: (3 * n_simplices,)

    # Compute penalties for edges exceeding l0
    penalties = np.where(edges > l0, 1e6 * (edges - l0)**2, 0)

    # Return the total penalty
    return np.sum(penalties)

def minimize_vesicle_energy(vesicle, kappa, l0):
    """
    Function to minimize the total energy of a vesicle
    """
    points = vesicle["points"].copy()
    hull = vesicle["hull"]

    def energy_func(flat_points):
        reshaped_points = flat_points.reshape(-1, 3)
        curvature = curvature_energy(reshaped_points, hull, kappa)
        tethering = penalty(reshaped_points, hull, l0)  # Pass the hull here
        return curvature + tethering

    result = minimize(energy_func, points.flatten(), method='L-BFGS-B')
    vesicle["points"] = result.x.reshape(-1, 3)
    vesicle["hull"] = ConvexHull(vesicle["points"])
    return vesicle

def generate_initial_vesicles(num_vesicles, radius, box_size):
    """
    Generate initial vesicles
    """
    vesicles = []
    for _ in range(num_vesicles):
        center = np.random.uniform(-box_size / 2, box_size / 2, size=3)
        points, hull = generate_sphere(radius)
        vesicle = {
            "center": center,
            "points": points + center,
            "hull": hull
        }
        vesicles.append(vesicle)
    return vesicles

def plot_vesicles(vesicles):
    """
    Visualization of vesicles
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    for vesicle in vesicles:
        points = vesicle["points"]
        hull = vesicle["hull"]
        
        # Draw the mesh lines
        for simplex in hull.simplices:
            line = points[simplex]
            ax.plot(line[:, 0], line[:, 1], line[:, 2], color='black', linewidth=0.5)
        
        # Optionally: Draw the node points
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='red', s=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-BOX_SIZE / 2, BOX_SIZE / 2])
    ax.set_ylim([-BOX_SIZE / 2, BOX_SIZE / 2])
    ax.set_zlim([-BOX_SIZE / 2, BOX_SIZE / 2])
    plt.show()

# Main flow
vesicles = generate_initial_vesicles(NUM_VESICLES, INITIAL_RADIUS, BOX_SIZE)

# Minimize energy for each vesicle
for i, vesicle in enumerate(vesicles):
    print(f"Minimizing energy of vesicle {i + 1}...")
    vesicle = minimize_vesicle_energy(vesicle, KAPPA, L0)

plot_vesicles(vesicles)
