# changes of distribution as changes in parameters (mean vector, covariance matrix)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_gaussian(mean, cov, ax, label=None, color='blue'):
    """
    Plots a 2D Gaussian distribution given a mean and covariance matrix.
    """
    # Draw the ellipse representing the covariance matrix
    eigvals, eigvecs = np.linalg.eig(cov)
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(eigvals)  # scale eigenvalues for visualization

    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      edgecolor=color, fc='None', lw=2, label=label)
    ax.add_patch(ellipse)
    ax.scatter(*mean, color=color, marker='x', s=100)  # plot mean
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 8))

# Define different mean vectors and covariance matrices
means = [
    [3, 2],
    [3, 3],
    [-2, -5]
]

covariances = [
    [[3, 4], [0, 1]],        # circular distribution
    [[2, 1], [1, 2]],        # elliptical distribution, rotated
    [[1, 0.8], [0.8, 1]],    # elliptical distribution with strong correlation
]

# Colors for each distribution
colors = ['blue', 'green', 'red']

# Plot each distribution
for i, (mean, cov) in enumerate(zip(means, covariances)):
    plot_gaussian(mean, cov, ax, label=f"Mean: {mean}, Covariance: {cov}", color=colors[i])

# Plot settings
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
plt.title("2D Gaussian Distributions with Different Mean Vectors and Covariance Matrices")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid(True)
plt.show()
