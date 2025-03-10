# Covariance Matrix
import numpy as np
import matplotlib.pyplot as plt

# Generate random data for two features
np.random.seed(42)
feature_1 = np.random.normal(5, 1.5, 100)  # Mean = 5, Std Dev = 1.5
feature_2 = 2 * feature_1 + np.random.normal(0, 2, 100)  # Linearly related with noise

# Combine features into a dataset
data = np.vstack((feature_1, feature_2)).T

# Estimate covariance matrix
cov_matrix = np.cov(data, rowvar=False)

# Properties of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
trace_cov = np.trace(cov_matrix)
determinant_cov = np.linalg.det(cov_matrix)

# Print the covariance matrix and its properties
print("Covariance Matrix:")
print(cov_matrix)
print("\nEigenvalues of the Covariance Matrix:")
print(eigenvalues)
print("\nEigenvectors of the Covariance Matrix:")
print(eigenvectors)
print(f"\nTrace of Covariance Matrix: {trace_cov}")
print(f"Determinant of Covariance Matrix: {determinant_cov}")

# Plot the data and the principal components
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], alpha=0.5, label="Data Points")

# Plot the eigenvectors as principal components
origin = np.mean(data, axis=0)  # Plot eigenvectors from the mean of the data
for i in range(len(eigenvalues)):
    plt.quiver(*origin, *eigenvectors[:, i] * np.sqrt(eigenvalues[i]),
               scale=5, scale_units='xy', angles='xy', color=['r', 'g'][i],
               label=f"PC{i+1} (Eigenvalue: {eigenvalues[i]:.2f})")

# Label plot
plt.title("Data with Covariance Matrix Principal Components")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
