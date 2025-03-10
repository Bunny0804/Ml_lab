#Bayesian classifier
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class BayesianClassifier:
    def __init__(self, mean_1, cov_1, mean_2, cov_2, prior_1=0.5, prior_2=0.5):
        self.mean_1 = mean_1
        self.cov_1 = cov_1
        self.mean_2 = mean_2
        self.cov_2 = cov_2
        self.prior_1 = prior_1
        self.prior_2 = prior_2

    def predict(self, X):
        # Calculate the likelihood for both classes
        likelihood_1 = multivariate_normal.pdf(X, mean=self.mean_1, cov=self.cov_1)
        likelihood_2 = multivariate_normal.pdf(X, mean=self.mean_2, cov=self.cov_2)
        # Calculate posterior probabilities
        posterior_1 = likelihood_1 * self.prior_1
        posterior_2 = likelihood_2 * self.prior_2
        # Return the class with the highest posterior probability
        return np.where(posterior_1 > posterior_2, 1, 0)

# Generate synthetic data
def generate_data(mean_1, cov_1, mean_2, cov_2, n_samples=100):
    class_1 = np.random.multivariate_normal(mean_1, cov_1, n_samples)
    class_2 = np.random.multivariate_normal(mean_2, cov_2, n_samples)
    return class_1, class_2

# Visualize decision boundaries
def visualize_decision_boundary(class_1, class_2, classifier):
    plt.figure(figsize=(10, 6))
    # Create a grid to plot the decision boundary
    x_min, x_max = -3, 7
    y_min, y_max = -3, 7
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    # Predict class for each point in the grid
    Z = classifier.predict(grid_points)
    Z = Z.reshape(xx.shape)
    # Plot decision boundary and data points
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(class_1[:, 0], class_1[:, 1], color='blue', label='Class 1', edgecolor='k')
    plt.scatter(class_2[:, 0], class_2[:, 1], color='red', label='Class 2', edgecolor='k')
    plt.title("Bayesian Classification with Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid()
    plt.show()

# Main execution
mean_1 = [2, 2]
cov_1 = [[1, 0.5], [0.5, 1]] # Class 1 parameters
mean_2 = [5, 5]
cov_2 = [[1, -0.5], [-0.5, 1]] # Class 2 parameters

# Generate data
class_1, class_2 = generate_data(mean_1, cov_1, mean_2, cov_2, n_samples=100)

# Create a Bayesian classifier
classifier = BayesianClassifier(mean_1, cov_1, mean_2, cov_2)

# Visualize decision boundary
visualize_decision_boundary(class_1, class_2, classifier)
