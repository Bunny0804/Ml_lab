# Max likelihood estimator
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class MLEClassifier:
    def __init__(self):
        self.means = None
        self.covs = None
        self.priors = None

    def fit(self, X, y):
        classes = np.unique(y)
        self.means = []
        self.covs = []
        self.priors = []

        for cls in classes:
            # Select the data points for the current class
            X_cls = X[y == cls]
            # Calculate MLE for mean and covariance
            self.means.append(np.mean(X_cls, axis=0))
            self.covs.append(np.cov(X_cls, rowvar=False))
            self.priors.append(len(X_cls) / len(X))

        self.means = np.array(self.means)
        self.covs = np.array(self.covs)
        self.priors = np.array(self.priors)

    def predict(self, X):
        # Calculate likelihood for each class
        likelihoods = []
        for mean, cov, prior in zip(self.means, self.covs, self.priors):
            likelihood = multivariate_normal.pdf(X, mean=mean, cov=cov)
            posterior = likelihood * prior
            likelihoods.append(posterior)

        # Classify based on the maximum posterior probability
        return np.argmax(likelihoods, axis=0)

# Generate synthetic data
def generate_data(mean_1, cov_1, mean_2, cov_2, n_samples=100):
    class_1 = np.random.multivariate_normal(mean_1, cov_1, n_samples)
    class_2 = np.random.multivariate_normal(mean_2, cov_2, n_samples)
    X = np.vstack((class_1, class_2))
    y = np.array([0] * n_samples + [1] * n_samples) # Labels
    return X, y

# Visualize decision boundaries
def visualize_decision_boundary(classifier, class_1, class_2):
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
    plt.title("Maximum Likelihood Estimation Classifier with Decision Boundary")
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
X, y = generate_data(mean_1, cov_1, mean_2, cov_2, n_samples=100)

# Create a MLE classifier and fit it to the data
classifier = MLEClassifier()
classifier.fit(X, y)

# Visualize decision boundary
visualize_decision_boundary(classifier, X[y == 0], X[y == 1])
