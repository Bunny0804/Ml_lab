#Kernel methods
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Generate synthetic dataset
X, y = datasets.make_moons(n_samples=100, noise=0.1, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a function to plot decision boundaries
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.grid()
    plt.show()

# Train SVM with different kernels
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
test_accuracy = {}

for kernel in kernels:
    model = SVC(kernel=kernel, C=1.0, gamma='scale')
    model.fit(X_train, y_train)
    
    # Plot the decision boundary
    plot_decision_boundary(model, X_train, y_train, f'SVM with {kernel} kernel')
    
    # Store test accuracy
    test_accuracy[kernel] = model.score(X_test, y_test)

# Test Accuracy for each kernel
print("Test Accuracy for each kernel:")
for kernel, accuracy in test_accuracy.items():
    print(f"{kernel.capitalize()} kernel: {accuracy:.2f}")
