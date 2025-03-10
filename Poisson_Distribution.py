# Poisson Distribution
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson

# Generate synthetic data for feature space
np.random.seed(0)
feature_1 = np.random.normal(5, 1.5, 1000)  # Feature 1 with normal distribution (mean=5, std_dev=1.5)
feature_2 = np.random.poisson(3, 1000)     # Feature 2 with Poisson distribution (lambda=3)

# Plot setup
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Statistical Modeling of Feature Space", fontsize=16)

# Model and plot Feature 1 with a Gaussian distribution
mean_1, std_dev_1 = np.mean(feature_1), np.std(feature_1)
x1 = np.linspace(min(feature_1) - 1, max(feature_1) + 1, 100)
y1 = norm.pdf(x1, mean_1, std_dev_1)
axs[0].hist(feature_1, bins=30, density=True, alpha=0.6, color='skyblue')
axs[0].plot(x1, y1, 'r-', lw=2, label=f"Normal fit (μ={mean_1:.2f}, σ={std_dev_1:.2f})")
axs[0].set_title("Feature 1 - Normal Distribution")
axs[0].set_xlabel("Feature 1 Value")
axs[0].set_ylabel("Density")
axs[0].legend()

# Model and plot Feature 2 with a Poisson distribution
mu_2 = np.mean(feature_2)
x2 = np.arange(0, max(feature_2) + 1)
y2 = poisson.pmf(x2, mu_2)
axs[1].hist(feature_2, bins=range(0, max(feature_2) + 1), density=True, alpha=0.6, color='lightgreen', align='left')
axs[1].stem(x2, y2, 'r-', basefmt=" ", linefmt='r-', markerfmt='ro', label=f"Poisson fit (λ={mu_2:.2f})")
axs[1].set_title("Feature 2 - Poisson Distribution")
axs[1].set_xlabel("Feature 2 Value")
axs[1].set_ylabel("Probability Mass")
axs[1].legend()

plt.tight_layout()
plt.show()
