#Characterize the density functions
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom, poisson

# Plot PDF for normal distribution
x = np.linspace(-4, 4, 1000)
pdf = norm.pdf(x, loc=0, scale=1)

plt.plot(x, pdf, label='Normal PDF')
plt.title('Probability Density Function (Normal)')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()
