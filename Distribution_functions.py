# Distribution functions
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson

# Normal Distribution
x = np.linspace(-4, 4, 1000)
pdf = norm.pdf(x, loc=0, scale=1)
cdf = norm.cdf(x, loc=0, scale=1)

plt.plot(x, pdf, label='Normal PDF')
plt.plot(x, cdf, label='Normal CDF')
plt.title('PDF and CDF (Normal)')
plt.xlabel('x')
plt.ylabel('Density / Cumulative Probability')
plt.legend()
plt.show()
