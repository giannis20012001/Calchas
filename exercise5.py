import numpy as np
import matplotlib.pyplot as plt

mu = 10500
sigma = 250
values = mu + sigma * np.random.randn(10000)
x = mu + sigma * np.random.randn(100)
y = mu + sigma * np.random.randn(100)

fig, axs = plt.subplots(ncols=2)
fig.suptitle('Vertically stacked subplots')
axs[0].hist(x, bins=25, edgecolor=None, facecolor="green")
axs[0].set_title('Histogram')
axs[0].set_ylabel('count of values')
axs[0].set_xlabel('Data')
axs[1].scatter(x, y, s=300,  edgecolor=None, c=np.random.rand(100,), alpha=0.25)
axs[1].set_title('Histogram')
axs[1].set_ylabel('y')
axs[1].set_xlabel('x')
axs[1].set_xlim(10000, 11000)
axs[1].set_ylim(10000, 11000)
axs[1].grid()
plt.tight_layout()
plt.show()
