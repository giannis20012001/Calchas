import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Question part a
url = "http://research.stlouisfed.org/fred2/series/UNRATE/downloaddata/UNRATE.csv"
df = pd.read_csv(url)

# ======================================================================================================================
# Question part b
df['DATE'] = pd.to_datetime(df.DATE)
df = df.set_index('DATE')
yearly_averages = df.groupby(pd.Grouper(freq='1Y')).mean()
yearly_averages.hist(bins=10, grid=True, figsize=(12, 8), color='#86bf91', zorder=2, rwidth=0.9)
plt.show()

# ======================================================================================================================
# Question part c
plt.rcParams['figure.figsize'] = (12.0, 9.0)
X = yearly_averages.iloc[:, 0]
Y = pd.DatetimeIndex(yearly_averages.index).year

# Building the model
X_mean = np.mean(X)
Y_mean = np.mean(Y)

num = 0
den = 0
for i in range(len(X)):
    num += (X[i] - X_mean)*(Y[i] - Y_mean)
    den += (X[i] - X_mean)**2
m = num / den
c = Y_mean - m*X_mean

print (m, c)

# Making predictions
Y_pred = m*X + c

plt.scatter(X, Y)  # actual values
# plt.scatter(X, Y_pred, color='red')
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  # predicted values
plt.show()
