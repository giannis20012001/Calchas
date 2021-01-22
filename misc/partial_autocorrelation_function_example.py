from pandas import read_csv
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_pacf

# Load the dataset
series = read_csv('../data/datasets/test_datasets/daily-min-temperatures.csv', header=0, index_col=0)
series.plot()
pyplot.show()

# Calculate and plot the partial autocorrelation function
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0)
plot_pacf(series, lags=50)
pyplot.show()