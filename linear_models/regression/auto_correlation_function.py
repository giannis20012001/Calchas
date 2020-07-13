from pandas import read_csv
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf

# Load the dataset
series = read_csv('../../datasets/daily-minimum-temperatures.csv', header=0, index_col=0)
series.plot()
pyplot.show()

# Calculate and plot the autocorrelation function
plot_acf(series)
pyplot.show()