import warnings
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


# load dataset
series = read_csv('dataset.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
X = series.values
X = X.astype('float32')
# predict
warnings.filterwarnings("ignore")
model_fit = ARIMA(X, order=(2, 0, 2))
results = model_fit.fit()
yhat = results.predict()[0]

print('>Predicted=%.3f' % (yhat))
