# ARIMA example
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARMA
from random import random
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, \
    mean_squared_log_error

import pandas as pd


def evaluate_forecast(y, pred):
    results = pd.DataFrame({'r2_score': r2_score(y, pred),
                            }, index=[0])
    results['mean_absolute_error'] = mean_absolute_error(y, pred)
    results['median_absolute_error'] = median_absolute_error(y, pred)
    results['mse'] = mean_squared_error(y, pred)
    results['msle'] = mean_squared_log_error(y, pred)
    results['mape'] = mean_absolute_percentage_error(y, pred)
    results['rmse'] = np.sqrt(results['mse'])
    return results


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Import data
df = pd.read_csv('../../data/datasets/international-airline-passengers.csv', header=None)
df['year'] = pd.to_datetime(df['year'], format='%Y-%m')
y = df.set_index('year')
ts_log = np.log(y)
ts_log_diff = ts_log.passengers - ts_log.passengers.shift()
ts = y.passengers - y.passengers.shift()
ts.dropna(inplace=True)

# ACF and PACF plots after differencing:
pyplot.figure()
pyplot.subplot(211)
plot_acf(ts, ax=pyplot.gca(), lags=30)
pyplot.subplot(212)
plot_pacf(ts, ax=pyplot.gca(), lags=30)
pyplot.show()

# Interpreting ACF plots
# divide into train and validation set
train = y[:int(0.75 * (len(y)))]
valid = y[int(0.75 * (len(y))):]

# plotting the data
train['passengers'].plot()
valid['passengers'].plot()
# ==============================================================================
# Arima
# ==============================================================================
# fit model
model = ARIMA(train, order=(1, 1, 1))
model_fit = model.fit(disp=1)
model_fit.summary()

start_index = valid.index.min()
end_index = valid.index.max()
# Predictions
predictions = model_fit.predict(start=start_index, end=end_index)

# report performance
mse = mean_squared_error(y[start_index:end_index], predictions)
rmse = sqrt(mse)
print('RMSE: {}, MSE:{}'.format(rmse, mse))

plt.plot(y.passengers)
plt.plot(predictions, color='red')
plt.title('RMSE: %.4f' % rmse)
plt.show()

# Fitted or predicted values:
predictions_ARIMA_diff = pd.Series(predictions, copy=True)
print(predictions_ARIMA_diff.head())

# Cumulative Sum to reverse differencing:
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())

# Adding 1st month value which was previously removed while differencing:
predictions_ARIMA_log = pd.Series(valid.passengers.iloc[0], index=valid.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA_log.head()

# Taking Exponent to reverse Log Transform:
plt.plot(y.passengers)
plt.plot(predictions_ARIMA_log)
plt.title('RMSE: %.4f' % np.sqrt(np.nansum((predictions_ARIMA_log - ts) ** 2) / len(ts)))

evaluate_forecast(y[start_index:end_index], predictions_ARIMA_log)
