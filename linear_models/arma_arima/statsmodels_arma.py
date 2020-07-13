# ARMA example
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARMA
from random import random

import pandas as pd
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Import data
df = pd.read_csv('../../datasets/international-airline-passengers.csv',header=None)
df['year'] = pd.to_datetime(df['year'], format='%Y-%m')
y = df.set_index('year')
ts_log = np.log(y)
ts_log_diff = ts_log.passengers - ts_log.passengers.shift()

# Fit model
model = ARMA(ts_log_diff, order=(2, 1))
model_fit = model.fit(disp=False)
model_fit.summary()

# Plot
plt.plot(ts_log_diff)
plt.plot(model_fit.fittedvalues, color='red')
plt.title('RSS: %.4f'% np.nansum((model_fit.fittedvalues-ts_log_diff)**2))