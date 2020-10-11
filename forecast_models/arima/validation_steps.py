import numpy as np
import scipy
import scipy.stats as stats
from math import sqrt
from pandas import read_csv
from matplotlib import pyplot
# Deprecated import: from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMAResults


def save_fitted_model(csv_file_name):
    # load data for 70% - 30%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_70_30.csv', header=None,
                      index_col=0, parse_dates=True, squeeze=True)
    print()
    print("==========================================================")
    print("For 70% - 30% we have...")
    print("Enter (p,  d, q) extracted by the conclusions made by stationarity and  ACF/PACF plots")
    p = input("Enter p value (Autoregression (AR) --> p): ")
    d = input("Enter d value (differencing --> d): ")
    q = input("Enter q value (Moving Average (MA) --> q): ")
    print('ARIMA(%d, %d, %d)' % (int(p), int(d), int(q)))
    # prepare data
    X = series.values
    X = X.astype('float32')
    # fit model
    model = ARIMA(X, order=(int(p), int(d), int(q)))
    model_fit = model.fit()
    # bias constant, could be calculated from in-sample mean residual
    # bias = 1.081624
    # save model
    print("Saving model...")
    model_fit.save('data/saved_models/' + csv_file_name.split('.csv')[0] + '_arima_70_30.pkl')
    # numpy.save('model_bias.npy', [bias])
    print("==========================================================")
    print()

    # ==================================================================================================================
    # load data for 80% - 20%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_80_20.csv', header=None,
                      index_col=0, parse_dates=True, squeeze=True)
    print()
    print("==========================================================")
    print("For 80% - 20% we have...")
    print("Enter (p,  d, q) extracted by the conclusions made by stationarity and  ACF/PACF plots")
    p = input("Enter p value (Autoregression (AR) --> p): ")
    d = input("Enter d value (differencing --> d): ")
    q = input("Enter q value (Moving Average (MA) --> q): ")
    print('ARIMA(%d, %d, %d)' % (int(p), int(d), int(q)))
    # prepare data
    X = series.values
    X = X.astype('float32')
    # fit model
    model = ARIMA(X, order=(int(p), int(d), int(q)))
    model_fit = model.fit()
    # bias constant, could be calculated from in-sample mean residual
    # bias = 1.081624
    # save model
    print("Saving model...")
    model_fit.save('data/saved_models/' + csv_file_name.split('.csv')[0] + '_arima_80_20.pkl')
    # numpy.save('model_bias.npy', [bias])
    print("==========================================================")
    print()

    # ==================================================================================================================
    # load data for 90% - 10%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_90_10.csv', header=None,
                      index_col=0, parse_dates=True, squeeze=True)
    print()
    print("==========================================================")
    print("For 90% - 10% we have...")
    print("Enter (p,  d, q) extracted by the conclusions made by stationarity and  ACF/PACF plots")
    p = input("Enter p value (Autoregression (AR) --> p): ")
    d = input("Enter d value (differencing --> d): ")
    q = input("Enter q value (Moving Average (MA) --> q): ")
    print('ARIMA(%d, %d, %d)' % (int(p), int(d), int(q)))
    # prepare data
    X = series.values
    X = X.astype('float32')
    # fit model
    model = ARIMA(X, order=(int(p), int(d), int(q)))
    model_fit = model.fit()
    # bias constant, could be calculated from in-sample mean residual
    # bias = 1.081624
    # save model
    print("Saving model...")
    model_fit.save('data/saved_models/' + csv_file_name.split('.csv')[0] + '_arima_90_10.pkl')
    # numpy.save('model_bias.npy', [bias])
    print("==========================================================")
    print()


def calculate_reliability_statistics(expected_vals, predictions):
    x = np.array(expected_vals)
    y = np.array(predictions)
    f = np.var(x, ddof=1) / np.var(y, ddof=1)  # calculate F test statistic
    dfn = x.size - 1  # define degrees of freedom numerator
    dfd = y.size - 1  # define degrees of freedom denominator
    alpha = 0.05  # Or whatever you want your alpha to be.
    p_value = 1 - scipy.stats.f.cdf(f, dfn, dfd)  # find p-value of F test statistic

    if p_value > alpha:
        print("Reject the null hypothesis that Var(X) == Var(Y)")
    else:
        print("Cannot reject the null hypothesis that Var(X) == Var(Y)")

    return f, p_value


def calculate_correlation_index():
    print()


def calculate_t_test():
    print()


def validate_arima_model(csv_file_name):
    # load data for 70% - 30%
    dataset = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_70_30.csv', header=None,
                       index_col=0, parse_dates=True, squeeze=True)
    X = dataset.values.astype('float32')
    history = [x for x in X]
    validation = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_validation_70_30.csv', header=None,
                          index_col=0, parse_dates=True, squeeze=True)
    y = validation.values.astype('float32')
    print()
    print("==========================================================")
    print("For 70% - 30% we have...")
    print("Enter (p,  d, q) extracted by the conclusions made by stationarity and  ACF/PACF plots")
    p = input("Enter p value (Autoregression (AR) --> p): ")
    d = input("Enter d value (differencing --> d): ")
    q = input("Enter q value (Moving Average (MA) --> q): ")
    print('ARIMA(%d, %d, %d)' % (int(p), int(d), int(q)))
    # load model
    print("Loading model...")
    model_fit = ARIMAResults.load('data/saved_models/' + csv_file_name.split('.csv')[0] + '_arima_70_30.pkl')
    # bias = numpy.load('model_bias.npy')
    # make first prediction
    print("Starting model evaluation...")
    predictions = list()
    # yhat = bias + float(model_fit.forecast()[0])
    yhat = float(model_fit.forecast()[0])
    predictions.append(yhat)
    history.append(y[0])
    print('>Predicted=%.3f, Expected=%.3f' % (yhat, y[0]))
    # rolling forecasts
    for i in range(1, len(y)):
        # predict
        model = ARIMA(history, order=(int(p), int(d), int(q)))
        model_fit = model.fit()
        # yhat = bias + float(model_fit.forecast()[0])
        yhat = float(model_fit.forecast()[0])
        predictions.append(yhat)
        # observation
        obs = y[i]
        history.append(obs)
        print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
    # report performance
    rmse = sqrt(mean_squared_error(y, predictions))
    print('RMSE: %.3f' % rmse)
    calculate_reliability_statistics(y, predictions)
    calculate_correlation_index()
    calculate_t_test()
    print("Model evaluation finished...")
    print("==========================================================")
    print()
    pyplot.plot(y)
    pyplot.plot(predictions, color='red')
    pyplot.show()

    # ==================================================================================================================
    # load data for 80% - 20%
    dataset = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_80_20.csv', header=None,
                       index_col=0, parse_dates=True, squeeze=True)
    X = dataset.values.astype('float32')
    history = [x for x in X]
    validation = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_validation_80_20.csv', header=None,
                          index_col=0, parse_dates=True, squeeze=True)
    y = validation.values.astype('float32')
    print()
    print("==========================================================")
    print("For 80% - 20% we have...")
    print("Enter (p,  d, q) extracted by the conclusions made by stationarity and  ACF/PACF plots")
    p = input("Enter p value (Autoregression (AR) --> p): ")
    d = input("Enter d value (differencing --> d): ")
    q = input("Enter q value (Moving Average (MA) --> q): ")
    print('ARIMA(%d, %d, %d)' % (int(p), int(d), int(q)))
    # load model
    print("Loading model...")
    model_fit = ARIMAResults.load('data/saved_models/' + csv_file_name.split('.csv')[0] + '_arima_80_20.pkl')
    # bias = numpy.load('model_bias.npy')
    # make first prediction
    print("Starting model evaluation...")
    predictions = list()
    # yhat = bias + float(model_fit.forecast()[0])
    yhat = float(model_fit.forecast()[0])
    predictions.append(yhat)
    history.append(y[0])
    print('>Predicted=%.3f, Expected=%.3f' % (yhat, y[0]))
    # rolling forecasts
    for i in range(1, len(y)):
        # predict
        model = ARIMA(history, order=(int(p), int(d), int(q)))
        model_fit = model.fit()
        # yhat = bias + float(model_fit.forecast()[0])
        yhat = float(model_fit.forecast()[0])
        predictions.append(yhat)
        # observation
        obs = y[i]
        history.append(obs)
        print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
    # report performance
    rmse = sqrt(mean_squared_error(y, predictions))
    print('RMSE: %.3f' % rmse)
    print("Model evaluation finished...")
    print("==========================================================")
    print()
    pyplot.plot(y)
    pyplot.plot(predictions, color='red')
    pyplot.show()

    # ==================================================================================================================
    # load data for 90% - 10%
    dataset = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_90_10.csv', header=None,
                       index_col=0, parse_dates=True, squeeze=True)
    X = dataset.values.astype('float32')
    history = [x for x in X]
    validation = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_validation_90_10.csv', header=None,
                          index_col=0, parse_dates=True, squeeze=True)
    y = validation.values.astype('float32')
    print()
    print("==========================================================")
    print("For 90% - 10% we have...")
    print("Enter (p,  d, q) extracted by the conclusions made by stationarity and  ACF/PACF plots")
    p = input("Enter p value (Autoregression (AR) --> p): ")
    d = input("Enter d value (differencing --> d): ")
    q = input("Enter q value (Moving Average (MA) --> q): ")
    print('ARIMA(%d, %d, %d)' % (int(p), int(d), int(q)))
    # load model
    print("Loading model...")
    model_fit = ARIMAResults.load('data/saved_models/' + csv_file_name.split('.csv')[0] + '_arima_90_10.pkl')
    # bias = numpy.load('model_bias.npy')
    # make first prediction
    print("Starting model evaluation...")
    predictions = list()
    # yhat = bias + float(model_fit.forecast()[0])
    yhat = float(model_fit.forecast()[0])
    predictions.append(yhat)
    history.append(y[0])
    print('>Predicted=%.3f, Expected=%.3f' % (yhat, y[0]))
    # rolling forecasts
    for i in range(1, len(y)):
        # predict
        model = ARIMA(history, order=(int(p), int(d), int(q)))
        model_fit = model.fit()
        # yhat = bias + float(model_fit.forecast()[0])
        yhat = float(model_fit.forecast()[0])
        predictions.append(yhat)
        # observation
        obs = y[i]
        history.append(obs)
        print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
    # report performance
    rmse = sqrt(mean_squared_error(y, predictions))
    print('RMSE: %.3f' % rmse)
    print("Model evaluation finished...")
    print("==========================================================")
    print()
    pyplot.plot(y)
    pyplot.plot(predictions, color='red')
    pyplot.show()
