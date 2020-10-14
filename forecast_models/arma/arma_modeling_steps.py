import warnings
from math import sqrt
from pandas import read_csv
from pandas import DataFrame
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARMA
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


def manual_arma(csv_file_name):
    # load data for 70% - 30%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_70_30.csv', header=None,
                      index_col=0, parse_dates=True, squeeze=True)
    print()
    print("==========================================================")
    print("For 70% - 30% we have...")
    print("Enter (p, q) extracted by the conclusions made by stationarity and  ACF/PACF plots")
    p = input("Enter p value (Autoregression (AR) --> p): ")
    q = input("Enter q value (Moving Average (MA) --> q): ")
    print('ARMA(%d, %d)' % (int(p), int(q)))
    # prepare data
    X = series.values
    X = X.astype('float32')
    train_size = int(len(X) * 0.50)
    train, test = X[0:train_size], X[train_size:]
    # walk-forward validation
    history = [x for x in train]
    predictions = list()
    for i in range(len(test)):
        # predict
        warnings.filterwarnings("ignore")
        model = ARMA(history, order=(int(p), int(q)))
        model_fit = model.fit(disp=False)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        # observation
        obs = test[i]
        history.append(obs)
        print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
    # report performance
    rmse = sqrt(mean_squared_error(test, predictions))
    print('RMSE: %.3f' % rmse)
    print("==========================================================")
    print()

    # load data for 80% - 20%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_80_20.csv', header=None,
                      index_col=0, parse_dates=True, squeeze=True)
    print()
    print("==========================================================")
    print("For 80% - 20% we have...")
    print("Enter (p, q) extracted by the conclusions made by stationarity and  ACF/PACF plots")
    p = input("Enter p value (Autoregression (AR) --> p): ")
    q = input("Enter q value (Moving Average (MA) --> q): ")
    print('ARMA(%d, %d)' % (int(p), int(q)))
    # prepare data
    X = series.values
    X = X.astype('float32')
    train_size = int(len(X) * 0.50)
    train, test = X[0:train_size], X[train_size:]
    # walk-forward validation
    history = [x for x in train]
    predictions = list()
    for i in range(len(test)):
        # predict
        warnings.filterwarnings("ignore")
        model = ARMA(history, order=(int(p), int(q)))
        model_fit = model.fit(disp=False)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        # observation
        obs = test[i]
        history.append(obs)
        print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
    # report performance
    rmse = sqrt(mean_squared_error(test, predictions))
    print('RMSE: %.3f' % rmse)
    print("==========================================================")
    print()

    # load data for 90% - 10%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_90_10.csv', header=None,
                      index_col=0, parse_dates=True, squeeze=True)
    print()
    print("==========================================================")
    print("For 90% - 10% we have...")
    print("Enter (p, q) extracted by the conclusions made by stationarity and  ACF/PACF plots")
    p = input("Enter p value (Autoregression (AR) --> p): ")
    q = input("Enter q value (Moving Average (MA) --> q): ")
    print('ARMA(%d, %d)' % (int(p), int(q)))
    # prepare data
    X = series.values
    X = X.astype('float32')
    train_size = int(len(X) * 0.50)
    train, test = X[0:train_size], X[train_size:]
    # walk-forward validation
    history = [x for x in train]
    predictions = list()
    for i in range(len(test)):
        # predict
        warnings.filterwarnings("ignore")
        model = ARMA(history, order=(int(p), int(q)))
        model_fit = model.fit(disp=False)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        # observation
        obs = test[i]
        history.append(obs)
        print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
    # report performance
    rmse = sqrt(mean_squared_error(test, predictions))
    print('RMSE: %.3f' % rmse)
    print("==========================================================")
    print()


def residual_errors_plot_arma(csv_file_name):
    # load data for 70% - 30%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_70_30.csv', header=None,
                      index_col=0, parse_dates=True, squeeze=True)
    print()
    print("==========================================================")
    print("For 70% - 30% we have...")
    print("Enter (p, q) extracted by the conclusions made by stationarity and  ACF/PACF plots")
    p = input("Enter p value (Autoregression (AR) --> p): ")
    q = input("Enter q value (Moving Average (MA) --> q): ")
    print('ARMA(%d, %d)' % (int(p), int(q)))
    print("==========================================================")
    print()
    # prepare data
    X = series.values
    X = X.astype('float32')
    train_size = int(len(X) * 0.50)
    train, test = X[0:train_size], X[train_size:]
    # walk-forward validation
    history = [x for x in train]
    predictions = list()
    for i in range(len(test)):
        # predict
        warnings.filterwarnings("ignore")
        model = ARMA(history, order=(int(p), int(q)))
        model_fit = model.fit(disp=False)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        # observation
        obs = test[i]
        history.append(obs)
    # errors
    residuals = [test[i] - predictions[i] for i in range(len(test))]
    residuals = DataFrame(residuals)
    print(residuals.describe())
    pyplot.figure()
    pyplot.subplot(211)
    residuals.hist(ax=pyplot.gca())
    pyplot.subplot(212)
    residuals.plot(kind='kde', ax=pyplot.gca())
    pyplot.show()

    # load data for 80% - 20%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_80_20.csv', header=None,
                      index_col=0, parse_dates=True, squeeze=True)
    print()
    print("==========================================================")
    print("For 80% - 20% we have...")
    print("Enter (p, q) extracted by the conclusions made by stationarity and  ACF/PACF plots")
    p = input("Enter p value (Autoregression (AR) --> p): ")
    q = input("Enter q value (Moving Average (MA) --> q): ")
    print('ARMA(%d, %d)' % (int(p), int(q)))
    print("==========================================================")
    print()
    # prepare data
    X = series.values
    X = X.astype('float32')
    train_size = int(len(X) * 0.50)
    train, test = X[0:train_size], X[train_size:]
    # walk-forward validation
    history = [x for x in train]
    predictions = list()
    for i in range(len(test)):
        # predict
        warnings.filterwarnings("ignore")
        model = ARMA(history, order=(int(p), int(q)))
        model_fit = model.fit(disp=False)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        # observation
        obs = test[i]
        history.append(obs)
    # errors
    residuals = [test[i] - predictions[i] for i in range(len(test))]
    residuals = DataFrame(residuals)
    print(residuals.describe())
    pyplot.figure()
    pyplot.subplot(211)
    residuals.hist(ax=pyplot.gca())
    pyplot.subplot(212)
    residuals.plot(kind='kde', ax=pyplot.gca())
    pyplot.show()

    # load data for 90% - 10%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_90_10.csv', header=None,
                      index_col=0, parse_dates=True, squeeze=True)
    print()
    print("==========================================================")
    print("For 90% - 10% we have...")
    print("Enter (p, q) extracted by the conclusions made by stationarity and  ACF/PACF plots")
    p = input("Enter p value (Autoregression (AR) --> p): ")
    q = input("Enter q value (Moving Average (MA) --> q): ")
    print('ARMA(%d, %d)' % (int(p), int(q)))
    print("==========================================================")
    print()
    # prepare data
    X = series.values
    X = X.astype('float32')
    train_size = int(len(X) * 0.50)
    train, test = X[0:train_size], X[train_size:]
    # walk-forward validation
    history = [x for x in train]
    predictions = list()
    for i in range(len(test)):
        # predict
        warnings.filterwarnings("ignore")
        model = ARMA(history, order=(int(p), int(q)))
        model_fit = model.fit(disp=False)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        # observation
        obs = test[i]
        history.append(obs)
    # errors
    residuals = [test[i] - predictions[i] for i in range(len(test))]
    residuals = DataFrame(residuals)
    print(residuals.describe())
    pyplot.figure()
    pyplot.subplot(211)
    residuals.hist(ax=pyplot.gca())
    pyplot.subplot(212)
    residuals.plot(kind='kde', ax=pyplot.gca())
    pyplot.show()


def residual_acf_errors_plot_arma(csv_file_name):
    # load data for 70% - 30%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_70_30.csv', header=None,
                      index_col=0, parse_dates=True, squeeze=True)
    print()
    print("==========================================================")
    print("For 70% - 30% we have...")
    print("Enter (p, q) extracted by the conclusions made by stationarity and  ACF/PACF plots")
    p = input("Enter p value (Autoregression (AR) --> p): ")
    q = input("Enter q value (Moving Average (MA) --> q): ")
    print('ARMA(%d, %d)' % (int(p), int(q)))
    print("==========================================================")
    print()
    # prepare data
    X = series.values
    X = X.astype('float32')
    train_size = int(len(X) * 0.50)
    train, test = X[0:train_size], X[train_size:]
    # walk-forward validation
    history = [x for x in train]
    predictions = list()
    for i in range(len(test)):
        # predict
        warnings.filterwarnings("ignore")
        model = ARMA(history, order=(int(p), int(q)))
        model_fit = model.fit(disp=False)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        # observation
        obs = test[i]
        history.append(obs)
    # errors
    residuals = [test[i] - predictions[i] for i in range(len(test))]
    residuals = DataFrame(residuals)
    pyplot.figure()
    pyplot.subplot(211)
    plot_acf(residuals, lags=25, ax=pyplot.gca())
    pyplot.subplot(212)
    plot_pacf(residuals, lags=25, ax=pyplot.gca())
    pyplot.show()

    # load data for 80% - 20%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_80_20.csv', header=None,
                      index_col=0, parse_dates=True, squeeze=True)
    print()
    print("==========================================================")
    print("For 80% - 20% we have...")
    print("Enter (p, q) extracted by the conclusions made by stationarity and  ACF/PACF plots")
    p = input("Enter p value (Autoregression (AR) --> p): ")
    q = input("Enter q value (Moving Average (MA) --> q): ")
    print('ARMA(%d, %d)' % (int(p), int(q)))
    print("==========================================================")
    print()
    # prepare data
    X = series.values
    X = X.astype('float32')
    train_size = int(len(X) * 0.50)
    train, test = X[0:train_size], X[train_size:]
    # walk-forward validation
    history = [x for x in train]
    predictions = list()
    for i in range(len(test)):
        # predict
        warnings.filterwarnings("ignore")
        model = ARMA(history, order=(int(p), int(q)))
        model_fit = model.fit(disp=False)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        # observation
        obs = test[i]
        history.append(obs)
    # errors
    residuals = [test[i] - predictions[i] for i in range(len(test))]
    residuals = DataFrame(residuals)
    pyplot.figure()
    pyplot.subplot(211)
    plot_acf(residuals, lags=25, ax=pyplot.gca())
    pyplot.subplot(212)
    plot_pacf(residuals, lags=25, ax=pyplot.gca())
    pyplot.show()

    # load data for 90% - 10%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_90_10.csv', header=None,
                      index_col=0, parse_dates=True, squeeze=True)
    print()
    print("==========================================================")
    print("For 90% - 10% we have...")
    print("Enter (p, q) extracted by the conclusions made by stationarity and  ACF/PACF plots")
    p = input("Enter p value (Autoregression (AR) --> p): ")
    q = input("Enter q value (Moving Average (MA) --> q): ")
    print('ARMA(%d, %d)' % (int(p), int(q)))
    print("==========================================================")
    print()
    # prepare data
    X = series.values
    X = X.astype('float32')
    train_size = int(len(X) * 0.50)
    train, test = X[0:train_size], X[train_size:]
    # walk-forward validation
    history = [x for x in train]
    predictions = list()
    for i in range(len(test)):
        # predict
        warnings.filterwarnings("ignore")
        model = ARMA(history, order=(int(p), int(q)))
        model_fit = model.fit(disp=False)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        # observation
        obs = test[i]
        history.append(obs)
    # errors
    residuals = [test[i] - predictions[i] for i in range(len(test))]
    residuals = DataFrame(residuals)
    pyplot.figure()
    pyplot.subplot(211)
    plot_acf(residuals, lags=25, ax=pyplot.gca())
    pyplot.subplot(212)
    plot_pacf(residuals, lags=25, ax=pyplot.gca())
    pyplot.show()
