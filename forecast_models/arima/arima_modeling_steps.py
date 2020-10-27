import warnings
from math import sqrt
from pandas import read_csv
from pandas import DataFrame
from matplotlib import pyplot
# Deprecated import: from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


def manual_arima(csv_file_name):
    # load data for 70% - 30%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_70_30.csv', header=None,
                      index_col=0, parse_dates=True, squeeze=True)
    print()
    print("==========================================================")
    print("For 70% - 30% we have...")
    print("Enter (p, d, q) extracted by the conclusions made by stationarity and  ACF/PACF plots")
    p = input("Enter p value (Autoregression (AR) --> p): ")
    d = input("Enter d value (differencing --> d): ")
    q = input("Enter q value (Moving Average (MA) --> q): ")
    print('ARIMA(%d, %d, %d)' % (int(p), int(d), int(q)))
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
        model = ARIMA(history, order=(int(p), int(d), int(q)))
        model_fit = model.fit()
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
    print("Enter (p, d, q) extracted by the conclusions made by stationarity and  ACF/PACF plots")
    p = input("Enter p value (Autoregression (AR) --> p): ")
    d = input("Enter d value (differencing --> d): ")
    q = input("Enter q value (Moving Average (MA) --> q): ")
    print('ARIMA(%d, %d, %d)' % (int(p), int(d), int(q)))
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
        model = ARIMA(history, order=(int(p), int(d), int(q)))
        model_fit = model.fit()
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
    print("Enter (p, d, q) extracted by the conclusions made by stationarity and  ACF/PACF plots")
    p = input("Enter p value (Autoregression (AR) --> p): ")
    d = input("Enter d value (differencing --> d): ")
    q = input("Enter q value (Moving Average (MA) --> q): ")
    print('ARIMA(%d, %d, %d)' % (int(p), int(d), int(q)))
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
        model = ARIMA(history, order=(int(p), int(d), int(q)))
        model_fit = model.fit()
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

    # load data for 95% - 5%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_95_5.csv', header=None,
                      index_col=0, parse_dates=True, squeeze=True)
    print()
    print("==========================================================")
    print("For 95% - 5% we have...")
    print("Enter (p, d, q) extracted by the conclusions made by stationarity and  ACF/PACF plots")
    p = input("Enter p value (Autoregression (AR) --> p): ")
    d = input("Enter d value (differencing --> d): ")
    q = input("Enter q value (Moving Average (MA) --> q): ")
    print('ARIMA(%d, %d, %d)' % (int(p), int(d), int(q)))
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
        model = ARIMA(history, order=(int(p), int(d), int(q)))
        model_fit = model.fit()
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


# evaluate an ARIMA model for a given order (p,d,q) and return RMSE
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    X = X.astype('float32')
    train_size = int(len(X) * 0.50)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    rmse = sqrt(mean_squared_error(test, predictions))

    return rmse


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    rmse = evaluate_arima_model(dataset, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print('ARIMA%s RMSE=%.3f' % (order, rmse))
                except:
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))


def grid_search_arima(csv_file_name):
    # load data for 70% - 30%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_70_30.csv', header=None,
                      index_col=0, parse_dates=True, squeeze=True)
    print()
    print("==========================================================")
    print("For 70% - 30% we have...")
    # evaluate parameters
    p_values = range(0, 13)
    d_values = range(0, 4)
    q_values = range(0, 13)
    warnings.filterwarnings("ignore")
    evaluate_models(series.values, p_values, d_values, q_values)
    print("==========================================================")
    print()

    # load data for 80% - 20%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_80_20.csv', header=None,
                      index_col=0, parse_dates=True, squeeze=True)
    print()
    print("==========================================================")
    print("For 80% - 20% we have...")
    # evaluate parameters
    p_values = range(0, 13)
    d_values = range(0, 4)
    q_values = range(0, 13)
    warnings.filterwarnings("ignore")
    evaluate_models(series.values, p_values, d_values, q_values)
    print("==========================================================")
    print()

    # load data for 90% - 10%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_90_10.csv', header=None,
                      index_col=0, parse_dates=True, squeeze=True)
    print()
    print("==========================================================")
    print("For 90% - 10% we have...")
    # evaluate parameters
    p_values = range(0, 13)
    d_values = range(0, 4)
    q_values = range(0, 13)
    warnings.filterwarnings("ignore")
    evaluate_models(series.values, p_values, d_values, q_values)
    print("==========================================================")
    print()

    # load data for 95% - 5%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_95_5.csv', header=None,
                      index_col=0, parse_dates=True, squeeze=True)
    print()
    print("==========================================================")
    print("For 95% - 5% we have...")
    # evaluate parameters
    p_values = range(0, 13)
    d_values = range(0, 4)
    q_values = range(0, 13)
    warnings.filterwarnings("ignore")
    evaluate_models(series.values, p_values, d_values, q_values)
    print("==========================================================")
    print()


def residual_errors_plot_arima(csv_file_name):
    # load data for 70% - 30%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_70_30.csv', header=None,
                      index_col=0, parse_dates=True, squeeze=True)
    print()
    print("==========================================================")
    print("For 70% - 30% we have...")
    print("Enter (p, d, q) extracted by the conclusions made by stationarity and  ACF/PACF plots")
    p = input("Enter p value (Autoregression (AR) --> p): ")
    d = input("Enter d value (differencing --> d): ")
    q = input("Enter q value (Moving Average (MA) --> q): ")
    print('ARIMA(%d, %d, %d)' % (int(p), int(d), int(q)))
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
        model = ARIMA(history, order=(int(p), int(d), int(q)))
        model_fit = model.fit()
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
    print("Enter (p, d, q) extracted by the conclusions made by stationarity and  ACF/PACF plots")
    p = input("Enter p value (Autoregression (AR) --> p): ")
    d = input("Enter d value (differencing --> d): ")
    q = input("Enter q value (Moving Average (MA) --> q): ")
    print('ARIMA(%d, %d, %d)' % (int(p), int(d), int(q)))
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
        model = ARIMA(history, order=(int(p), int(d), int(q)))
        model_fit = model.fit()
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
    print("Enter (p, d, q) extracted by the conclusions made by stationarity and  ACF/PACF plots")
    p = input("Enter p value (Autoregression (AR) --> p): ")
    d = input("Enter d value (differencing --> d): ")
    q = input("Enter q value (Moving Average (MA) --> q): ")
    print('ARIMA(%d, %d, %d)' % (int(p), int(d), int(q)))
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
        model = ARIMA(history, order=(int(p), int(d), int(q)))
        model_fit = model.fit()
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

    # load data for 95% - 5%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_95_5.csv', header=None,
                      index_col=0, parse_dates=True, squeeze=True)
    print()
    print("==========================================================")
    print("For 95% - 5% we have...")
    print("Enter (p, d, q) extracted by the conclusions made by stationarity and  ACF/PACF plots")
    p = input("Enter p value (Autoregression (AR) --> p): ")
    d = input("Enter d value (differencing --> d): ")
    q = input("Enter q value (Moving Average (MA) --> q): ")
    print('ARIMA(%d, %d, %d)' % (int(p), int(d), int(q)))
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
        model = ARIMA(history, order=(int(p), int(d), int(q)))
        model_fit = model.fit()
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


def residual_acf_errors_plot_arima(csv_file_name):
    # load data for 70% - 30%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_70_30.csv', header=None,
                      index_col=0, parse_dates=True, squeeze=True)
    print()
    print("==========================================================")
    print("For 70% - 30% we have...")
    print("Enter (p, d, q) extracted by the conclusions made by stationarity and  ACF/PACF plots")
    p = input("Enter p value (Autoregression (AR) --> p): ")
    d = input("Enter d value (differencing --> d): ")
    q = input("Enter q value (Moving Average (MA) --> q): ")
    print('ARIMA(%d, %d, %d)' % (int(p), int(d), int(q)))
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
        model = ARIMA(history, order=(int(p), int(d), int(q)))
        model_fit = model.fit()
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
    plot_acf(residuals, ax=pyplot.gca())
    pyplot.subplot(212)
    plot_pacf(residuals, ax=pyplot.gca())
    pyplot.show()

    # load data for 80% - 20%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_80_20.csv', header=None,
                      index_col=0, parse_dates=True, squeeze=True)
    print()
    print("==========================================================")
    print("For 80% - 20% we have...")
    print("Enter (p, d, q) extracted by the conclusions made by stationarity and  ACF/PACF plots")
    p = input("Enter p value (Autoregression (AR) --> p): ")
    d = input("Enter d value (differencing --> d): ")
    q = input("Enter q value (Moving Average (MA) --> q): ")
    print('ARIMA(%d, %d, %d)' % (int(p), int(d), int(q)))
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
        model = ARIMA(history, order=(int(p), int(d), int(q)))
        model_fit = model.fit()
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
    plot_acf(residuals, ax=pyplot.gca())
    pyplot.subplot(212)
    plot_pacf(residuals, ax=pyplot.gca())
    pyplot.show()

    # load data for 90% - 10%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_90_10.csv', header=None,
                      index_col=0, parse_dates=True, squeeze=True)
    print()
    print("==========================================================")
    print("For 90% - 10% we have...")
    print("Enter (p, d, q) extracted by the conclusions made by stationarity and  ACF/PACF plots")
    p = input("Enter p value (Autoregression (AR) --> p): ")
    d = input("Enter d value (differencing --> d): ")
    q = input("Enter q value (Moving Average (MA) --> q): ")
    print('ARIMA(%d, %d, %d)' % (int(p), int(d), int(q)))
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
        model = ARIMA(history, order=(int(p), int(d), int(q)))
        model_fit = model.fit()
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
    plot_acf(residuals, ax=pyplot.gca())
    pyplot.subplot(212)
    plot_pacf(residuals, ax=pyplot.gca())
    pyplot.show()

# load data for 95% - 5%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_95_5.csv', header=None,
                      index_col=0, parse_dates=True, squeeze=True)
    print()
    print("==========================================================")
    print("For 95% - 5% we have...")
    print("Enter (p, d, q) extracted by the conclusions made by stationarity and  ACF/PACF plots")
    p = input("Enter p value (Autoregression (AR) --> p): ")
    d = input("Enter d value (differencing --> d): ")
    q = input("Enter q value (Moving Average (MA) --> q): ")
    print('ARIMA(%d, %d, %d)' % (int(p), int(d), int(q)))
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
        model = ARIMA(history, order=(int(p), int(d), int(q)))
        model_fit = model.fit()
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
    plot_acf(residuals, ax=pyplot.gca())
    pyplot.subplot(212)
    plot_pacf(residuals, ax=pyplot.gca())
    pyplot.show()
