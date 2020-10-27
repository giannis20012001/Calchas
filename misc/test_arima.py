import warnings
import numpy as np
from math import sqrt
from pandas import read_csv
from matplotlib import pyplot
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima_model import ARIMAResults


# split into a training and validation dataset
def split_dataset(csv_file_name):
    series = read_csv('../data/datasets/test_datasets/water.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
    # Split dataset 70% - 30%
    split_point = int(len(series) * 0.7)
    dataset, validation = series[0:split_point], series[split_point:]
    print()
    print("==========================================================")
    print("For 70% - 30% we have...")
    print('Observations: %d' % (len(series)))
    print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
    print("==========================================================")
    print("Save dataset & validation set to path...")
    print("==========================================================")
    print()
    dataset.to_csv(r'../data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_70_30.csv', header=False)
    validation.to_csv(r'../data/datasets/' + csv_file_name.split('.csv')[0] + '_validation_70_30.csv', header=False)

    # Split dataset 80% - 20%
    split_point = int(len(series) * 0.8)
    dataset, validation = series[0:split_point], series[split_point:]
    print()
    print("==========================================================")
    print("For 80% - 20% we have...")
    print('Observations: %d' % (len(series)))
    print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
    print("==========================================================")
    print("Save dataset & validation set to path...")
    print("==========================================================")
    print()
    dataset.to_csv(r'../data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_80_20.csv', header=False)
    validation.to_csv(r'../data/datasets/' + csv_file_name.split('.csv')[0] + '_validation_80_20.csv', header=False)

    # Split dataset 90% - 10%
    split_point = int(len(series) * 0.9)
    dataset, validation = series[0:split_point], series[split_point:]
    print()
    print("==========================================================")
    print("For 90% - 10% we have...")
    print('Observations: %d' % (len(series)))
    print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
    print("==========================================================")
    print("Save dataset & validation set to path...")
    print("==========================================================")
    print()
    dataset.to_csv(r'../data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_90_10.csv', header=False)
    validation.to_csv(r'../data/datasets/' + csv_file_name.split('.csv')[0] + '_validation_90_10.csv', header=False)

    # Split dataset 95% - 5%
    split_point = int(len(series) * 0.95)
    dataset, validation = series[0:split_point], series[split_point:]
    print()
    print("==========================================================")
    print("For 95% - 5% we have...")
    print('Observations: %d' % (len(series)))
    print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
    print("==========================================================")
    print("Save dataset & validation set to path...")
    print("==========================================================")
    print()
    dataset.to_csv(r'../data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_95_5.csv', header=False)
    validation.to_csv(r'../data/datasets/' + csv_file_name.split('.csv')[0] + '_validation_95_5.csv', header=False)


# Create line plots of the time series
def create_line_plots(csv_file_name):
    # load data for 70% - 30%
    series = read_csv('../data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_70_30.csv', header=None,
                      index_col=0, parse_dates=True, squeeze=True)
    series.plot()
    pyplot.show()

    # load data for 80% - 20%
    series = read_csv('../data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_80_20.csv', header=None,
                      index_col=0, parse_dates=True, squeeze=True)
    series.plot()
    pyplot.show()

    # load data for 90% - 10%
    series = read_csv('../data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_90_10.csv', header=None,
                      index_col=0, parse_dates=True, squeeze=True)
    series.plot()
    pyplot.show()

    # load data for 95% - 5%
    series = read_csv('../data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_95_5.csv', header=None,
                      index_col=0, parse_dates=True, squeeze=True)
    series.plot()
    pyplot.show()


def manual_arima(csv_file_name):
    # load data for 70% - 30%
    series = read_csv('../data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_70_30.csv', header=None,
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
    series = read_csv('../data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_80_20.csv', header=None,
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
    series = read_csv('../data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_90_10.csv', header=None,
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
    series = read_csv('../data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_95_5.csv', header=None,
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


def save_fitted_model_arima(csv_file_name):
    # load data for 70% - 30%
    series = read_csv('../data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_70_30.csv', header=None,
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
    warnings.filterwarnings("ignore")
    model = ARIMA(X, order=(int(p), int(d), int(q)))
    model_fit = model.fit()
    # bias constant, could be calculated from in-sample mean residual
    # bias = 1.081624
    # save model
    print("Saving model...")
    model_fit.save('../data/saved_models/' + csv_file_name.split('.csv')[0] + '_arima_70_30.pkl')
    # numpy.save('model_bias.npy', [bias])
    print("==========================================================")
    print()

    # ==================================================================================================================
    # load data for 80% - 20%
    series = read_csv('../data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_80_20.csv', header=None,
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
    warnings.filterwarnings("ignore")
    model = ARIMA(X, order=(int(p), int(d), int(q)))
    model_fit = model.fit()
    # bias constant, could be calculated from in-sample mean residual
    # bias = 1.081624
    # save model
    print("Saving model...")
    model_fit.save('../data/saved_models/' + csv_file_name.split('.csv')[0] + '_arima_80_20.pkl')
    # numpy.save('model_bias.npy', [bias])
    print("==========================================================")
    print()

    # ==================================================================================================================
    # load data for 90% - 10%
    series = read_csv('../data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_90_10.csv', header=None,
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
    warnings.filterwarnings("ignore")
    model = ARIMA(X, order=(int(p), int(d), int(q)))
    model_fit = model.fit()
    # bias constant, could be calculated from in-sample mean residual
    # bias = 1.081624
    # save model
    print("Saving model...")
    model_fit.save('../data/saved_models/' + csv_file_name.split('.csv')[0] + '_arima_90_10.pkl')
    # numpy.save('model_bias.npy', [bias])
    print("==========================================================")
    print()

    # ==================================================================================================================
    # load data for 95% - 5%
    series = read_csv('../data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_95_5.csv', header=None,
                      index_col=0, parse_dates=True, squeeze=True)
    print()
    print("==========================================================")
    print("For 95% - 5% we have...")
    print("Enter (p,  d, q) extracted by the conclusions made by stationarity and  ACF/PACF plots")
    p = input("Enter p value (Autoregression (AR) --> p): ")
    d = input("Enter d value (differencing --> d): ")
    q = input("Enter q value (Moving Average (MA) --> q): ")
    print('ARIMA(%d, %d, %d)' % (int(p), int(d), int(q)))
    # prepare data
    X = series.values
    X = X.astype('float32')
    # fit model
    warnings.filterwarnings("ignore")
    model = ARIMA(X, order=(int(p), int(d), int(q)))
    model_fit = model.fit()
    # bias constant, could be calculated from in-sample mean residual
    # bias = 1.081624
    # save model
    print("Saving model...")
    model_fit.save('../data/saved_models/' + csv_file_name.split('.csv')[0] + '_arima_95_5.pkl')
    # numpy.save('model_bias.npy', [bias])
    print("==========================================================")
    print()


def calculate_forecasting_performance_measures(expected, predictions):
    y_true = np.array(expected)
    y_pred = np.array(predictions)
    forecast_errors = [y_true[i] - y_pred[i] for i in range(len(y_true))]
    bias = sum(forecast_errors) * 1.0 / len(expected)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))

    print()
    print('Forecast Errors: %s' % forecast_errors)
    print("The units of the forecast error are the same as the units of the prediction. "
          "A forecast error of zero indicates no error, or perfect skill for that forecast.")
    print()
    print('Forecast Bias (Mean Forecast Error): %f' % bias)
    print("The units of the forecast bias are the same as the units of the predictions. "
          "A forecast bias of zero, or a very small number near zero, shows an unbiased model.")
    print()
    print('Mean Absolute Error (MAE): %f' % mae)
    print("These error values are in the original units of the predicted values. "
          "A mean absolute error of zero indicates no error.")
    print()
    print('Mean Squared Error (MSE): %f' % mse)
    print("The error values are in squared units of the predicted values. "
          "A mean squared error of zero indicates perfect skill, or no error.")
    print()
    print('Root Mean Squared Error (RMSE): %.3f' % rmse)
    print("The RMES error values are in the same units as the predictions. "
          "As with the mean squared error, an RMSE of zero indicates no error.")
    print()


# noinspection PyTypeChecker
def calculate_correlation_index(expected, predictions):
    y_true = np.array(expected)
    y_pred = np.array(predictions)

    # calculate Pearson's correlation
    corr, _ = pearsonr(y_true, y_pred)
    print('Pearsons correlation: %.3f' % corr)
    print("The coefficient returns a value between -1 and 1 that represents the limits of correlation "
          "from a full negative correlation to a full positive correlation. A value of 0 means no correlation. "
          "The value must be interpreted, where often a value below -0.5 or above 0.5 indicates a notable correlation, "
          "and values below those values suggests a less notable correlation.")
    print()

    # calculate spearman's correlation
    corr, _ = spearmanr(y_true, y_pred)
    print('Spearmans correlation: %.3f' % corr)
    print("The coefficient returns a value between -1 and 1 that represents the limits of correlation "
          "from a full negative correlation to a full positive correlation. A value of 0 means no correlation. "
          "The value must be interpreted, where often a value below -0.5 or above 0.5 indicates a notable correlation, "
          "and values below those values suggests a less notable correlation.")
    print()

    # calculate the coefficient of determination,
    r_sqr = r2_score(y_true, y_pred)
    print('R^2 (coefficient of determination): %.3f' % r_sqr)
    print()


def validate_arima_model(csv_file_name):
    # load data for 70% - 30%
    dataset = read_csv('../data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_70_30.csv', header=None,
                       index_col=0, parse_dates=True, squeeze=True)
    X = dataset.values.astype('float32')
    history = [x for x in X]
    validation = read_csv('../data/datasets/' + csv_file_name.split('.csv')[0] + '_validation_70_30.csv', header=None,
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
    model_fit = ARIMAResults.load('../data/saved_models/' + csv_file_name.split('.csv')[0] + '_arima_70_30.pkl')
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
        warnings.filterwarnings("ignore")
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
    calculate_forecasting_performance_measures(y, predictions)
    calculate_correlation_index(y, predictions)
    print("Model evaluation finished...")
    print("==========================================================")
    print()
    pyplot.plot(y)
    pyplot.plot(predictions, color='red')
    pyplot.show()

    # ==================================================================================================================
    # load data for 80% - 20%
    dataset = read_csv('../data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_80_20.csv', header=None,
                       index_col=0, parse_dates=True, squeeze=True)
    X = dataset.values.astype('float32')
    history = [x for x in X]
    validation = read_csv('../data/datasets/' + csv_file_name.split('.csv')[0] + '_validation_80_20.csv', header=None,
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
    model_fit = ARIMAResults.load('../data/saved_models/' + csv_file_name.split('.csv')[0] + '_arima_80_20.pkl')
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
        warnings.filterwarnings("ignore")
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
    calculate_forecasting_performance_measures(y, predictions)
    calculate_correlation_index(y, predictions)
    print("Model evaluation finished...")
    print("==========================================================")
    print()
    pyplot.plot(y)
    pyplot.plot(predictions, color='red')
    pyplot.show()

    # ==================================================================================================================
    # load data for 90% - 10%
    dataset = read_csv('../data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_90_10.csv', header=None,
                       index_col=0, parse_dates=True, squeeze=True)
    X = dataset.values.astype('float32')
    history = [x for x in X]
    validation = read_csv('../data/datasets/' + csv_file_name.split('.csv')[0] + '_validation_90_10.csv', header=None,
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
    model_fit = ARIMAResults.load('../data/saved_models/' + csv_file_name.split('.csv')[0] + '_arima_90_10.pkl')
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
        warnings.filterwarnings("ignore")
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
    calculate_forecasting_performance_measures(y, predictions)
    calculate_correlation_index(y, predictions)
    print("Model evaluation finished...")
    print("==========================================================")
    print()
    pyplot.plot(y)
    pyplot.plot(predictions, color='red')
    pyplot.show()

    # ==================================================================================================================
    # load data for 95% - 5%
    dataset = read_csv('../data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_95_5.csv', header=None,
                       index_col=0, parse_dates=True, squeeze=True)
    X = dataset.values.astype('float32')
    history = [x for x in X]
    validation = read_csv('../data/datasets/' + csv_file_name.split('.csv')[0] + '_validation_95_5.csv', header=None,
                          index_col=0, parse_dates=True, squeeze=True)
    y = validation.values.astype('float32')
    print()
    print("==========================================================")
    print("For 95% - 5% we have...")
    print("Enter (p,  d, q) extracted by the conclusions made by stationarity and  ACF/PACF plots")
    p = input("Enter p value (Autoregression (AR) --> p): ")
    d = input("Enter d value (differencing --> d): ")
    q = input("Enter q value (Moving Average (MA) --> q): ")
    print('ARIMA(%d, %d, %d)' % (int(p), int(d), int(q)))
    # load model
    print("Loading model...")
    model_fit = ARIMAResults.load('../data/saved_models/' + csv_file_name.split('.csv')[0] + '_arima_95_5.pkl')
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
        warnings.filterwarnings("ignore")
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
    calculate_forecasting_performance_measures(y, predictions)
    calculate_correlation_index(y, predictions)
    print("Model evaluation finished...")
    print("==========================================================")
    print()
    pyplot.plot(y)
    pyplot.plot(predictions, color='red')
    pyplot.show()


# ======================================================================================================
# Common steps
# ======================================================================================================
csv_file_name = "water.csv"
print("Performing initial common steps for " + csv_file_name + " dataset...")
print("==========================================================")
print("First step split initial dataset to training & validation ones...")
split_dataset(csv_file_name)
input("Press Enter to continue...")

print("Second step create line plots...")
create_line_plots(csv_file_name)
input("Press Enter to continue...")


print("ARIMA model steps...")
# ======================================================================================================
# Modeling steps
# ======================================================================================================
print("Third step run manual ARIMA for specific (p, q, d) values...")
manual_arima(csv_file_name)
input("Press Enter to continue...")

# ======================================================================================================
# Validation steps
# ======================================================================================================
print("Fourth step save fitted model...")
save_fitted_model_arima(csv_file_name)
input("Press Enter to continue...")

print("Fifth step validate fitted model...")
validate_arima_model(csv_file_name)
input("Press Enter to continue...")
