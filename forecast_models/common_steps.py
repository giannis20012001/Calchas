from math import sqrt
from pandas import Grouper
from pandas import read_csv
from matplotlib import pyplot
from pandas import DataFrame, concat
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


# split into a training and validation dataset
def split_dataset(csv_file_name):
    series = read_csv('data/datasets/' + csv_file_name, header=0, index_col=0, parse_dates=True, squeeze=True)
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
    dataset.to_csv(r'data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_70_30.csv',
                   index_label='index', header=['values'])
    validation.to_csv(r'data/datasets/' + csv_file_name.split('.csv')[0] + '_validation_70_30.csv',
                      index_label='index', header=['values'])

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
    dataset.to_csv(r'data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_80_20.csv',
                   index_label='index', header=['values'])
    validation.to_csv(r'data/datasets/' + csv_file_name.split('.csv')[0] + '_validation_80_20.csv',
                      index_label='index', header=['values'])

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
    dataset.to_csv(r'data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_90_10.csv',
                   index_label='index', header=['values'])
    validation.to_csv(r'data/datasets/' + csv_file_name.split('.csv')[0] + '_validation_90_10.csv',
                      index_label='index', header=['values'])

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
    dataset.to_csv(r'data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_95_5.csv',
                   index_label='index', header=['values'])
    validation.to_csv(r'data/datasets/' + csv_file_name.split('.csv')[0] + '_validation_95_5.csv',
                      index_label='index', header=['values'])


# evaluate a persistence model
def create_persistence_model_from_dataset(csv_file_name):
    # load data for 70% - 30%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_70_30.csv',
                      header=0, index_col=0, parse_dates=True, squeeze=True)
    # prepare data
    X = series.values
    X = X.astype('float32')
    train_size = int(len(X) * 0.50)
    train, test = X[0:train_size], X[train_size:]
    # walk-forward validation
    history = [x for x in train]
    predictions = list()
    print()
    print("==========================================================")
    print("For 70% - 30% we have...")
    for i in range(len(test)):
        # predict
        yhat = history[-1]
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
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_80_20.csv',
                      header=0, index_col=0, parse_dates=True, squeeze=True)
    # prepare data
    X = series.values
    X = X.astype('float32')
    train_size = int(len(X) * 0.50)
    train, test = X[0:train_size], X[train_size:]
    # walk-forward validation
    history = [x for x in train]
    predictions = list()
    print()
    print("==========================================================")
    print("For 80% - 20% we have...")
    for i in range(len(test)):
        # predict
        yhat = history[-1]
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
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_90_10.csv',
                      header=0, index_col=0, parse_dates=True, squeeze=True)
    # prepare data
    X = series.values
    X = X.astype('float32')
    train_size = int(len(X) * 0.50)
    train, test = X[0:train_size], X[train_size:]
    # walk-forward validation
    history = [x for x in train]
    predictions = list()
    print()
    print("==========================================================")
    print("For 90% - 10% we have...")
    for i in range(len(test)):
        # predict
        yhat = history[-1]
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
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_95_5.csv',
                      header=0, index_col=0, parse_dates=True, squeeze=True)
    # prepare data
    X = series.values
    X = X.astype('float32')
    train_size = int(len(X) * 0.50)
    train, test = X[0:train_size], X[train_size:]
    # walk-forward validation
    history = [x for x in train]
    predictions = list()
    print()
    print("==========================================================")
    print("For 95% - 5% we have...")
    for i in range(len(test)):
        # predict
        yhat = history[-1]
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


# summary statistics of the time series
def summary_statistics(csv_file_name):
    # load data for 70% - 30%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_70_30.csv',
                      header=0, index_col=0, parse_dates=True, squeeze=True)
    print()
    print("==========================================================")
    print("For 70% - 30% we have...")
    print(series.describe())
    print("==========================================================")
    print()

    # load data for 80% - 20%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_80_20.csv',
                      header=0, index_col=0, parse_dates=True, squeeze=True)
    print()
    print("==========================================================")
    print("For 80% - 20% we have...")
    print(series.describe())
    print("==========================================================")
    print()

    # load data for 90% - 10%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_90_10.csv',
                      header=0, index_col=0, parse_dates=True, squeeze=True)
    print()
    print("==========================================================")
    print("For 90% - 10% we have...")
    print(series.describe())
    print("==========================================================")
    print()

    # load data for 95% - 5%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_95_5.csv',
                      header=0, index_col=0, parse_dates=True, squeeze=True)
    print()
    print("==========================================================")
    print("For 95% - 5% we have...")
    print(series.describe())
    print("==========================================================")
    print()


# Create line plots of the time series
def create_line_plots(csv_file_name):
    # load data for 70% - 30%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_70_30.csv',
                      header=0, index_col=0, parse_dates=True, squeeze=True)
    series.plot()
    pyplot.show()

    # load data for 80% - 20%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_80_20.csv',
                      header=0, index_col=0, parse_dates=True, squeeze=True)
    series.plot()
    pyplot.show()

    # load data for 90% - 10%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_90_10.csv',
                      header=0, index_col=0, parse_dates=True, squeeze=True)
    series.plot()
    pyplot.show()

    # load data for 95% - 5%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_95_5.csv',
                      header=0, index_col=0, parse_dates=True, squeeze=True)
    series.plot()
    pyplot.show()


# Create density plots of the time series
def create_density_plots(csv_file_name):
    # load data for 70% - 30%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_70_30.csv',
                      header=0, index_col=0, parse_dates=True, squeeze=True)
    pyplot.figure(1)
    pyplot.subplot(211)
    series.hist()
    pyplot.subplot(212)
    series.plot(kind='kde')
    pyplot.show()

    # load data for 80% - 20%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_80_20.csv',
                      header=0, index_col=0, parse_dates=True, squeeze=True)
    pyplot.figure(1)
    pyplot.subplot(211)
    series.hist()
    pyplot.subplot(212)
    series.plot(kind='kde')
    pyplot.show()

    # load data for 90% - 10%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_90_10.csv',
                      header=0, index_col=0, parse_dates=True, squeeze=True)
    pyplot.figure(1)
    pyplot.subplot(211)
    series.hist()
    pyplot.subplot(212)
    series.plot(kind='kde')
    pyplot.show()

    # load data for 95% - 5%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_95_5.csv',
                      header=0, index_col=0, parse_dates=True, squeeze=True)
    pyplot.figure(1)
    pyplot.subplot(211)
    series.hist()
    pyplot.subplot(212)
    series.plot(kind='kde')
    pyplot.show()


# Create boxplots plots of the time series
def create_boxplots_plots(csv_file_name):
    # load data for 70% - 30%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_70_30.csv',
                      header=0, index_col=0, parse_dates=True, squeeze=True)
    groups = series.groupby(Grouper(freq='A'))
    years = DataFrame()
    appended_data = []
    for name, group in groups:
        years[name.year] = group.values
        appended_data.append(years)
        years = DataFrame()

    appended_data = concat(appended_data)
    appended_data.boxplot()
    pyplot.show()

    # load data for 80% - 20%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_80_20.csv',
                      header=0, index_col=0, parse_dates=True, squeeze=True)
    groups = series.groupby(Grouper(freq='A'))
    years = DataFrame()
    appended_data = []
    for name, group in groups:
        years[name.year] = group.values
        appended_data.append(years)
        years = DataFrame()

    appended_data = concat(appended_data)
    appended_data.boxplot()
    pyplot.show()

    # load data for 90% - 10%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_90_10.csv',
                      header=0, index_col=0, parse_dates=True, squeeze=True)
    groups = series.groupby(Grouper(freq='A'))
    years = DataFrame()
    appended_data = []
    for name, group in groups:
        years[name.year] = group.values
        appended_data.append(years)
        years = DataFrame()

    appended_data = concat(appended_data)
    appended_data.boxplot()
    pyplot.show()

    # load data for 95% - 5%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_95_5.csv',
                      header=0, index_col=0, parse_dates=True, squeeze=True)
    groups = series.groupby(Grouper(freq='A'))
    years = DataFrame()
    appended_data = []
    for name, group in groups:
        years[name.year] = group.values
        appended_data.append(years)
        years = DataFrame()

    appended_data = concat(appended_data)
    appended_data.boxplot()
    pyplot.show()


# statistical test for the stationarity of the time series
def check_stationarity(csv_file_name):
    # load data for 70% - 30%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_70_30.csv',
                      header=0, index_col=0, parse_dates=True, squeeze=True)

    # check if stationary
    result = adfuller(series)
    print()
    print("==========================================================")
    print("For 70% - 30% we have...")
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    print("==========================================================")
    print()

    # load data for 80% - 20%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_80_20.csv',
                      header=0, index_col=0, parse_dates=True, squeeze=True)

    # check if stationary
    result = adfuller(series)
    print()
    print("==========================================================")
    print("For 80% - 20% we have...")
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    print("==========================================================")
    print()

    # load data for 90% - 10%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_90_10.csv',
                      header=0, index_col=0, parse_dates=True, squeeze=True)

    # check if stationary
    result = adfuller(series)
    print()
    print("==========================================================")
    print("For 90% - 10% we have...")
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    print("==========================================================")
    print()

    # check if stationary
    result = adfuller(series)
    print()
    print("==========================================================")
    print("For 95% - 5% we have...")
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    print("==========================================================")
    print()


def acf_pacf_plots(csv_file_name):
    # load data for 70% - 30%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_70_30.csv',
                      header=0, index_col=0, parse_dates=True, squeeze=True)
    pyplot.figure()
    pyplot.subplot(211)
    plot_acf(series, ax=pyplot.gca())
    pyplot.subplot(212)
    plot_pacf(series, ax=pyplot.gca())
    pyplot.show()

    # load data for 80% - 20%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_80_20.csv',
                      header=0, index_col=0, parse_dates=True, squeeze=True)
    pyplot.figure()
    pyplot.subplot(211)
    plot_acf(series, ax=pyplot.gca())
    pyplot.subplot(212)
    plot_pacf(series, ax=pyplot.gca())
    pyplot.show()

    # load data for 90% - 10%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_90_10.csv',
                      header=0, index_col=0, parse_dates=True, squeeze=True)
    pyplot.figure()
    pyplot.subplot(211)
    plot_acf(series, ax=pyplot.gca())
    pyplot.subplot(212)
    plot_pacf(series, ax=pyplot.gca())
    pyplot.show()

    # load data for 95% - 5%
    series = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_95_5.csv',
                      header=0, index_col=0, parse_dates=True, squeeze=True)
    pyplot.figure()
    pyplot.subplot(211)
    plot_acf(series, ax=pyplot.gca())
    pyplot.subplot(212)
    plot_pacf(series, ax=pyplot.gca())
    pyplot.show()
