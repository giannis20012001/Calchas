import numpy as np
from math import sqrt
from matplotlib import pyplot
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import r2_score
from pandas import read_csv, DataFrame, concat
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# transform list into supervised learning format
# noinspection DuplicatedCode
def series_to_supervised(data, n_in=1, n_out=1):
    df = DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = concat(cols, axis=1)
    # drop rows with NaN values
    agg.dropna(inplace=True)

    return agg.values


# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


# fit a model
def model_fit(train, config):
    # unpack config
    n_input, n_nodes, n_epochs, n_batch = config
    # prepare data
    data = series_to_supervised(train, n_in=n_input)
    train_x, train_y = data[:, :-1], data[:, -1]
    # define model
    model = Sequential()
    model.add(Dense(n_nodes, activation='relu', input_dim=n_input))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    # fit
    model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)

    return model


# forecast with a pre-fit model
def model_predict(model, history, config):
    # unpack config
    n_input, _, _, _ = config
    # prepare data
    x_input = np.array(history[-n_input:]).reshape(1, n_input)
    # forecast
    yhat = model.predict(x_input, verbose=0)

    return yhat[0]


# noinspection DuplicatedCode
def calculate_forecasting_performance_measures(expected, predictions):
    y_true = np.array(expected)
    y_pred = np.array(predictions)
    forecast_errors = [y_true[i] - y_pred[i] for i in range(len(y_true))]
    bias = sum(forecast_errors) * 1.0 / len(expected)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    print('Forecast Bias (Mean Forecast Error): %f' % bias)
    print('Mean Absolute Error (MAE): %f' % mae)
    print('Mean Squared Error (MSE): %f' % mse)


# noinspection PyTypeChecker
def calculate_correlation_index(expected, predictions):
    y_true = np.array(expected)
    y_pred = np.array(predictions)
    # calculate the coefficient of determination,
    r_sqr = r2_score(y_true, y_pred)
    print('R^2 (coefficient of determination): %.3f' % r_sqr)
    print()


# walk-forward validation for univariate data
def walk_forward_validation(train, test, cfg):
    predictions = list()
    # fit model
    model = model_fit(train, cfg)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = model_predict(model, history, cfg)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    error = measure_rmse(test, predictions)
    print(' > %.3f' % error)
    # report performance
    calculate_forecasting_performance_measures(test, predictions)
    calculate_correlation_index(test, predictions)
    pyplot.plot(test)
    pyplot.plot(predictions, color='red')
    pyplot.show()

    return error


# summarize model performance
# noinspection PyStringFormat
def summarize_scores(scores):
    # print a summary
    scores_m, score_std = np.mean(scores), np.std(scores)
    print('%s: %.3f RMSE (+/- %.3f)' % ('mlp', scores_m, score_std))
    print("Model evaluation finished...")
    print("==========================================================")
    print()
    # box and whisker plot
    pyplot.boxplot(scores)
    pyplot.show()


# repeat evaluation of a config
# noinspection DuplicatedCode,PyPep8Naming
def repeat_evaluate(csv_file_name, n_repeats=30):
    # load data for 70% - 30%
    dataset = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_70_30.csv',
                       header=0, index_col=0, parse_dates=True, squeeze=True)
    validation = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_validation_70_30.csv',
                          header=0, index_col=0, parse_dates=True, squeeze=True)
    print()
    print("==========================================================")
    print("For 70% - 30% we have...")
    train = dataset.values
    test = validation.values
    # define config
    config = [24, 500, 100, 100]
    # fit and evaluate the model n times
    scores = [walk_forward_validation(train, test, config) for _ in range(n_repeats)]
    summarize_scores(scores)

    # ==================================================================================================================
    # load data for 80% - 20%
    dataset = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_80_20.csv',
                       header=0, index_col=0, parse_dates=True, squeeze=True)
    validation = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_validation_80_20.csv',
                          header=0, index_col=0, parse_dates=True, squeeze=True)
    print()
    print("==========================================================")
    print("For 80% - 20% we have...")
    input("Press Enter to continue...")
    train = dataset.values
    test = validation.values
    # define config
    config = [24, 500, 100, 100]
    # fit and evaluate the model n times
    scores = [walk_forward_validation(train, test, config) for _ in range(n_repeats)]
    summarize_scores(scores)

    # ==================================================================================================================
    # load data for 90% - 10%
    dataset = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_dataset_90_10.csv',
                       header=0, index_col=0, parse_dates=True, squeeze=True)
    X = dataset.values.astype('float32')
    history = [x for x in X]
    validation = read_csv('data/datasets/' + csv_file_name.split('.csv')[0] + '_validation_90_10.csv',
                          header=0, index_col=0, parse_dates=True, squeeze=True)
    print()
    print("==========================================================")
    print("For 90% - 10% we have...")
    input("Press Enter to continue...")
    train = dataset.values
    test = validation.values
    # define config
    config = [24, 500, 100, 100]
    # fit and evaluate the model n times
    scores = [walk_forward_validation(train, test, config) for _ in range(n_repeats)]
    summarize_scores(scores)
