import impyute as impy
from pandas import read_csv
from pandas import concat, DataFrame
from statsmodels.multivariate.pca import PCA


# transform list into supervised learning format
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

    return agg.values


series = read_csv('../data/datasets/test_datasets/openstack_controller_server_final_week_nan.csv',
                  header=0, index_col=0, parse_dates=True, squeeze=True)

A = series_to_supervised(series.values, n_in=3)
# A = series.values.reshape(-1, 1)
test_2 = impy.moving_window(A, wsize=5)
# print(test_2)


pc = PCA(data=A, ncomp=1, missing='fill-em')
B = pc._adjusted_data
print(B)
