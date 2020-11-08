import nolds
import numpy as np
from pandas import read_csv
from matplotlib import pyplot

# lm = nolds.logistic_map(0.1, 1000, r=4)
# x = np.fromiter(lm, dtype="float32")
# l = max(nolds.lyap_e(x))
# k = nolds.lyap_r(x)
# print(l)
# print(k)

# load dataset
series = read_csv('../data/datasets/openstack_controller_server_final_week.csv',
                  header=0, index_col=0, parse_dates=True, squeeze=True)
x = series.values
x = x.astype('float32')
# x = np.fromiter(series.values, dtype="float32")

l = max(nolds.lyap_e(x))
k = nolds.lyap_r(x, lag=None, min_tsep=None)
print(l)
print(k)
