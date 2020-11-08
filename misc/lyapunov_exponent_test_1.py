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

emb_dim_ext = 3
matrix_dim_ext = 3
min_nb_ext = min(2 * matrix_dim_ext, matrix_dim_ext + 4)
e = max(nolds.lyap_e(x, emb_dim=emb_dim_ext, matrix_dim=matrix_dim_ext, min_nb=min_nb_ext))
r = nolds.lyap_r(x, emb_dim=emb_dim_ext, lag=None, min_tsep=None, min_neighbors=3, trajectory_len=3)
print("Eckmann: " + str(e))
print("Rosenstein: " + str(r))
