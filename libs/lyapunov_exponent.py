import nolds
from pandas import read_csv

# lm = nolds.logistic_map(0.1, 1000, r=4)
# x = np.fromiter(lm, dtype="float32")
# l = max(nolds.lyap_e(x))
# k = nolds.lyap_r(x)
# print(l)
# print(k)

# load dataset
series = read_csv('../data/datasets/ubuntu_application_server_final_week.csv',
                  header=0, index_col=0, parse_dates=True, squeeze=True)
x = series.values
x = x.astype('float32')
# x = np.fromiter(series.values, dtype="float32")

# emb_dim_ext = 3
# matrix_dim_ext = 3
# min_nb_ext = min(2 * matrix_dim_ext, matrix_dim_ext + 4)
# e = max(nolds.lyap_e(x, emb_dim=emb_dim_ext, matrix_dim=matrix_dim_ext, min_nb=min_nb_ext))
# print("Eckmann: " + str(e))
# r = nolds.lyap_r(x, emb_dim=emb_dim_ext, lag=None, min_tsep=None, min_neighbors=3, trajectory_len=3)

print("Calculate LLE using Rosenstein algorithm with "
      "emb_dim: " + str(3) +
      " lag: " + str(1) +
      "min_neighbors: " + str(3) +
      "trajectory_len: " + str(3))
r = nolds.lyap_r(x, emb_dim=3, lag=1, min_tsep=None, min_neighbors=3, trajectory_len=3)
print("result: " + str(r))
print("Calculate LLE using Rosenstein algorithm with "
      "emb_dim: " + str(3) +
      " lag: " + str(1) +
      "min_neighbors: " + str(5) +
      "trajectory_len: " + str(5))
r = nolds.lyap_r(x, emb_dim=3, lag=1, min_tsep=None, min_neighbors=5, trajectory_len=5)
print("result: " + str(r))
print("Calculate LLE using Rosenstein algorithm with "
      "emb_dim: " + str(3) +
      " lag: " + str(1) +
      "min_neighbors: " + str(7) +
      "trajectory_len: " + str(7))
r = nolds.lyap_r(x, emb_dim=3, lag=1, min_tsep=None, min_neighbors=7, trajectory_len=7)
print("result: " + str(r))


print("Calculate LLE using Rosenstein algorithm with "
      "emb_dim: " + str(5) +
      " lag: " + str(1) +
      "min_neighbors: " + str(3) +
      "trajectory_len: " + str(3))
r = nolds.lyap_r(x, emb_dim=5, lag=1, min_tsep=None, min_neighbors=3, trajectory_len=3)
print("result: " + str(r))
print("Calculate LLE using Rosenstein algorithm with "
      "emb_dim: " + str(5) +
      " lag: " + str(1) +
      "min_neighbors: " + str(5) +
      "trajectory_len: " + str(5))
r = nolds.lyap_r(x, emb_dim=5, lag=1, min_tsep=None, min_neighbors=5, trajectory_len=5)
print("result: " + str(r))
print("Calculate LLE using Rosenstein algorithm with "
      "emb_dim: " + str(5) +
      " lag: " + str(1) +
      "min_neighbors: " + str(7) +
      "trajectory_len: " + str(7))
r = nolds.lyap_r(x, emb_dim=5, lag=1, min_tsep=None, min_neighbors=7, trajectory_len=7)
print("result: " + str(r))

print("Calculate LLE using Rosenstein algorithm with "
      "emb_dim: " + str(7) +
      " lag: " + str(1) +
      "min_neighbors: " + str(3) +
      "trajectory_len: " + str(3))
r = nolds.lyap_r(x, emb_dim=7, lag=1, min_tsep=None, min_neighbors=3, trajectory_len=3)
print("result: " + str(r))
print("Calculate LLE using Rosenstein algorithm with "
      "emb_dim: " + str(7) +
      " lag: " + str(1) +
      "min_neighbors: " + str(5) +
      "trajectory_len: " + str(5))
r = nolds.lyap_r(x, emb_dim=7, lag=1, min_tsep=None, min_neighbors=5, trajectory_len=5)
print("result: " + str(r))
print("Calculate LLE using Rosenstein algorithm with "
      "emb_dim: " + str(7) +
      " lag: " + str(1) +
      "min_neighbors: " + str(7) +
      "trajectory_len: " + str(7))
r = nolds.lyap_r(x, emb_dim=7, lag=1, min_tsep=None, min_neighbors=7, trajectory_len=7)
print("result: " + str(r))
# ======================================================================================================================
print("Calculate LLE using Rosenstein algorithm with "
      "emb_dim: " + str(3) +
      " lag: " + str(2) +
      "min_neighbors: " + str(3) +
      "trajectory_len: " + str(3))
r = nolds.lyap_r(x, emb_dim=3, lag=2, min_tsep=None, min_neighbors=3, trajectory_len=3)
print("result: " + str(r))
print("Calculate LLE using Rosenstein algorithm with "
      "emb_dim: " + str(3) +
      " lag: " + str(2) +
      "min_neighbors: " + str(5) +
      "trajectory_len: " + str(5))
r = nolds.lyap_r(x, emb_dim=3, lag=2, min_tsep=None, min_neighbors=5, trajectory_len=5)
print("result: " + str(r))
print("Calculate LLE using Rosenstein algorithm with "
      "emb_dim: " + str(3) +
      " lag: " + str(2) +
      "min_neighbors: " + str(7) +
      "trajectory_len: " + str(7))
r = nolds.lyap_r(x, emb_dim=3, lag=2, min_tsep=None, min_neighbors=7, trajectory_len=7)
print("result: " + str(r))

print("Calculate LLE using Rosenstein algorithm with "
      "emb_dim: " + str(5) +
      " lag: " + str(2) +
      "min_neighbors: " + str(3) +
      "trajectory_len: " + str(3))
r = nolds.lyap_r(x, emb_dim=5, lag=2, min_tsep=None, min_neighbors=3, trajectory_len=3)
print("result: " + str(r))
print("Calculate LLE using Rosenstein algorithm with "
      "emb_dim: " + str(5) +
      " lag: " + str(2) +
      "min_neighbors: " + str(5) +
      "trajectory_len: " + str(5))
r = nolds.lyap_r(x, emb_dim=5, lag=2, min_tsep=None, min_neighbors=5, trajectory_len=5)
print("result: " + str(r))
print("Calculate LLE using Rosenstein algorithm with "
      "emb_dim: " + str(5) +
      " lag: " + str(2) +
      "min_neighbors: " + str(7) +
      "trajectory_len: " + str(7))
r = nolds.lyap_r(x, emb_dim=5, lag=2, min_tsep=None, min_neighbors=7, trajectory_len=7)
print("result: " + str(r))

print("Calculate LLE using Rosenstein algorithm with "
      "emb_dim: " + str(7) +
      " lag: " + str(2) +
      "min_neighbors: " + str(3) +
      "trajectory_len: " + str(3))
r = nolds.lyap_r(x, emb_dim=7, lag=2, min_tsep=None, min_neighbors=3, trajectory_len=3)
print("result: " + str(r))
print("Calculate LLE using Rosenstein algorithm with "
      "emb_dim: " + str(7) +
      " lag: " + str(2) +
      "min_neighbors: " + str(5) +
      "trajectory_len: " + str(5))
r = nolds.lyap_r(x, emb_dim=7, lag=2, min_tsep=None, min_neighbors=5, trajectory_len=5)
print("result: " + str(r))
print("Calculate LLE using Rosenstein algorithm with "
      "emb_dim: " + str(7) +
      " lag: " + str(2) +
      "min_neighbors: " + str(7) +
      "trajectory_len: " + str(7))
r = nolds.lyap_r(x, emb_dim=7, lag=2, min_tsep=None, min_neighbors=7, trajectory_len=7)
print("result: " + str(r))
# ======================================================================================================================
print("Calculate LLE using Rosenstein algorithm with "
      "emb_dim: " + str(3) +
      " lag: " + str(3) +
      "min_neighbors: " + str(3) +
      "trajectory_len: " + str(3))
r = nolds.lyap_r(x, emb_dim=3, lag=3, min_tsep=None, min_neighbors=3, trajectory_len=3)
print("result: " + str(r))
print("Calculate LLE using Rosenstein algorithm with "
      "emb_dim: " + str(3) +
      " lag: " + str(3) +
      "min_neighbors: " + str(5) +
      "trajectory_len: " + str(5))
r = nolds.lyap_r(x, emb_dim=3, lag=3, min_tsep=None, min_neighbors=5, trajectory_len=5)
print("result: " + str(r))
print("Calculate LLE using Rosenstein algorithm with "
      "emb_dim: " + str(3) +
      " lag: " + str(3) +
      "min_neighbors: " + str(7) +
      "trajectory_len: " + str(7))
r = nolds.lyap_r(x, emb_dim=3, lag=3, min_tsep=None, min_neighbors=7, trajectory_len=7)
print("result: " + str(r))

print("Calculate LLE using Rosenstein algorithm with "
      "emb_dim: " + str(5) +
      " lag: " + str(3) +
      "min_neighbors: " + str(3) +
      "trajectory_len: " + str(3))
r = nolds.lyap_r(x, emb_dim=5, lag=3, min_tsep=None, min_neighbors=3, trajectory_len=3)
print("result: " + str(r))
print("Calculate LLE using Rosenstein algorithm with "
      "emb_dim: " + str(5) +
      " lag: " + str(3) +
      "min_neighbors: " + str(5) +
      "trajectory_len: " + str(5))
r = nolds.lyap_r(x, emb_dim=5, lag=3, min_tsep=None, min_neighbors=5, trajectory_len=5)
print("result: " + str(r))
print("Calculate LLE using Rosenstein algorithm with "
      "emb_dim: " + str(5) +
      " lag: " + str(3) +
      "min_neighbors: " + str(7) +
      "trajectory_len: " + str(7))
r = nolds.lyap_r(x, emb_dim=5, lag=3, min_tsep=None, min_neighbors=7, trajectory_len=7)
print("result: " + str(r))

print("Calculate LLE using Rosenstein algorithm with "
      "emb_dim: " + str(7) +
      " lag: " + str(3) +
      "min_neighbors: " + str(3) +
      "trajectory_len: " + str(3))
r = nolds.lyap_r(x, emb_dim=7, lag=3, min_tsep=None, min_neighbors=3, trajectory_len=3)
print("result: " + str(r))
print("Calculate LLE using Rosenstein algorithm with "
      "emb_dim: " + str(7) +
      " lag: " + str(3) +
      "min_neighbors: " + str(5) +
      "trajectory_len: " + str(5))
r = nolds.lyap_r(x, emb_dim=7, lag=3, min_tsep=None, min_neighbors=5, trajectory_len=5)
print("result: " + str(r))
print("Calculate LLE using Rosenstein algorithm with "
      "emb_dim: " + str(7) +
      " lag: " + str(3) +
      "min_neighbors: " + str(7) +
      "trajectory_len: " + str(7))
r = nolds.lyap_r(x, emb_dim=7, lag=3, min_tsep=None, min_neighbors=7, trajectory_len=7)
print("result: " + str(r))
