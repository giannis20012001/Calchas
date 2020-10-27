from pandas import read_csv
from statsmodels.multivariate.pca import PCA


series = read_csv('../data/datasets/test_datasets/openstack_controller_server_final_week.csv',
                  header=None, index_col=0, parse_dates=True, squeeze=True)

A = series.values
pc = PCA(data=series, ncomp=1, missing='fill-em')
A = pc.adjusted_data
print(A)
