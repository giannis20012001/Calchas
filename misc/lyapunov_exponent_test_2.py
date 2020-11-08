import os
import numpy
from math import log
from pandas import read_csv


def d(series, i, j):
    return abs(series[i] - series[j])


# load dataset
# series = read_csv('../data/datasets/openstack_controller_server_final_week.csv',
                  # header=0, index_col=0, parse_dates=True, squeeze=True)
# x = series.values.reshape(-1, 1)
# numpy.savetxt('timeseries.txt', x, fmt='%.15f', newline=os.linesep)

f = open('timeseries.txt', 'r')
series = [float(i) for i in f.read().split()]
f.close()

N = len(series)
eps = input('Initial diameter bound: ')
dlist = [[] for i in range(N)]
n = 0  # number of nearby pairs found
for i in range(N):
    for j in range(i + 1, N):
        if d(series, i, j) < float(eps):
            n += 1
            print
            n
            for k in range(min(N - i, N - j)):
                test = d(series, i + k, j + k)
                dlist[k].append(log(test))
f = open('lyapunov.txt', 'w')
for i in range(len(dlist)):
    if len(dlist[i]):
        print >> f, i, sum(dlist[i]) / len(dlist[i])
f.close()
