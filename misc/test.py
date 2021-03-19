import numpy as np
import multiprocessing
from joblib import Parallel, delayed

global_var = np.zeros(10)


def populate(idx):
    print('I am core', idx, '\'')
    global_var[idx] = idx


num_cores = multiprocessing.cpu_count()
Parallel(n_jobs=num_cores, require='sharedmem')(delayed(populate)(idx) for idx in range(len(global_var)))

print(global_var)
