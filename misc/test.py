# import nolds
# import numpy as np
from p_tqdm import p_map
# from pandas import read_csv

# lm = nolds.logistic_map(0.1, 1000, r=4)
# x = np.fromiter(lm, dtype="float32")

# sample_entropy = nolds.sampen(x, emb_dim=3, tolerance=None)
# print("Sample entropy: " + str(sample_entropy))
# sample_entropy = nolds.sampen(x, emb_dim=5, tolerance=None)
# print("Sample entropy: " + str(sample_entropy))
# sample_entropy = nolds.sampen(x, emb_dim=7, tolerance=None)
# print("Sample entropy: " + str(sample_entropy))

# load dataset
# series = read_csv('../data/datasets/test_datasets/white_noise_sample_dataset.csv',
#                   header=0, index_col=0, parse_dates=True, squeeze=True)
# x = series.values
# x = x.astype('float32')

# sample_entropy = nolds.sampen(x, emb_dim=3, tolerance=None)
# print("Sample entropy: " + str(sample_entropy))
# sample_entropy = nolds.sampen(x, emb_dim=5, tolerance=None)
# print("Sample entropy: " + str(sample_entropy))
# sample_entropy = nolds.sampen(x, emb_dim=7, tolerance=None)
# print("Sample entropy: " + str(sample_entropy))

# hurst_exponent = nolds.hurst_rs(x)
# print("Sample entropy: " + str(hurst_exponent))


def add(a, b):
    return a + b


l1 = [1, 2, 3]
l2 = [1, 2, 3]

added = p_map(add, ['1', '2', '3'], ['a', 'b', 'c'])

result = list(added)
print(result)
# added == ['1a', '2b', '3c']
