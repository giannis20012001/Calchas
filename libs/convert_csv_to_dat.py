import pandas

df = pandas.read_csv("../data/datasets/test_datasets/timeseries.csv")
df.to_csv("../data/datasets/test_datasets/timeseries.dat", sep="|", index=False)
