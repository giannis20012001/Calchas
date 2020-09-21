import sqlite3
import pandas as pd
from pandas import DataFrame

# Create a SQL connection to our SQLite database
con = sqlite3.connect("/home/lumi/Dropbox/unipi/paper_NVD_forcasting/sqlight_db/nvd_nist.db")

# Read sqlite query results into a pandas DataFrame
df = pd.read_sql_query("SELECT published_datetime, score from openstack_controller_server_final", con)
con.close()
df.published_datetime = pd.to_datetime(df.published_datetime)

# Various experiments
# df['published_datetime'] = df['published_datetime'].dt.date
# df['published_datetime'] = df['published_datetime'].dt.normalize()
# p = df.set_index('published_datetime')

# See duplicate values for the same date
print(df[df.published_datetime.duplicated()].count())
temp = df[df.published_datetime.duplicated()]
print(temp[temp.published_datetime == '2005-01-10'])
print()
print("===============================================")

# Remove duplicate values for the same date
df = df.groupby('published_datetime').max()
df = df.reset_index()
print(df[df.published_datetime.duplicated()].count())
temp = df[df.published_datetime.duplicated()]
print(temp[temp.published_datetime == '2005-01-10'])
print()
print("===============================================")

# create series from dataframe
ts = pd.Series(df['score'].values, index=df['published_datetime'])
ts.index = pd.DatetimeIndex(ts.index)
# Create all missing date values
idx = pd.date_range(df['published_datetime'].min(), df['published_datetime'].max())
ts = ts.reindex(idx, fill_value=0)
print(ts['2005-01-10'])
print()
print("===============================================")


# Verify that result of SQL query is stored in the dataframe
# print(ts.head())
# print(ts.size)
print(ts.describe())
print()
print("===============================================")

# Create missing values table view for day
temp = DataFrame()
