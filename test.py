import sqlite3
import pandas as pd

# Create a SQL connection to our SQLite database
con = sqlite3.connect("/home/lumi/Dropbox/unipi/paper_NVD_forcasting/sqlight_db/nvd_nist.db")
# Read sqlite query results into a pandas DataFrame
df = pd.read_sql_query("SELECT published_datetime, score from openstack_controller_server_final", con)
df.published_datetime = pd.to_datetime(df.published_datetime)

df['published_datetime'] = df['published_datetime'].dt.date
# df['published_datetime'] = df['published_datetime'].dt.normalize()

# create series from dataframe
ts = pd.Series(df['score'].values, index=df['published_datetime'])

p = df.set_index('published_datetime')

# Verify that result of SQL query is stored in the dataframe
# print(df.head())
# print(ts.size)
print(ts.describe())

con.close()
