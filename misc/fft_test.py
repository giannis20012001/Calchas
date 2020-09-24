import sqlite3
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================================================================================
# Function Space
# ======================================================================================================================
# noinspection DuplicatedCode
def fft_plot(sig, dt=None, plot=True):
    # here it's assumes analytic signal (real signal...) - so only half of the axis is required
    if dt is None:
        dt = 1
        t = np.arange(0, sig.shape[-1])
        x_label = 'samples'
    else:
        t = np.arange(0, sig.shape[-1]) * dt
        x_label = 'freq [Hz]'

    if sig.shape[0] % 2 != 0:
        warnings.warn("signal preferred to be even in size, autoFixing it...")
        t = t[0:-1]
        sig = sig[0:-1]

    sig_fft = np.fft.fft(sig) / t.shape[0]  # divided by size t for coherent magnitude

    freq = np.fft.fftfreq(t.shape[0], d=dt)

    print(freq[:10])
    first_ten = 0
    for x in np.nditer(freq[:11]):
        first_ten = first_ten + x

    print(freq[-10:])
    last_ten = 0
    for x in np.nditer(freq[-10:]):
        last_ten = last_ten + x

    result = last_ten / first_ten

    # plot analytic signal - right half of freq axis needed only...
    first_neg_ind = np.argmax(freq < 0)
    freq_axis_pos = freq[0:first_neg_ind]
    sig_fft_pos = 2 * sig_fft[0:first_neg_ind]  # *2 because of magnitude of analytic signal

    if plot:
        plt.figure()
        plt.plot(freq_axis_pos, np.abs(sig_fft_pos))
        plt.xlabel(x_label)
        plt.ylabel('mag')
        plt.title('Analytic FFT plot')
        plt.show()

    return result

# ======================================================================================================================
# ======================================================================================================================
# Create a SQL connection to our SQLite database
# noinspection DuplicatedCode
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
# temp = DataFrame()

D = ts.size
time_interval = ts.index[-1] - ts.index[1]
# fs is sampling frequency
fs = float(D / time_interval.days)

print(fft_plot(ts.values, dt=fs))
