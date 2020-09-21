import datetime
import warnings
import numpy as np
import pandas as pd
import scipy.fftpack
import matplotlib.pyplot as plt


def fftPlot(sig, dt=None, plot=True):
    # here it's assumes analytic signal (real signal...)- so only half of the axis is required

    if dt is None:
        dt = 1
        t = np.arange(0, sig.shape[-1])
        xLabel = 'samples'
    else:
        t = np.arange(0, sig.shape[-1]) * dt
        xLabel = 'freq [Hz]'

    if sig.shape[0] % 2 != 0:
        warnings.warn("signal prefered to be even in size, autoFixing it...")
        t = t[0:-1]
        sig = sig[0:-1]

    sigFFT = np.fft.fft(sig) / t.shape[0]  # divided by size t for coherent magnitude

    freq = np.fft.fftfreq(t.shape[0], d=dt)

    # plot analytic signal - right half of freq axis needed only...
    firstNegInd = np.argmax(freq < 0)
    freqAxisPos = freq[0:firstNegInd]
    sigFFTPos = 2 * sigFFT[0:firstNegInd]  # *2 because of magnitude of analytic signal

    if plot:
        plt.figure()
        plt.plot(freqAxisPos, np.abs(sigFFTPos))
        plt.xlabel(xLabel)
        plt.ylabel('mag')
        plt.title('Analytic FFT plot')
        plt.show()

    return sigFFTPos, freqAxisPos


df0 = pd.read_csv('./data/datasets/GHCND_sample_csv.csv',
                  na_values=(-9999), parse_dates=['DATE'])
df = df0[df0['DATE'] >= '19940101']
# print(df.head())

df_avg = df.dropna().groupby('DATE').mean()
# print(df_avg.head())

date = pd.to_datetime(df_avg.index)
temp = (df_avg['TMAX'] + df_avg['TMIN']) / 20.
N = len(temp)

# fig, ax = plt.subplots(1, 1, figsize=(6, 3))
# temp.plot(ax=ax, lw=.5)
# ax.set_ylim(-10, 40)
# ax.set_xlabel('Date')
# ax.set_ylabel('Mean temperature')
# plt.show()

# Set freq
# dt = 1 / 365
# res in freqs
# fftPlot(temp, dt=dt)
# res in samples (if freqs axis is unknown)
# fftPlot(temp)

temp_fft = scipy.fftpack.fft2(temp)
