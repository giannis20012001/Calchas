import datetime
import warnings
import numpy as np
import pandas as pd
# import scipy.fftpack
from scipy import fftpack
import matplotlib.pyplot as plt


f = 10  # Frequency, in cycles per second, or Hertz
f_s = 100  # Sampling rate, or number of measurements per second

t = np.linspace(0, 2, 2 * f_s, endpoint=False)
x = np.sin(f * 2 * np.pi * t)

fig, ax = plt.subplots()
ax.plot(t, x)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Signal amplitude')
plt.show()

X = fftpack.fft(x)
freqs = fftpack.fftfreq(len(x)) * f_s

fig, ax = plt.subplots()

ax.stem(freqs, np.abs(X))
ax.set_xlabel('Frequency in Hertz [Hz]')
ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
ax.set_xlim(-f_s / 2, f_s / 2)
ax.set_ylim(-5, 110)
plt.show()

# ===============================================================================
# ===============================================================================
# def fft_plot(sig, dt=None, plot=True):
#     # here it's assumes analytic signal (real signal...)- so only half of the axis is required
#
#     if dt is None:
#         dt = 1
#         t = np.arange(0, sig.shape[-1])
#         xLabel = 'samples'
#     else:
#         t = np.arange(0, sig.shape[-1]) * dt
#         xLabel = 'freq [Hz]'
#
#     if sig.shape[0] % 2 != 0:
#         warnings.warn("signal prefered to be even in size, autoFixing it...")
#         t = t[0:-1]
#         sig = sig[0:-1]
#
#     sigFFT = np.fft.fft(sig) / t.shape[0]  # divided by size t for coherent magnitude
#
#     freq = np.fft.fftfreq(t.shape[0], d=dt)
#
#     # plot analytic signal - right half of freq axis needed only...
#     firstNegInd = np.argmax(freq < 0)
#     freqAxisPos = freq[0:firstNegInd]
#     sigFFTPos = 2 * sigFFT[0:firstNegInd]  # *2 because of magnitude of analytic signal
#
#     if plot:
#         plt.figure()
#         plt.plot(freqAxisPos, np.abs(sigFFTPos))
#         plt.xlabel(xLabel)
#         plt.ylabel('mag')
#         plt.title('Analytic FFT plot')
#         plt.show()
#
#     return sigFFTPos, freqAxisPos


# Set freq
# dt = 1 / 365
# res in freqs
# fftPlot(temp, dt=dt)
# res in samples (if freqs axis is unknown)
# fft_plot(temp)


# ===============================================================================
# ===============================================================================
# df0 = pd.read_csv('./data/datasets/GHCND_sample_csv.csv',
#                   na_values=(-9999), parse_dates=['DATE'])
# df = df0[df0['DATE'] >= '19940101']
# # print(df.head())
#
# df_avg = df.dropna().groupby('DATE').mean()
# # print(df_avg.head())
#
# date = pd.to_datetime(df_avg.index)
# temp = (df_avg['TMAX'] + df_avg['TMIN']) / 20.
# N = len(temp)
#
# fig, ax = plt.subplots(1, 1, figsize=(6, 3))
# temp.plot(ax=ax, lw=.5)
# ax.set_ylim(-10, 40)
# ax.set_xlabel('Date')
# ax.set_ylabel('Mean temperature')
# plt.show()
#
# temp_fft = scipy.fftpack.fft(temp.to_numpy())
# temp_psd = np.abs(temp_fft) ** 2
# fftfreq = scipy.fftpack.fftfreq(len(temp_psd), 1. / 365)
# i = fftfreq > 0
#
# fig, ax = plt.subplots(1, 1, figsize=(8, 4))
# ax.plot(fftfreq[i], 10 * np.log10(temp_psd[i]))
# ax.set_xlim(0, 5)
# ax.set_xlabel('Frequency (1/year)')
# ax.set_ylabel('PSD (dB)')
# plt.show()
#
# temp_fft_bis = temp_fft.copy()
# temp_fft_bis[np.abs(fftfreq) > 1.1] = 0
#
# temp_fft_bis = temp_fft.copy()
# temp_fft_bis[np.abs(fftfreq) > 1.1] = 0
#
# temp_slow = np.real(scipy.fftpack.ifft(temp_fft_bis))
#
# fig, ax = plt.subplots(1, 1, figsize=(6, 3))
# temp.plot(ax=ax, lw=.5)
# ax.plot_date(date, temp_slow, '-')
# ax.set_xlim(datetime.date(1994, 1, 1),
#             datetime.date(2000, 1, 1))
# ax.set_ylim(-10, 40)
# ax.set_xlabel('Date')
# ax.set_ylabel('Mean temperature')
# plt.show()
