# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:38:40 2019
Low pass filter 

Sample to filter the low frequency components from the signal
add the oder and cutoff as per need
@author: VijayB
"""
import numpy as np
from scipy.signal import butter, lfilter, freqz, fftconvolve
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from IPython import get_ipython
from matplotlib.mlab import find
get_ipython().run_line_magic('matplotlib', 'qt')


def butter_lowpass(cutoff, fs, order=6):
    nyq = 0.5 * fs
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=6):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b,a,data)
    return y

def custom_fft(y, fs):
    T = 1.0 / fs
    print(T)
    N = y.shape[0]
    Y    = fft(y)
    freq = np.fft.fftfreq(len(y), t[1] - t[0])
    return freq[0:N//2], 2.0/N *np.abs(Y[0:N//2])

order=6
fs = 30.0
cutoff = 5
b, a = butter_lowpass(cutoff, fs, order)
w, h = freqz(b, a, worN=512)

plt.subplot(3,1,1)
plt.plot(0.5 * fs * w/np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5 * np.sqrt(2), 'ko') 
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5 * fs)
plt.title('Low pass filter Frequency response')
plt.xlabel('Frequency [Hz]')
plt.grid()
