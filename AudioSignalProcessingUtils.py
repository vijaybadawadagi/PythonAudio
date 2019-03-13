# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:54:37 2019

@author: VijayB
"""
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
from scipy.signal import fft, freqz, lfilter, butter, fftconvolve
get_ipython().run_line_magic('matplotlib', 'qt')

class pyAudio:
    __dealy = 0
    __corr = 0
    __lag = 0
    __nyq = 0
    def __init__(self):
        self.__dealy = 0
        self.__corr = 0
        self.__lag = 0
        self.__zz = 0
        self.__nyq = 0
        
    def _get_corr(self):
        return self.__corr
    
    def _get_delay(self, src1, src2):
        self.__corr = fftconvolve(src1 ,src2[::-1])
        self.__zz = np.arange(self.__corr.size)
        self.__lag = self.__zz - (src2.size - 1)
        max  = np.argmax(np.abs(self.__corr))
        self.__delay = self.__lag[max]
        return self.__delay
    
    def _butter_lowpass(self, cutoff, fs, order=6):
        self.__nyq = 0.5 * fs
        normal_cutoff = cutoff/self.__nyq
        b, a = butter(self, order, normal_cutoff, btype='high', analog=False)
        return b, a

    def _butter_lowpass_filter(self, data, cutoff, fs, order=6):
        b, a = self._butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b,a,data)
        return y

    def _custom_fft(self, y, fs):
        T = 1.0 / fs
        print(T)
        N = y.shape[0]
        Y    = fft(y)
        freq = np.fft.fftfreq(len(y), T)
        return freq[0:N//2], 2.0/N *np.abs(Y[0:N//2])


audioObj = pyAudio()
sig1 = np.array([0,1,2,3,4,5,6,7,8,0,0,0,0])
sig2 = np.array([0,0,0,1,2,3,4,5,6,7,8,0,0])
delay = audioObj._get_delay(sig1, sig2)
print('Delay between signals',delay)
plt.subplot(3,1,1)
plt.plot(np.arange(-len(sig1)+1, len(sig1)),audioObj._get_corr())
plt.show()
    



order=6
fs = 30.0
cutoff = 5
b, a = audioObj._butter_lowpass(cutoff, fs, order)
w, h = freqz(b, a, worN=512)

plt.subplot(3,1,1)
plt.plot(0.5 * fs * w/np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5 * np.sqrt(2), 'ko') 
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5 * fs)
plt.title('Low pass filter Frequency response')
plt.xlabel('Frequency [Hz]')
plt.grid()

T = 5.0
n = int(T*fs) # total number of samples
t = np.linspace(0, T, n, endpoint=False)
#data = np.sin(1.2 * 2 * np.pi * t) + 1.5 * np.cos(9*2*np.pi*t)+ 0.5 * np.sin(12.0 * 2 * np.pi*t)
data = np.sin(5*2*np.pi*t) + 3*np.cos(10*2*np.pi*t) + 0.5*np.sin(7.0*2*np.pi*t)

xf, vals = audioObj._custom_fft(data, fs)
plt.figure(figsize=(12,4))
plt.title('FFT of recording sampled with ' + str(fs) + 'Hz')
plt.plot(xf, vals)
plt.xlabel('Frequency')
plt.grid()
plt.show()

y = audioObj._butter_lowpass_filter(data, cutoff, fs, order)
plt.subplot(3, 1, 2)
#plt.plot(t, data, 'b-', label='data')
plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()

plt.subplots_adjust(hspace=0.35)
plt.show()

xf, vals = audioObj._custom_fft(y, fs)
plt.figure(figsize=(12,4))
plt.title('FFT of recording sampled with ' + str(fs) + 'Hz')
plt.plot(xf, vals)
plt.xlabel('Frequency')
plt.grid()
plt.show()
    