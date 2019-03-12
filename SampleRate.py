# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:38:40 2019

Find the freq of the input signal using Zero crossings of the signal 

@author: VijayB
"""
import numpy as np
from matplotlib.mlab import find

Fs = 8000
f = 350
T = 5.0
n = int(T * Fs)
x = np.linspace(0,T,n,endpoint=False)
signal = 0.9 * np.sin(2*np.pi*f*x)


indices = find((signal[1:] >= 0) & (signal[:-1] < 0))

crossings = indices
yy = np.diff(crossings)
print('freq', Fs/np.mean(np.diff(crossings)))
