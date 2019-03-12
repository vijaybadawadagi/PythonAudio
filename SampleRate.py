# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:38:40 2019

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
xx = np.mean(yy)
print('freq', Fs/np.mean(np.diff(crossings)))

signalTrans = np.transpose(signal)

cross = np.sign(signalTrans[:])
zero = np.where(np.diff(cross) > 0)
ZC = len(zero)

print('SR',ZC*Fs)
