# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 23:38:22 2024

@author: balazs
"""

from scipy import signal
import numpy as np


class Lock_in_amplifier:
    def __init__(self, time_constant, filter_order = 4):
        self.cutoff_freq = 2*np.pi/time_constant
        self.filter_order = filter_order
    
    def get_ref(self, ref_freq, ts):
        ref_cos = np.cos(2*np.pi*ref_freq*ts)
        ref_sin = np.sin(2*np.pi*ref_freq*ts)
        
        return ref_cos, ref_sin
    
    def sig_out(self, sig_in, ref_freq):
        sig = sig_in[:, 1]
        ts = sig_in[:, 0]
        
        ref_cos, ref_sin = self.get_ref(ref_freq, ts)
        
        i_sig = sig*ref_cos
        q_sig = sig*ref_sin
        
        sos = signal.butter(self.filter_order, self.cutoff_freq, 'low', output='sos', fs = 1/(ts[1]-ts[0]))
        
        i_sig_filt = signal.sosfilt(sos, i_sig)
        q_sig_filt = signal.sosfilt(sos, q_sig)
        
        return i_sig_filt, q_sig_filt
    
    