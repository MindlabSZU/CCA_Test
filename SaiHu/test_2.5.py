# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 17:06:30 2024

@author: husai
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
#计算自采corr，result为平均加的结果
idx_sub=0
idx_run=0
t = np.arange(0, 1, 0.002)
fft_result=np.zeros((250,))

freqs = np.fft.fftfreq(len(t), t[1] - t[0])
freq_bin=np.arange(8, 16, 0.4)

corr=np.zeros((40,20))
cca = CCA(n_components=2)

for idx_sub in range(1):
    for idx_run in range(5):
        data = np.load('..\data\data1\X_data_subject_'+str(idx_sub+1)+'.'+str(idx_run+1)+'.npy')
        print([idx_sub,idx_run])
        for idx_trial in range(40):
            X=data[idx_trial,:,:].T
            for idx_freq in range(20):
                Y=np.zeros((500,4))
                Y[:,0] = np.sin(2 * np.pi * freq_bin[idx_freq] * t) 
                Y[:,1] = np.cos(2 * np.pi * freq_bin[idx_freq] * t) 
                Y[:,2] = np.sin(2 * np.pi * 2 * freq_bin[idx_freq] * t) 
                Y[:,3] = np.cos(2 * np.pi * 2 * freq_bin[idx_freq] * t)     
                cca.fit(X, Y)
                corr[idx_trial,idx_freq] += cca.score(X, Y)
        
result=np.zeros((20,20))
for idx_trial in range(20):
    result[idx_trial,:]=np.mean(corr[idx_trial:(idx_trial+1),:], axis=0)

sorted_idx = np.argsort(freq_bin)
result=result[:,sorted_idx]