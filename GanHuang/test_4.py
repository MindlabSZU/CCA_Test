import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA

idx_sub=0
data = np.load('..\data\data2\X_data_subject_[1].npy')
label = np.load('..\data\data2\y_labels_subject_[1].npy')

t = np.arange(0, 0.75, 0.005)
fft_result=np.zeros((150,))

freqs = np.fft.fftfreq(len(t), t[1] - t[0])
freq_bin=[8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,8.2,9.2,10.2,11.2,12.2,13.2,14.2,15.2,8.4,9.4,10.4,11.4,12.4,13.4,14.4,15.4,8.6,9.6,10.6,11.6,12.6,13.6,14.6,15.6,8.8,9.8,10.8,11.8,12.8,13.8,14.8,15.8]
freq_bin = np.array(freq_bin)

corr=np.zeros((240,40))
cca = CCA(n_components=2)
for idx_trial in range(240):
    X=data[idx_trial,:,:].T
    for idx_freq in range(40):
        Y=np.zeros((150,6))
        Y[:,0] = np.sin(2 * np.pi * freq_bin[idx_freq] * t) 
        Y[:,1] = np.cos(2 * np.pi * freq_bin[idx_freq] * t) 
        Y[:,2] = np.sin(2 * np.pi * 2 * freq_bin[idx_freq] * t) 
        Y[:,3] = np.cos(2 * np.pi * 2 * freq_bin[idx_freq] * t)     
        Y[:,4] = np.sin(2 * np.pi * 3 * freq_bin[idx_freq] * t) 
        Y[:,5] = np.cos(2 * np.pi * 3 * freq_bin[idx_freq] * t)     
                 
        cca.fit(X, Y)
        corr[idx_trial,idx_freq] += cca.score(X, Y)
result=np.zeros((40,40))
for idx_trial in range(40):
    result[idx_trial,:]=np.mean(corr[idx_trial*6:(idx_trial+1)*6,:], axis=0)

sorted_idx = np.argsort(freq_bin)
result=result[:,sorted_idx]
# result=result[sorted_idx,:]