import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from scipy.io import loadmat
#计算70人最大值预测准确率

t=np.arange(0, 3, 0.004)
t_idx = np.where((t > 0.14) & (t < 2))[0]
t=t[t_idx]
# Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, O2
ch_idx=np.array([48,55,54,57,58,56,61,62,63])-1

freqs = np.fft.fftfreq(len(t), t[1] - t[0])
freq_bin = np.arange(8, 16, 0.2)

corr=np.zeros((160,40))
cca = CCA(n_components=1)

sub_idx=1
mat_data = loadmat('..\data\data3\S'+str(sub_idx+1)+'.mat',struct_as_record=True)['data']
data=mat_data['EEG'][0, 0]
marker=mat_data['suppl_info'][0, 0]['freqs'][0, 0]
# del mat_data
sorted_idx = np.squeeze(np.argsort(marker))
data=data[:,:,:,sorted_idx]
data=data[:,t_idx,:,:]
# data=data[ch_idx,:,:,:]
# del ch_idx,t_idx,sorted_idx
result=np.zeros(160)
for idx_trial in range(40):
    for idx_block in range(4):
        X=data[:,:,idx_block,idx_trial].T
        for idx_freq in range(40):
            Y=np.zeros((len(t),8))
            Y[:,0] = np.sin(2 * np.pi * freq_bin[idx_freq] * t) 
            Y[:,1] = np.cos(2 * np.pi * freq_bin[idx_freq] * t) 
            Y[:,2] = np.sin(2 * np.pi * 2 * freq_bin[idx_freq] * t) 
            Y[:,3] = np.cos(2 * np.pi * 2 * freq_bin[idx_freq] * t)     
            Y[:,4] = np.sin(2 * np.pi * 3 * freq_bin[idx_freq] * t) 
            Y[:,5] = np.cos(2 * np.pi * 3 * freq_bin[idx_freq] * t)     
            Y[:,6] = np.sin(2 * np.pi * 4 * freq_bin[idx_freq] * t) 
            Y[:,7] = np.cos(2 * np.pi * 4 * freq_bin[idx_freq] * t)     
                     
            cca.fit(X, Y)
            corr[idx_trial*4+idx_block,idx_freq] = cca.score(X, Y)
        if np.argmax(corr[idx_trial*4+idx_block,:])==idx_trial:
            result[idx_trial*4+idx_block]=1
print(np.sum(result)/160*100)

plt.plot(result)