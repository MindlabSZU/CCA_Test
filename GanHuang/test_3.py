import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA

idx_sub=0
idx_run=0
t = np.arange(0, 1, 0.001)
fft_result=np.zeros((500,))

freqs = np.fft.fftfreq(len(t), t[1] - t[0])
freq_bin=np.arange(8, 16, 0.4)

corr=np.zeros((40,20))
cca = CCA(n_components=2)
data=np.zeros((40,0,1000))

for idx_sub in range(1):
    for idx_run in range(2):
        temp = np.load('..\data\data1\X_data_subject_'+str(idx_sub+16)+'.'+str(idx_run+1)+'.npy')
        data = np.concatenate((data, temp), axis=1)


for idx_trial in range(40):
    for k in range(10):
        idx_ch=np.random.choice(np.arange(120), size=8, replace=False)
        X=data[idx_trial,idx_ch,:].T
        print([idx_trial,k])
        for idx_freq in range(20):
            Y=np.zeros((1000,4))
            Y[:,0] = np.sin(2 * np.pi * freq_bin[idx_freq] * t) 
            Y[:,1] = np.cos(2 * np.pi * freq_bin[idx_freq] * t) 
            Y[:,2] = np.sin(2 * np.pi * 2 * freq_bin[idx_freq] * t) 
            Y[:,3] = np.cos(2 * np.pi * 2 * freq_bin[idx_freq] * t)     
                    
            cca.fit(X, Y)
            corr[idx_trial,idx_freq] += cca.score(X, Y)
        
