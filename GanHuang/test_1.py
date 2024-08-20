import numpy as np
import matplotlib.pyplot as plt


for idx_label in range(20):
    t = np.arange(0, 1, 0.001)
    fft_result=np.zeros((500,))
    freqs = np.fft.fftfreq(len(t), t[1] - t[0])
    for idx_sub in range(1):
        for idx_run in range(5):
            X = np.load('..\data\data1\X_data_subject_'+str(idx_sub+1)+'.'+str(idx_run+1)+'.npy')
            for idx_chn in range(8):
                signal=X[0+idx_label*2,idx_chn,:]
                fft_result += np.abs(np.fft.fft(signal))[:len(freqs)//2]
                signal=X[1+idx_label*2,idx_chn,:]
                fft_result += np.abs(np.fft.fft(signal))[:len(freqs)//2]
        
    plt.plot(freqs[:len(freqs)//2], np.abs(fft_result)[:len(freqs)//2])
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(5, 60)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# 绘制原始信号
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Original Signal')

# 绘制频谱
plt.subplot(2, 1, 2)
