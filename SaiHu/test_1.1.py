# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 19:14:24 2024

@author: husai
"""
import numpy as np
import matplotlib.pyplot as plt

# 用于查看自采数据fft效果
t = np.arange(0, 5, 0.001)
freqs = np.fft.fftfreq(len(t), t[1] - t[0])

# 初始化fft_result数组，以便累积所有idx_label的结果
fft_result = np.zeros((500,))

for idx_label in range(20):
    # 由于fft_result已在外层循环初始化，这里不再重新初始化
    
    for idx_sub in range(1):  # 子试次
        for idx_run in range(3):  # 每个子试次有3个run
            X = np.load('..\data\data1\X_data_subject_' + str(idx_sub + 16) + '.' + str(idx_run + 1) + '.npy')
            
            for idx_chn in range(8):  # 每个run有8个通道
                signal = np.zeros((1000,))
                
                # 加载第一个信号段
                signal[0:1000] = X[0 + idx_label * 2, idx_chn, :]
                fft_result += np.abs(np.fft.fft(signal))[:len(freqs)//2]
                
                # 加载第二个信号段
                signal[0:1000] = X[1 + idx_label * 2, idx_chn, :]
                fft_result += np.abs(np.fft.fft(signal))[:len(freqs)//2]

    # 绘图
    plt.plot(freqs[:len(freqs)//2], np.abs(fft_result)[:len(freqs)//2])
    plt.plot(freqs[int(idx_label*2+40+10)], np.abs(fft_result)[int(idx_label*2+40+10)], '*')
    plt.title('Frequency Spectrum ' + str(idx_label) + ' ' + str(idx_label*0.4+8) + 'Hz')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(5, 30)
    plt.grid(True)
    plt.tight_layout()
    plt.show()