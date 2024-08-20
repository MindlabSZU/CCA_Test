# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 19:38:41 2024

@author: husai
"""
import numpy as np
import matplotlib.pyplot as plt
def average_reference(data):
    # 计算所有电极的平均电位
    average_potential = np.mean(data, axis=1, keepdims=True)
    
    # 从每个电极的数据中减去平均电位
    re_referenced_data = data - average_potential
    
    return re_referenced_data
def re_reference(data, reference_channel_index):
    # 对每个样本进行重参考
    re_referenced_data = data - data[:, reference_channel_index, np.newaxis]
    return re_referenced_data

# 假设我们要使用Oz电极作为参考电极
reference_channel_index = 2
#原自采fft
# 创建时间轴和频率轴
t = np.arange(0, 1, 0.001)
freqs = np.fft.fftfreq(len(t), t[1] - t[0])

# 绘制原始信号及其频谱的准备
plt.figure(figsize=(10, 8))

for idx_label in range(20):
    # 初始化fft_result数组
    fft_result = np.zeros((500,))
    
    for idx_sub in range(1):  # 子试次
        for idx_run in range(5):  # 每个子试次有2个run
            X = np.load('..\data\data1\X_data_subject_' + str(idx_sub + 1) + '.' + str(idx_run + 1) + '.npy')
            '''X = average_reference(X)'''
            X = re_reference(X, reference_channel_index)
            for idx_chn in range(8):  # 每个run有8个通道
                # 保存原始信号
                signal = X[0 + idx_label * 2, idx_chn, :]
                # 计算FFT
                fft_result += np.abs(np.fft.fft(signal))[:len(freqs)//2]
                
                # 处理第二个信号
                signal = X[1 + idx_label * 2, idx_chn, :]
                fft_result += np.abs(np.fft.fft(signal))[:len(freqs)//2]
    
    # 绘制频谱
    plt.plot(freqs[:len(freqs)//2], np.abs(fft_result)[:len(freqs)//2])
    plt.title('Frequency Spectrum'+''+str(idx_label*0.4+8)+'Hz')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(5, 60)
    plt.grid(True)

    # 绘制原始信号


    plt.tight_layout()
    plt.show()

