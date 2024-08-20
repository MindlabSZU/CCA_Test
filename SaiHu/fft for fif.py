# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 21:43:20 2024

@author: husai
"""
import mne
import numpy as np
import matplotlib.pyplot as plt

# 读取.fif文件
def read_fif_file(file_path):
    return mne.io.read_raw_fif(file_path, preload=True)

# 执行FFT变换
def fft_transform(data, sfreq):
    fft_data = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(fft_data), 1/sfreq)
    return fft_data, freqs

# 绘制频域图
def plot_frequency_spectrum(fft_data_list, freqs, channel_names, epoch_index):
    # 平均所有通道的幅度谱
    avg_amplitude_spectrum = np.mean(np.abs(fft_data_list), axis=0)
    
    plt.figure(figsize=(12, 8))
    plt.plot(freqs, avg_amplitude_spectrum)
    plt.title(f'Average Frequency Spectrum, Epoch {epoch_index}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    
    plt.xlim(7, 40)  # 仅绘制7 Hz到40 Hz之间的频率
    plt.ylim(0, np.max(avg_amplitude_spectrum) * 1.1)  # 设置y轴范围，显示最大振幅的10%
    
    plt.grid(True)
    plt.show()

# 数据预处理
'''def preprocess_data(raw_data, l_freq=7.0, h_freq=55.0):
    # 50 Hz陷波滤波
    raw_data.notch_filter(50, picks='eeg', method='fir', fir_window='hamming', fir_design='firwin')
    
    # 带通滤波
    raw_data.filter(l_freq, h_freq, l_trans_bandwidth=2, h_trans_bandwidth=5, phase='zero-double')
    
    # 参考信号均值重参考
    raw_data_rereferenced, _ = mne.set_eeg_reference(raw_data, ref_channels='average', copy=True)
    return raw_data_rereferenced'''
# 数据预处理
# 数据预处理
# 数据预处理
def preprocess_data(raw_data, l_freq=7.0, h_freq=55.0):
    # 50 Hz陷波滤波
    raw_data.notch_filter(50, picks='eeg', method='fir', fir_window='hamming', fir_design='firwin')
    
    # 带通滤波
    raw_data.filter(l_freq, h_freq, l_trans_bandwidth=2, h_trans_bandwidth=5, phase='zero-double')
    
    # 使用FCz作为参考电极进行重参考
    # 检查FCz是否存在于数据中
    if 'FCZ' in raw_data.ch_names:
        # 使用FCz作为参考电极进行重参考
        raw_data_rereferenced, _ = mne.set_eeg_reference(raw_data, ref_channels=['FCZ'], copy=True)
    else:
        # 如果不存在FCz，则抛出异常
        raise ValueError("Channel 'FCz' not found in the data. Please check the data or use another reference.")
    
    return raw_data_rereferenced
# 创建Epochs
def create_epochs(raw_data, events, event_id, tmin, tmax, baseline):
    epochs = mne.Epochs(raw_data, events, event_id, tmin, tmax, baseline, preload=True)
    return epochs

# 主程序
if __name__ == '__main__':
    file_path = r'C:\Users\husai\Desktop\metabci\MetaBCI\MetaBCI\data\test\fif\muti\6\17.1.fif'  # 替换为您的.fif文件路径
    raw_data = read_fif_file(file_path)
     
    # 打印通道名称
    print("Channel names in the data:")
    print(raw_data.ch_names)
    # 预处理数据
    raw_data = preprocess_data(raw_data)
    
    # 读取事件信息
    events = mne.find_events(raw_data)
    event_id =  {
        '1': 8,
        '2': 8,
        '3': 8,
        '4': 8
    }  # 根据实际数据调整事件ID
    tmin = -0.2  # 事件开始时间（相对于事件触发时间）
    tmax = 1  # 事件结束时间
    
    # 创建Epochs
    epochs = create_epochs(raw_data, events, event_id, tmin, tmax, baseline=(None, 0))
    
    # 选择前8个通道
    channels_to_use = epochs.ch_names[:8]
    
    # 获取采样频率
    sfreq = epochs.info['sfreq']
    
    # 遍历每个epoch并执行FFT
    for epoch_index, epoch_data in enumerate(epochs.get_data()):
        # 存储每个通道的FFT数据
        fft_data_list = []
        
        # 遍历每个通道
        for channel_index, channel_name in enumerate(channels_to_use):
            channel_data = epoch_data[channel_index]
            fft_data, freqs = fft_transform(channel_data, sfreq)
            fft_data_list.append(fft_data)
        
        # 调用修改后的plot_frequency_spectrum函数
        plot_frequency_spectrum(fft_data_list, freqs, channels_to_use, epoch_index + 1)