import numpy as np
import matplotlib.pyplot as plt
#计算wyj fft
# 文件路径
file_path = '..\\data\\data2\\X_data_subject_[1].npy'

# 读取数据
data = np.load(file_path)

# 定义采样率
sampling_rate = 250  # 单位为 Hz

# 信号长度
n_timepoints = data.shape[2]  # 假设最后一个维度是时间点
time_interval = 1 / sampling_rate  # 时间间隔，单位为秒

# 创建时间向量
time = np.arange(0, n_timepoints * time_interval, time_interval)

# 频率向量
freqs = np.fft.fftfreq(n_timepoints, d=time_interval)
positive_freqs = freqs[:len(freqs) // 2]  # 我们只关心正频率

# 循环遍历前40组信号
for idx_label in range(40):
    fft_result = np.zeros((len(time) // 2,))  # 存储FFT结果
    
    # 对每组信号的所有通道进行FFT
    for idx_chn in range(data.shape[1]):
        signal = data[idx_label, idx_chn, :]
        fft_result += np.abs(np.fft.fft(signal))[:len(positive_freqs)]
    
    # 计算平均值
    fft_result /= data.shape[1]
    
    # 绘制频率谱图
    plt.figure(figsize=(10, 5))
    plt.plot(positive_freqs, np.abs(fft_result))
    plt.title('Frequency Spectrum - Sample {}'.format(idx_label + 1))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(5, 20)
    plt.grid(True)
    plt.tight_layout()
    plt.show()