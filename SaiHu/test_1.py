import numpy as np
import matplotlib.pyplot as plt

#用于查看自采数据fft效果
t = np.arange(0, 5, 0.001)
freqs = np.fft.fftfreq(len(t), t[1] - t[0])
for idx_label in range(20):
    fft_result=np.zeros((500*5,))
    for idx_sub in range(1):
        for idx_run in range(1):
            X = np.load('..\data\data1\X_data_subject_'+str(idx_sub+17)+'.'+str(idx_run+1)+'.npy')
            for idx_chn in range(8):
                signal=np.zeros((5000,))
                signal[0:1000]=X[0+idx_label*2,idx_chn,:]
                fft_result += np.abs(np.fft.fft(signal))[:len(freqs)//2]
                signal[0:1000]=X[1+idx_label*2,idx_chn,:]
                fft_result += np.abs(np.fft.fft(signal))[:len(freqs)//2]
        
    plt.plot(freqs[:len(freqs)//2], np.abs(fft_result)[:len(freqs)//2])
    plt.plot(freqs[int(idx_label*2+40+10)],np.abs(fft_result)[int(idx_label*2+40+10)],'*')
    plt.title('Frequency Spectrum '+str(idx_label)+' '+str(idx_label*0.4+8)+'Hz')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(5, 30)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# # 绘制原始信号
# plt.subplot(2, 1, 1)
# plt.plot(t, signal)
# plt.title('Original Signal')

# # 绘制频谱
# plt.subplot(2, 1, 2)


'''# 定义信号和噪声频率范围
signal_freq = 15  # Hz
noise_bandwidth = 2  # Hz

# 用于查看自采数据FFT效果
for idx_label in range(20):
    t = np.arange(0, 1, 0.001)
    fft_result = np.zeros((500,))
    freqs = np.fft.fftfreq(len(t), t[1] - t[0])

    # 初始化变量存储信号和噪声的功率
    signal_power = 0
    noise_power = 0
    noise_count = 0

    for idx_sub in range(1):
        for idx_run in range(1):
            X = np.load('..\data\data1\X_data_subject_' + str(idx_sub + 18) + '.' + str(idx_run + 4) + '.npy')
            for idx_chn in range(8):
                signal = X[0 + idx_label * 2, idx_chn, :]
                fft_result += np.abs(np.fft.fft(signal))[:len(freqs)//2]
                signal = X[1 + idx_label * 2, idx_chn, :]
                fft_result += np.abs(np.fft.fft(signal))[:len(freqs)//2]

                # 计算信号和噪声的功率
                if abs(freqs[freqs // 2:]) <= (signal_freq + noise_bandwidth / 2) and abs(freqs[freqs // 2:]) >= (signal_freq - noise_bandwidth / 2):
                    if abs(freqs[freqs // 2:]) == signal_freq:
                        signal_power += np.abs(np.fft.fft(signal))[:len(freqs)//2][freqs[freqs // 2:] == signal_freq][0] ** 2
                    else:
                        noise_power += np.abs(np.fft.fft(signal))[:len(freqs)//2][freqs[freqs // 2:] == abs(freqs[freqs // 2:])]
                        noise_count += 1

    # 计算SNR
    snr = signal_power / (noise_power / noise_count)

    plt.figure(figsize=(10, 6))

    # 绘制频谱
    plt.subplot(2, 1, 1)
    plt.plot(freqs[:len(freqs)//2], np.abs(fft_result)[:len(freqs)//2])
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(5, 20)
    plt.grid(True)
    plt.tight_layout()

    # 绘制原始信号
    plt.subplot(2, 1, 2)
    plt.plot(t, signal)
    plt.title('Original Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # 显示SNR
    plt.text(0.05, 0.95, f'SNR: {snr:.2f} dB', transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.show()'''