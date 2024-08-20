import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
import mne
def preprocess_data(raw_data, l_freq=8.0, h_freq=55.0):
    # 50 Hz陷波滤波
    raw_data.notch_filter(50, picks='eeg', method='fir', fir_window='hamming', fir_design='firwin')
    
    # 带通滤波
    raw_data.filter(l_freq, h_freq, l_trans_bandwidth=2, h_trans_bandwidth=5, phase='zero-double')
    
    # 参考信号均值重参考
    raw_data_rereferenced, _ = mne.set_eeg_reference(raw_data, ref_channels='average', copy=True)
    return raw_data_rereferenced
def create_epochs(raw_data, events, event_id, tmin, tmax, baseline):
    epochs = mne.Epochs(raw_data, events, event_id, tmin, tmax, baseline, preload=True)
    return epochs
# 注意：你需要根据你的实际FIF文件路径来设置文件名
fif_file =r'C:\Users\陈禹\Desktop\robot game in the world\4y1c\data\ljh\combine\combined_data.fif'
# 读取FIF文件
raw = mne.io.read_raw_fif(fif_file, preload=True)
# 选取特定通道
ch_names = raw.ch_names
channels = ['TP10', 'O2', 'OZ', 'O1', 'POZ', 'PZ', 'TP9', 'FCZ']
ch_idx =[ [ch_names.index(ch) for ch in channels if ch in ch_names]]
# 获取数据
# data = raw.get_data()
data = preprocess_data(raw)
# data = data[:-1, :]
# data = np.reshape(data, (, 1000, 6))

# 读取事件信息
events = mne.find_events(data)
event_id =  {'9': 9, '10': 10, '11': 11, '12': 12} # 根据实际数据调整事件ID
tmin = -0.2  # 事件开始时间（相对于事件触发时间）
tmax = 0.8  # 事件结束时间

# 创建Epochs
epochs = create_epochs(data, events, event_id, tmin, tmax,baseline=(None, 0))
# 提取时间和频率信息
sfreq = raw.info['sfreq']
t =np.arange(0, 1, 0.001)
# t_idx = np.where((t > 0.14) & (t < 2))[0]
# t = t[t_idx]
# data = data[ch_idx, t_idx]
data = epochs.get_data()
data = data[:, :-1,:-1]
# 设置频率参数
freq_bin = np.arange(9, 13, 1)  # 假设的四个频率

# 初始化相关矩阵
corr = np.zeros((24, 4))  # 每个试验有4个频率和4个块
cca = CCA(n_components=2)

# 处理每个试验
result = np.zeros((24, 4))
for idx_block in range(1):
    for idx_trial in range(24):
        X=data[idx_trial,:,:].T
        for idx_freq in range(4):
            # 假设每个块大小相同，且每个块的开始和结束是连续的

            Y = np.zeros((len(t), 6))  # 对于每个频率，只需要正弦和余弦两个基函数
            Y[:, 0] = np.sin(2 * np.pi * freq_bin[idx_freq] * t)
            Y[:, 1] = np.cos(2 * np.pi * freq_bin[idx_freq] * t)
            Y[:,2] = np.sin(2 * np.pi * 2 * freq_bin[idx_freq] * t) 
            Y[:,3] = np.cos(2 * np.pi * 2 * freq_bin[idx_freq] * t)     
            Y[:,4] = np.sin(2 * np.pi * 3 * freq_bin[idx_freq] * t) 
            Y[:,5] = np.cos(2 * np.pi * 3 * freq_bin[idx_freq] * t) 

            cca.fit(X, Y)
            corr[idx_trial, idx_freq] = cca.score(X, Y)

    # 假设正确的频率对应于最大的相关系数
    # correct_freq_idx = np.argmax(corr[idx_trial * 6:(idx_trial + 1) * 6, :], axis=1)
    # result[idx_trial :(idx_trial + 1) ] = (correct_freq_idx == np.arange(4)).astype(int)
    for idx_trial in range(24):
        result[idx_trial,:]=np.mean(corr[idx_trial:(idx_trial+1),:], axis=0)
    
print(np.sum(result) / (4 * 6) * 100)

# plt.plot(result)
# plt.show()
