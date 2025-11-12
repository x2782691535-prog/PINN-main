import mne
import numpy as np
import os
import scipy.io as sio
from scipy.io import loadmat

# 设置数据路径
data_path = 'E:/pycharm/EEG-ATCNet/BCICIV_2a_gdf'
save_path = 'E:/pycharm/EEG-ATCNet/preprocess/BCI-IV-2a'

# 数据预处理
for i in range(1, 10):  # 数据集包含 9 个参与者
    # 读取 GDF 文件（例如：A01T.gdf 和 A01E.gdf）
    raw_file = os.path.join(data_path, f'A0{i}T.mat')  # 训练数据
    # raw = mne.io.read_raw_gdf(raw_file, preload=True)
    raw_data = loadmat(raw_file)
    raw = raw_data['data']
    # 读取标注文件（例如：A01T.mat 或 A01E.mat）
    label_file = os.path.join(data_path, f'true_labels/A0{i}T.mat')
    labels_data = loadmat(label_file)  # 使用 loadmat 加载 .mat 文件
    labels = labels_data['classlabel']  # 获取标签数据，注意根据 .mat 文件的结构，标签的键名可能不同

    # 设置通道信息和参考信息（BCI 数据集通常是参考于耳朵）
    raw.set_eeg_reference(ref_channels='average', projection=True)

    # 低通和高通滤波
    raw.filter(l_freq=1.0, h_freq=50.0)  # 高频噪声移除，低频噪声移除

    # 设置事件（使用对应的标注，类似 769, 770, 771 等）
    events, _ = mne.events_from_annotations(raw)

    # 将数据分段（Epoching），例如根据刺激的时间进行切割
    event_id = {'left_hand': 1, 'right_hand': 2, 'foot': 3, 'tongue': 4}  # 设置具体的事件ID
    epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=4, baseline=None, detrend=1, reject_by_annotation=True, preload=True, event_repeated='merge')

    # ICA 去伪迹（例如去眼动伪迹）
    ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
    ica.fit(epochs)

    # 使用 find_bads_eog 来检测 EOG 伪迹并进行去除
    eog_indices, scores = ica.find_bads_eog(epochs, ch_name='EOG-left')  # 你也可以换成 'EOG-central' 或 'EOG-right'
    ica.exclude = eog_indices  # 排除被标记为 EOG 伪迹的组件

    # 将数据与 ICA 伪迹去除应用于数据
    epochs_clean = ica.apply(epochs)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 保存预处理后的数据
    processed_file = os.path.join(save_path, f'processed_A0{i}T.mat')
    data_array = epochs_clean.get_data()
    sio.savemat(processed_file, {'data': data_array})

    # 保存事件标签
    label_file = os.path.join(save_path, f'processed_A0{i}T_labels.mat')
    sio.savemat(label_file, {'classlabel': labels})

    print(f"Processing of subject A0{i} completed!")
