# 使用esinet预测BCI-IV-2a数据集的源定位分析
# 本教程使用BCI-IV-2a运动想象EEG数据集

import mne
import numpy as np
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, '../')
from esinet import util
from esinet import Simulation
from esinet import Net
from esinet.evaluate.evaluate import eval_mean_localization_error, eval_mse, eval_auc
import os
import re
import scipy.io
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import time
import random
import pickle
from joblib import Parallel, delayed
import scipy.signal

# # 固定随机种子，保证仿真数据可复现
# np.random.seed(42)
# random.seed(42)

# 🚀 GPU配置 - 强制使用GPU训练
print("🔧 配置GPU环境...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 设置GPU内存增长
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ 检测到 {len(gpus)} 个GPU设备")
        for i, gpu in enumerate(gpus):
            gpu_details = tf.config.experimental.get_device_details(gpu)
            print(f"  GPU {i}: {gpu_details.get('device_name', 'Unknown')}")
        
        # 设置默认GPU设备
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"🎯 将使用GPU进行训练: {gpus[0]}")
    except RuntimeError as e:
        print(f"⚠️  GPU配置失败: {e}")

# 设置TensorFlow日志级别
tf.get_logger().setLevel('ERROR')

# 设置matplotlib字体为微软黑体，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 添加一个函数来设置MNE图形的中文字体
def set_chinese_font_for_figure(fig):
    """为MNE生成的图形设置中文字体支持"""
    for ax in fig.get_axes():
        for text in ax.texts:
            text.set_fontproperties('Microsoft YaHei')
        ax.set_title(ax.get_title(), fontproperties='Microsoft YaHei')
        ax.set_xlabel(ax.get_xlabel(), fontproperties='Microsoft YaHei')
        ax.set_ylabel(ax.get_ylabel(), fontproperties='Microsoft YaHei')
    
    # 设置图形标题
    if hasattr(fig, '_suptitle') and fig._suptitle is not None:
        fig._suptitle.set_fontproperties('Microsoft YaHei')

# 添加一个自定义add_text方法，使用中文字体
def add_text_with_chinese(brain, x, y, text, text_type, font_size=14, font_family='Microsoft YaHei'):
    """使用中文字体添加文本到Brain对象"""
    try:
        # 使用带有中文字体的方法
        brain.add_text(x, y, text, text_type, font_size=font_size, font_family=font_family)
    except TypeError:
        # 如果方法不接受font_family参数，使用原始方法
        brain.add_text(x, y, text, text_type, font_size=font_size)
        print(f"警告: 无法为3D可视化设置中文字体'{font_family}'，使用默认字体")

# BCI-IV-2a电极位置定义 (22通道，符合国际10-20系统)
def get_bci_iv_2a_channel_names():
    """获取BCI-IV-2a数据集的电极名称"""
    # BCI-IV-2a数据集使用的22个EEG电极位置
    ch_names = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
    ]
    return ch_names

def load_bci_iv_2a_data(data_path, subject_ids=None):
    """
    加载BCI-IV-2a数据集
    
    Parameters
    ----------
    data_path : str
        BCI-IV-2a数据集的路径
    subject_ids : list or None
        要加载的被试ID列表，如果为None则加载所有被试
    
    Returns
    -------
    all_epochs : list
        所有被试的epochs数据列表
    """
    if subject_ids is None:
        # 自动检测所有mat文件
        mat_files = [f for f in os.listdir(data_path) if f.endswith('.mat')]
        print(f"检测到 {len(mat_files)} 个数据文件: {mat_files}")
    else:
        mat_files = [f"subject_{sid}.mat" for sid in subject_ids]
    
    all_epochs = []
    ch_names = get_bci_iv_2a_channel_names()
    
    for mat_file in mat_files:
        file_path = os.path.join(data_path, mat_file)
        if not os.path.exists(file_path):
            print(f"警告: 文件 {file_path} 不存在，跳过")
            continue
        
        # 加载MATLAB文件
        try:
            mat_data = scipy.io.loadmat(file_path)
        except Exception as e:
            print(f"加载文件 {mat_file} 失败: {e}")
            continue
        
        # 提取数据 - 根据你提供的格式 rawdata(1000×22×576) 和 label(576×1)
        if 'rawdata' in mat_data and 'label' in mat_data:
            raw_data = mat_data['rawdata']  # (1000, 22, 576)
            labels = mat_data['label'].flatten()  # (576,)
        else:
            # 如果键名不同，尝试找到数据
            # print(f"数据文件 {mat_file} 中的键: {list(mat_data.keys())}")
            data_key = None
            label_key = None
            for key, value in mat_data.items():
                if isinstance(value, np.ndarray):
                    if value.shape == (1000, 22, 576):
                        data_key = key
                    elif value.shape == (576, 1) or value.shape == (576,):
                        label_key = key
            if data_key is None or label_key is None:
                print(f"无法在 {mat_file} 中找到正确格式的数据，跳过")
                continue
            raw_data = mat_data[data_key]  # (1000, 22, 576)
            labels = mat_data[label_key].flatten()  # (576,)
        # 转换数据格式: (采样点, 通道, 试次) -> (试次, 通道, 采样点)
        eeg_data = np.transpose(raw_data, (2, 1, 0))  # (576, 22, 1000)
        sfreq = 250  # BCI-IV-2a的采样率通常是250Hz
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        montage = mne.channels.make_standard_montage('standard_1020')
        info.set_montage(montage)
        valid_trials = labels != 0
        valid_eeg_data = eeg_data[valid_trials]
        valid_labels = labels[valid_trials]
        lr_trials = np.isin(valid_labels, [1, 2])
        final_eeg_data = valid_eeg_data[lr_trials]
        final_labels = valid_labels[lr_trials]
        n_final_trials = len(final_eeg_data)
        events = np.zeros((n_final_trials, 3), dtype=int)
        events[:, 0] = np.arange(n_final_trials) * 1000  # 时间戳
        events[:, 2] = final_labels.astype(int)  # 事件类型
        unique_labels = np.unique(final_labels.astype(int))
        if len(unique_labels) >= 2:
            if 1 in unique_labels and 2 in unique_labels:
                event_id = {'left_hand': 1, 'right_hand': 2}
            else:
                event_id = {f'condition_{i+1}': label for i, label in enumerate(unique_labels)}
        else:
            print(f"警告: 只检测到一种标签 {unique_labels}，跳过此文件")
            continue
        # 保留一份未标准化的原始数据用于可视化
        eeg_data_for_plot = final_eeg_data.copy()
        # 标准化数据（仅用于后续分析/建模）
        for trial in range(final_eeg_data.shape[0]):
            for ch in range(final_eeg_data.shape[1]):
                final_eeg_data[trial, ch, :] = (final_eeg_data[trial, ch, :] - np.mean(final_eeg_data[trial, ch, :])) / np.std(final_eeg_data[trial, ch, :])
        tmin = 0.0
        tmax = (final_eeg_data.shape[2] - 1) / sfreq
        # 创建标准化数据的Epochs对象
        epochs = mne.EpochsArray(
            final_eeg_data, info, events=events, event_id=event_id,
            tmin=tmin, verbose=False
        )
        # 创建未标准化数据的Epochs对象用于可视化
        epochs_plot = mne.EpochsArray(
            eeg_data_for_plot, info, events=events, event_id=event_id,
            tmin=tmin, verbose=False
        )
        all_epochs.append((epochs, epochs_plot))
    return all_epochs

# 设置绘图参数
plot_params = dict(surface='inflated', cortex="low_contrast", hemi='both', verbose=0)

# 加载BCI-IV-2a数据
print("加载BCI-IV-2a运动想象EEG数据集...")

# BCI-IV-2a数据路径
bci_data_path = r"E:\pycharm\PINN\cursor-gPINN\BCI2a"

# 加载数据
epochs_list = load_bci_iv_2a_data(bci_data_path)

# 使用第三个被试的数据进行演示
epochs, epochs_plot = epochs_list[2]
print(f"使用第三个被试的数据进行分析，包含 {len(epochs)} 个试次")

# 打印通道信息
print(f"数据集中的EEG通道: {len(epochs.ch_names)}")
print(f"所有EEG通道: {epochs.ch_names}")
print(f"事件类型: {list(epochs.event_id.keys())}")

# ====== 图像保存相关 ======
FIG_DIR = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)

def save_figure(fig, filename):
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"已保存图片: {path}")

def save_brain_screenshot(brain, filename):
    path = os.path.join(FIG_DIR, filename)
    img = brain.screenshot()
    import matplotlib.pyplot as plt
    plt.imsave(path, img)
    print(f"已保存3D脑图截图: {path}")
# =========================

# 1. 原始数据可视化 - 左右手各自第一个样本的未平均波形（仅显示运动相关通道）
print("正在显示：左右手各自第一个样本的未平均EEG波形（运动相关通道）")

# 定义与左右手运动相关的通道
motor_channels = ['C3', 'Cz', 'C4']

# 分别处理左手和右手的第一个样本
for hand_label, hand_name in zip(['left_hand', 'right_hand'], ['左手', '右手']):
    hand_epochs = epochs_plot[hand_label]
    if len(hand_epochs) > 0:
        # 获取第一个样本的数据
        first_sample_data = hand_epochs[0].get_data()[0]  # shape: (n_channels, n_times)
        # 创建一个伪evoked对象来绘制单个样本
        first_sample_evoked = mne.EvokedArray(first_sample_data, hand_epochs.info, tmin=hand_epochs.times[0])
        
        # 选择运动相关通道
        # 找到可用的运动相关通道
        available_motor_channels = [ch for ch in motor_channels if ch in hand_epochs.ch_names]
        if available_motor_channels:
            motor_evoked = first_sample_evoked.pick_channels(available_motor_channels, ordered=True)
            fig_raw_evoked = motor_evoked.plot(verbose=0, show=False)
            fig_raw_evoked.suptitle(f'原始EEG波形: {hand_name}第一个样本（未平均，运动相关通道: {available_motor_channels}）')
            # 添加图例显示通道和颜色对应关系
            axes = fig_raw_evoked.get_axes()
            if len(axes) > 0:
                axes[0].legend(available_motor_channels, loc='upper right', fontsize=8)
            set_chinese_font_for_figure(fig_raw_evoked)
            save_figure(fig_raw_evoked, f'raw_{hand_label}_first_sample_motor_channels_evoked.png')
            plt.show()
        else:
            print(f"错误: 没有找到可用的运动相关通道")
    else:
        print(f"警告: {hand_name}数据中没有样本")

# 添加平均波形图显示（仅显示C3、C4、Cz通道）
print("正在显示：左右手平均EEG波形和头皮地形图（C3、C4、Cz通道）")
central_channels = ['C3', 'Cz', 'C4']
plot_times = [0.5, 1.0, 2.0, 3.0]

# 分别处理左手和右手的平均波形和地形图
for hand_label, hand_name in zip(['left_hand', 'right_hand'], ['左手', '右手']):
    hand_epochs = epochs_plot[hand_label]
    if len(hand_epochs) > 0:
        # 获取平均波形
        averaged_evoked = hand_epochs.average()
        
        # --- 1. 绘制平均波形图 ---
        # 选择中央运动皮层通道
        available_central_channels = [ch for ch in central_channels if ch in hand_epochs.ch_names]
        if available_central_channels:
            central_evoked = averaged_evoked.pick_channels(available_central_channels, ordered=True)
            fig_avg_evoked = central_evoked.plot(verbose=0, show=False)
            fig_avg_evoked.suptitle(f'平均EEG波形: {hand_name}（{available_central_channels}通道）')
            # 添加图例显示通道和颜色对应关系
            axes = fig_avg_evoked.get_axes()
            if len(axes) > 0:
                axes[0].legend(available_central_channels, loc='upper right', fontsize=8)
            set_chinese_font_for_figure(fig_avg_evoked)
            save_figure(fig_avg_evoked, f'raw_{hand_label}_averaged_central_channels_evoked.png')
            plt.show()

            # --- 2. 绘制头皮地形图 ---
            # 获取有效时间点
            valid_times = [t for t in plot_times if averaged_evoked.times[0] <= t <= averaged_evoked.times[-1]]
            if len(valid_times) < len(plot_times):
                print(f"警告: 部分时间点超出 {hand_name} 信号范围，仅绘制有效时间点: {valid_times}")

            fig_raw_topomap = central_evoked.plot_topomap(times=valid_times, time_format='%0.1f s')
            fig_raw_topomap.suptitle(f'原始EEG头皮地形图（{available_central_channels}通道）- {hand_name}')
            set_chinese_font_for_figure(fig_raw_topomap)
            save_figure(fig_raw_topomap, f'raw_{hand_label}_topomap_central_channels.png')
            plt.show()
            plt.close()

        else:
            print(f"错误: 没有找到可用的中央通道")
    else:
        print(f"警告: {hand_name}数据中没有样本")

# ===================================================================
# ERD/ERS现象可视化 (根据 plotERP1.m 逻辑)
# ===================================================================
print("\n===== 正在根据 plotERP1.m 逻辑和S3.mat数据进行ERD/ERS可视化 =====")

# --- 1. 参数定义 ---
mat_path = 'E:/pycharm/PINN/esinet-main/data/S8.mat'  # 使用正斜杠以保证路径兼容性
fs = 250  # 采样率
filter_band = [10, 12]  # Hz, 滤波频段
filter_order = 6  # 滤波器阶数
smooth_window = 100 # 平滑窗口

# 定义MATLAB脚本中的基线周期
baseline_period_matlab = (-2.5, -1.5) # s, 这是plotERP1.m中一个关键且特殊的设定

# --- 2. 加载和预处理数据 ---
if not os.path.exists(mat_path):
    print(f"错误: 未在以下路径找到数据文件: {mat_path}")
    print("请将S3.mat文件放置到 'data' 文件夹下。跳过此可视化部分。")
else:
    print(f"正在从 {mat_path} 加载数据...")
    mat_data = scipy.io.loadmat(mat_path)
    x_train_mat = mat_data['x_train']  # shape: (samples, channels, trials)
    y_train = mat_data['y_train'].flatten()

    print("正在对每个试次进行Z-score标准化...")
    x_norm = np.zeros_like(x_train_mat, dtype=float)
    for i in range(x_train_mat.shape[2]):
        trial = x_train_mat[:, :, i]
        mean_val = np.mean(trial)
        std_val = np.std(trial)
        x_norm[:, :, i] = (trial - mean_val) / std_val if std_val > 0 else trial - mean_val

    # --- 3. ERD/ERS 计算与绘图 ---
    n_samples = x_norm.shape[0]
    n_channels = x_norm.shape[1]

    # 根据MATLAB绘图脚本的刻度重新定义时间向量 (t=0 at sample 750)
    time_vec = (np.arange(n_samples) - 750) / fs
    print(f"数据时间范围: {time_vec[0]:.2f}s to {time_vec[-1]:.2f}s")
    
    # 找到MATLAB脚本定义的基线周期的样本索引
    baseline_start_idx = np.abs(time_vec - baseline_period_matlab[0]).argmin()
    baseline_end_idx = np.abs(time_vec - baseline_period_matlab[1]).argmin()
    print(f"使用基线周期: {baseline_period_matlab} s (样本索引: {baseline_start_idx} 到 {baseline_end_idx})")
    
    # --- 信号处理 ---
    # a. 带通滤波 (按试次进行)
    b, a = scipy.signal.butter(filter_order, [f / (fs / 2) for f in filter_band], btype='bandpass')
    filtered_data = np.zeros_like(x_norm)
    for i in range(x_norm.shape[2]):
        filtered_data[:, :, i] = scipy.signal.filtfilt(b, a, x_norm[:, :, i], axis=0)

    # b. 计算功率 (平方)
    power_data = np.square(filtered_data)

    # c. 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False) # 不共享Y轴以匹配MATLAB行为
    colors = {'C3': 'r', 'C4': 'b', 'Cz': 'g'}
    # 假设S3.mat的3个通道顺序是 C3, C4, Cz
    ch_map = {0: 'C3', 1: 'C4', 2: 'Cz'}

    for ax, (label_val, hand_name) in zip(axes, zip([1, 2], ['左手', '右手'])):
        print(f"\n--- 正在处理: {hand_name} ---")
        
        # 选择当前任务的试次的功率数据
        hand_power_trials = power_data[:, :, y_train == label_val]
        
        # d. 在试次上平均
        avg_power = np.mean(hand_power_trials, axis=2) # shape: (samples, channels)
        
        # e. 平滑
        smoothed_power = np.apply_along_axis(
            lambda x: np.convolve(x, np.ones(smooth_window)/smooth_window, mode='same'),
            axis=0, arr=avg_power
        )
        
        # f. 基线校正 (A-R)/R * 100
        baseline_power = np.mean(smoothed_power[baseline_start_idx:baseline_end_idx, :], axis=0)
        baseline_power[baseline_power == 0] = 1e-9 # 避免除以零
        
        erd_ers_percent = ((smoothed_power - baseline_power) / baseline_power) * 100

        # g. 绘图
        for i in range(n_channels):
            ch_name = ch_map.get(i, f'Ch{i+1}')
            ax.plot(time_vec, erd_ers_percent[:, i], label=ch_name, color=colors.get(ch_name))

        ax.set_title(hand_name)
        ax.set_xlabel('时间/s')
        ax.axvline(0, linestyle='--', color='k', linewidth=0.8, label='Cue')
        ax.axhline(0, linestyle='-', color='k', linewidth=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-3, 4)

    axes[0].set_ylabel('ERD/ERS (%)')
    fig.suptitle(f'μ频段 ({filter_band[0]}-{filter_band[1]} Hz) 的ERD/ERS (根据plotERP1.m逻辑)')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_figure(fig, 'erd_ers_S3_plotERP1_style.png')
    plt.show()

# 获取样本数据集的subjects_dir
print("设置FreeSurfer样本数据集路径...")
subjects_dir = str(mne.datasets.sample.data_path() / 'subjects')
subject = 'fsaverage'  # 使用fsaverage样本主题

# 创建适合BCI-IV-2a数据的前向模型
print("创建BCI-IV-2a数据的前向模型...")

# 使用fsaverage样本数据中的MRI创建前向模型
# 首先设置源空间 - 使用预先计算的样本主题的源空间
src_fname = os.path.join(subjects_dir, subject, 'bem', f'{subject}-oct-6-src.fif')
if not os.path.exists(src_fname):
    # 如果预计算源空间不存在，则创建一个
    src = mne.setup_source_space(subject, spacing='oct3', 
                                subjects_dir=subjects_dir,
                                add_dist=False, verbose=True)
    mne.write_source_spaces(src_fname, src, overwrite=True)
else:
    # 加载预计算的源空间
    src = mne.read_source_spaces(src_fname, verbose=True)

# 加载适合EEG的BEM模型
bem_fname = os.path.join(subjects_dir, subject, 'bem', f'{subject}-5120-5120-5120-bem-sol.fif')
bem = mne.read_bem_solution(bem_fname, verbose=True)

# 创建坐标变换 - 使用标准变换文件
trans_fname = os.path.join(subjects_dir, subject, 'bem', f'{subject}-trans.fif')
trans = mne.read_trans(trans_fname, verbose=True)

# 创建前向模型
fwd = mne.make_forward_solution(
    epochs.info, trans=trans, src=src, bem=bem,
    meg=False, eeg=True, mindist=5.0,
    verbose=True
)

# 转换前向模型为固定方向
fwd = mne.convert_forward_solution(
    fwd, surf_ori=True, force_fixed=True,
    use_cps=True, verbose=False
)

print(f"成功创建了适合BCI-IV-2a数据的前向模型，包含 {fwd['sol']['data'].shape[1]} 个源点，{fwd['sol']['data'].shape[0]} 个EEG通道")

# 设置subjects_dir供后续可视化使用
mne.set_config('SUBJECTS_DIR', subjects_dir)

# 创建一个epochs的副本供后续使用
epochs_stripped = epochs.copy().load_data()

# 模拟BCI-IV-2a数据
# 运动想象EEG通常在mu(8-12Hz)和beta(13-30Hz)频带显示ERD/ERS模式
# 首先，我们计算EEG数据的信噪比（SNR）
target_snr = util.calc_snr_range(epochs, baseline_span=(0.0, 0.5), data_span=(0.5, 3.5))
print(f'目标SNR为 {target_snr:.2f}')

# 评估
# 使用具有BCI-IV-2a特性的数据进行测试 - 优化MI-EEG模拟以更接近真实数据
# BCI-IV-2a模拟设置，优化了MI-EEG特性，增强ERD/ERS模式
# 分别生成左右手的模拟数据
bci_test_settings_left = dict(
    duration_of_trial=4.0,      # BCI-IV-2a试次长度
    target_snr=2,
    number_of_sources=2,   # 源数量
    extents=(5, 15),         # 源范围
    beta_source=(0.8, 1.2),       # beta参数
    source_time_course='mi_eeg',
    laterality='left',  # 左手特定设置
)

bci_test_settings_right = dict(
    duration_of_trial=4.0,      # BCI-IV-2a试次长度
    target_snr=2,
    number_of_sources=2,   # 源数量
    extents=(5, 15),         # 源范围
    beta_source=(0.8, 1.2),       # beta参数
    source_time_course='mi_eeg',
    laterality='right',  # 右手特定设置
)

n_samples = 100  
# ========== 左右手测试用模拟数据保存/加载 ==========
SIM_TEST_LEFT_PATH = 'simulation_test_left.pkl'
SIM_TEST_RIGHT_PATH = 'simulation_test_right.pkl'

# 生成左手模拟数据
if os.path.exists(SIM_TEST_LEFT_PATH):
    print(f'检测到已保存的左手测试模拟数据，直接加载: {SIM_TEST_LEFT_PATH}')
    with open(SIM_TEST_LEFT_PATH, 'rb') as f:
        simulation_test_left = pickle.load(f)
else:
    print('未检测到左手测试模拟数据，开始生成...')
    simulation_test_left = Simulation(fwd, epochs_stripped.info, settings=bci_test_settings_left, verbose=True)
    simulation_test_left.simulate(n_samples=n_samples)
    with open(SIM_TEST_LEFT_PATH, 'wb') as f:
        pickle.dump(simulation_test_left, f)
    print(f'左手测试模拟数据已保存到: {SIM_TEST_LEFT_PATH}')

# 生成右手模拟数据
if os.path.exists(SIM_TEST_RIGHT_PATH):
    print(f'检测到已保存的右手测试模拟数据，直接加载: {SIM_TEST_RIGHT_PATH}')
    with open(SIM_TEST_RIGHT_PATH, 'rb') as f:
        simulation_test_right = pickle.load(f)
else:
    print('未检测到右手测试模拟数据，开始生成...')
    simulation_test_right = Simulation(fwd, epochs_stripped.info, settings=bci_test_settings_right, verbose=True)
    simulation_test_right.simulate(n_samples=n_samples)
    with open(SIM_TEST_RIGHT_PATH, 'wb') as f:
        pickle.dump(simulation_test_right, f)
    print(f'右手测试模拟数据已保存到: {SIM_TEST_RIGHT_PATH}')

# 创建左右手模拟数据的字典，模仿真实数据的结构
simulation_test_dict = {
    'left_hand': simulation_test_left,
    'right_hand': simulation_test_right
}

# 对模拟测试数据进行低通滤波以平滑信号
print("对模拟测试数据进行平滑处理 (40Hz低通滤波)...")
for hand_label, simulation_test_hand in simulation_test_dict.items():
    print(f"正在滤波{hand_label}数据...")
    for i in range(len(simulation_test_hand.eeg_data)):
        simulation_test_hand.eeg_data[i].filter(l_freq=None, h_freq=40., verbose=False)

# 统一可视化时间点
plot_times = [0.5, 1.0, 2.0, 3.0, 4.0]

# ===================================================================
# 模拟数据集ERD/ERS现象可视化 (使用与现有相同的方法)
# ===================================================================
print("\n===== 正在对模拟数据集进行ERD/ERS现象可视化 =====")

# --- 1. 参数定义 ---
fs = 250  # 采样率
filter_band = [10, 12]  # Hz, 滤波频段 (与现有ERD/ERS方法相同)
filter_order = 6  # 滤波器阶数
smooth_window = 100 # 平滑窗口

# 定义MATLAB脚本中的基线周期 (完全与真实数据一致)
baseline_period_matlab = (-2.5, -1.5) # s, 这是plotERP1.m中一个关键且特殊的设定

# --- 2. 处理模拟数据 ---
print("正在处理模拟数据集进行ERD/ERS分析...")

# 首先检查模拟数据的设置
print("检查模拟数据设置:")
for hand_label in ['left_hand', 'right_hand']:
    simulation_test_hand = simulation_test_dict[hand_label]
    print(f"{hand_label} 设置:")
    print(f"  source_time_course: {simulation_test_hand.settings.get('source_time_course', '未设置')}")
    print(f"  duration_of_trial: {simulation_test_hand.settings.get('duration_of_trial', '未设置')}")
    print(f"  laterality: {simulation_test_hand.settings.get('laterality', '未设置')}")

# 收集左右手的所有模拟数据
sim_data_dict = {}
sim_labels_dict = {}
for hand_label, hand_name in zip(['left_hand', 'right_hand'], ['左手', '右手']):
    simulation_test_hand = simulation_test_dict[hand_label]
    print(f"正在收集{hand_name}模拟数据...")
    
    # 收集所有试次的EEG数据
    all_trial_data = []
    for i, epochs_data in enumerate(simulation_test_hand.eeg_data):
        # 获取每个epochs的数据 (n_trials, n_channels, n_times)
        trial_data = epochs_data.get_data()
        all_trial_data.append(trial_data)
        if i == 0:  # 只显示第一个epochs的信息
            print(f"  第一个epochs形状: {trial_data.shape}")
            print(f"  数据范围: {np.min(trial_data):.6f} 到 {np.max(trial_data):.6f}")
    
    # 合并所有试次数据 (total_trials, n_channels, n_times)
    combined_data = np.concatenate(all_trial_data, axis=0)
    sim_data_dict[hand_label] = combined_data
    print(f"  {hand_name}合并数据形状: {combined_data.shape}")
    
    # 创建标签 (1=左手, 2=右手)
    label_val = 1 if hand_label == 'left_hand' else 2
    sim_labels_dict[hand_label] = np.full(combined_data.shape[0], label_val)

# 合并左右手数据
x_train_sim = np.concatenate([sim_data_dict['left_hand'], sim_data_dict['right_hand']], axis=0)
y_train_sim = np.concatenate([sim_labels_dict['left_hand'], sim_labels_dict['right_hand']], axis=0)

# 转换数据格式为 (samples, channels, trials) 以匹配现有ERD/ERS处理逻辑
x_train_sim = np.transpose(x_train_sim, (2, 1, 0))  # (n_times, n_channels, n_trials)
print(f"模拟数据形状: {x_train_sim.shape}")

print("正在对每个试次进行Z-score标准化...")
x_norm_sim = np.zeros_like(x_train_sim, dtype=float)
print(f"原始数据范围: {np.min(x_train_sim):.6f} 到 {np.max(x_train_sim):.6f}")
for i in range(x_train_sim.shape[2]):
    trial = x_train_sim[:, :, i]
    mean_val = np.mean(trial)
    std_val = np.std(trial)
    x_norm_sim[:, :, i] = (trial - mean_val) / std_val if std_val > 0 else trial - mean_val

print(f"标准化后数据范围: {np.min(x_norm_sim):.6f} 到 {np.max(x_norm_sim):.6f}")

# 检查特定时间点的数据
sample_time_idx = np.abs(time_vec - 1.0).argmin()  # 检查1秒时的数据
# 添加边界检查，确保索引不越界
sample_time_idx = min(sample_time_idx, x_train_sim.shape[0] - 1)
print(f"查找1秒时间点的索引: {sample_time_idx}, 实际时间: {time_vec[sample_time_idx]:.3f}s")
print(f"数据形状: {x_train_sim.shape}, 时间向量长度: {len(time_vec)}")
print(f"1秒时原始数据 (C3通道): {x_train_sim[sample_time_idx, 0, :5]}")
print(f"1秒时标准化数据 (C3通道): {x_norm_sim[sample_time_idx, 0, :5]}")

# --- 3. ERD/ERS 计算与绘图 (修正以适应模拟数据) ---
n_samples = x_norm_sim.shape[0]
n_channels = x_norm_sim.shape[1]

print(f"模拟数据维度: {x_norm_sim.shape}")
print(f"模拟数据样本数: {n_samples}")

# 获取模拟数据的实际时间向量
sim_time_vec = simulation_test_dict['left_hand'].eeg_data[0].times
print(f"模拟数据实际时间范围: {sim_time_vec[0]:.2f}s to {sim_time_vec[-1]:.2f}s")

# MI-EEG模拟数据的特殊时间轴：从-3秒开始，ERD/ERS现象在0-1.75秒，1.75秒后回到基线
# 使用模拟数据自己的时间轴，但使用合理的基线期
time_vec = sim_time_vec
baseline_period_sim = baseline_period_matlab  # 使用(-2.5, -1.5)作为基线期
print(f"使用模拟数据时间轴: {time_vec[0]:.2f}s to {time_vec[-1]:.2f}s")
print(f"MI-EEG信号特点: ERD/ERS现象主要在0-1.75秒，1.75秒后为基线噪声")

# 找到基线周期的样本索引
baseline_start_idx = np.abs(time_vec - baseline_period_sim[0]).argmin()
baseline_end_idx = np.abs(time_vec - baseline_period_sim[1]).argmin()
print(f"使用基线周期: {baseline_period_sim} s (样本索引: {baseline_start_idx} 到 {baseline_end_idx})")
print(f"基线期实际时间: {time_vec[baseline_start_idx]:.2f}s 到 {time_vec[baseline_end_idx]:.2f}s")

# --- 信号处理 (完全按照现有方法) ---
# a. 带通滤波 (按试次进行)
b, a = scipy.signal.butter(filter_order, [f / (fs / 2) for f in filter_band], btype='bandpass')
filtered_data_sim = np.zeros_like(x_norm_sim)
for i in range(x_norm_sim.shape[2]):
    filtered_data_sim[:, :, i] = scipy.signal.filtfilt(b, a, x_norm_sim[:, :, i], axis=0)

print(f"滤波后数据范围: {np.min(filtered_data_sim):.6f} 到 {np.max(filtered_data_sim):.6f}")
# 确保索引在滤波数据中也不越界
safe_idx = min(sample_time_idx, filtered_data_sim.shape[0] - 1)
print(f"1秒时滤波数据 (C3通道): {filtered_data_sim[safe_idx, 0, :5]}")

# b. 计算功率 (平方)
power_data_sim = np.square(filtered_data_sim)
print(f"功率数据范围: {np.min(power_data_sim):.6f} 到 {np.max(power_data_sim):.6f}")

# 检查μ频段是否有信号
print(f"检查μ频段滤波效果...")
test_trial = 0
test_ch = 0  # C3通道
original_signal = x_norm_sim[:, test_ch, test_trial]
filtered_signal = filtered_data_sim[:, test_ch, test_trial]
print(f"原始信号标准差: {np.std(original_signal):.6f}")
print(f"滤波信号标准差: {np.std(filtered_signal):.6f}")
print(f"滤波信号/原始信号比例: {np.std(filtered_signal)/np.std(original_signal):.6f}")

# c. 绘图 (完全按照现有方法)
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False) # 不共享Y轴以匹配MATLAB行为
colors = {'C3': 'r', 'C4': 'b', 'Cz': 'g'}
# 获取模拟数据的通道名称
ch_names_sim = simulation_test_dict['left_hand'].eeg_data[0].ch_names
ch_map_sim = {}
for i, ch_name in enumerate(ch_names_sim):
    if ch_name in ['C3', 'C4', 'Cz']:
        ch_map_sim[i] = ch_name

for ax, (label_val, hand_name) in zip(axes, zip([1, 2], ['左手', '右手'])):
    print(f"\n--- 正在处理模拟数据: {hand_name} ---")
    
    # 选择当前任务的试次的功率数据
    hand_power_trials = power_data_sim[:, :, y_train_sim == label_val]
    print(f"{hand_name}数据形状: {hand_power_trials.shape}")
    
    # d. 在试次上平均
    avg_power = np.mean(hand_power_trials, axis=2) # shape: (samples, channels)
    print(f"{hand_name}平均功率形状: {avg_power.shape}")
    print(f"{hand_name}平均功率范围: {np.min(avg_power):.6f} 到 {np.max(avg_power):.6f}")
    
    # e. 平滑
    smoothed_power = np.apply_along_axis(
        lambda x: np.convolve(x, np.ones(smooth_window)/smooth_window, mode='same'),
        axis=0, arr=avg_power
    )
    print(f"{hand_name}平滑后功率范围: {np.min(smoothed_power):.6f} 到 {np.max(smoothed_power):.6f}")
    
    # f. 基线校正 (A-R)/R * 100
    baseline_power = np.mean(smoothed_power[baseline_start_idx:baseline_end_idx, :], axis=0)
    baseline_power[baseline_power == 0] = 1e-9 # 避免除以零
    print(f"{hand_name}基线功率: {baseline_power}")
    
    erd_ers_percent = ((smoothed_power - baseline_power) / baseline_power) * 100
    print(f"{hand_name}ERD/ERS范围: {np.min(erd_ers_percent):.2f}% 到 {np.max(erd_ers_percent):.2f}%")

    # g. 绘图
    for i in ch_map_sim.keys():
        ch_name = ch_map_sim.get(i, f'Ch{i+1}')
        print(f"正在绘制{hand_name}的{ch_name}通道，数据范围: {np.min(erd_ers_percent[:, i]):.2f}% 到 {np.max(erd_ers_percent[:, i]):.2f}%")
        ax.plot(time_vec, erd_ers_percent[:, i], label=ch_name, color=colors.get(ch_name, 'k'))

    ax.set_title(f'模拟数据: {hand_name}')
    ax.set_xlabel('时间/s')
    ax.axvline(0, linestyle='--', color='k', linewidth=0.8, label='Cue')
    ax.axhline(0, linestyle='-', color='k', linewidth=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    # 只显示有意义的时间范围：ERD/ERS现象主要在-1到2秒
    ax.set_xlim(-1, 2)

axes[0].set_ylabel('ERD/ERS (%)')
fig.suptitle(f'模拟数据集μ频段 ({filter_band[0]}-{filter_band[1]} Hz) 的ERD/ERS现象\n(ERD/ERS主要在0-1.75s，基线期: -2.5至-1.5s)')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

save_figure(fig, 'simulated_data_erd_ers_visualization.png')
plt.show()

print(f"\n===== 正在分别可视化左右手模拟测试集数据 =====")
# 分别处理左右手模拟数据，模仿真实数据的处理方式
for hand_label, hand_name in zip(['left_hand', 'right_hand'], ['左手', '右手']):
    simulation_test_hand = simulation_test_dict[hand_label]
    
    # 从测试集中选择一个样本进行可视化
    idx = 1
    print(f"\n--- 正在显示: 模拟测试集{hand_name} - 真实源 (样本 {idx}) ---")
    source = simulation_test_hand.source_data[idx]
    evoked = simulation_test_hand.eeg_data[idx].average()

    # 添加模拟测试集真正的平均波形显示（对所有样本求平均）
    print(f"正在显示：模拟测试集{hand_name}真正的平均EEG波形（所有样本平均，C3、C4、Cz通道）")
    # 正确的方法：将所有epochs合并，然后求平均
    all_epochs_list = []
    for i in range(len(simulation_test_hand.eeg_data)):
        # 每个eeg_data[i]已经是一个epochs对象，直接添加到列表中
        all_epochs_list.append(simulation_test_hand.eeg_data[i])
    
    # 合并所有epochs
    all_epochs_combined = mne.concatenate_epochs(all_epochs_list)
    # 计算真正的平均（nave会正确显示样本数）
    true_averaged_evoked = all_epochs_combined.average()

    central_channels = ['C3', 'Cz', 'C4']
    available_central_channels = [ch for ch in central_channels if ch in true_averaged_evoked.ch_names]
    if available_central_channels:
        central_evoked = true_averaged_evoked.pick_channels(available_central_channels, ordered=True)
        fig_sim_avg_evoked = central_evoked.plot(verbose=0, show=False)
        fig_sim_avg_evoked.suptitle(f'模拟测试集{hand_name}真正平均EEG波形（{len(all_epochs_combined)}个样本平均，{available_central_channels}通道）- 源定位前')
        # 添加图例显示通道和颜色对应关系
        axes = fig_sim_avg_evoked.get_axes()
        if len(axes) > 0:
            axes[0].legend(available_central_channels, loc='upper right', fontsize=8)
        set_chinese_font_for_figure(fig_sim_avg_evoked)
        save_figure(fig_sim_avg_evoked, f'sim_test_{hand_label}_true_averaged_central_channels_before_source_loc.png')
        plt.show()
    else:
        print(f"错误: 没有找到可用的中央通道")

    # 真实源: 3D脑图
    brain_true = source.plot(**plot_params)
    add_text_with_chinese(brain_true, 0.1, 0.9, f'模拟测试集{hand_name}: 真实源 (样本 {idx})', 'title', font_size=14)
    save_brain_screenshot(brain_true, f'sim_test_{hand_label}_true_brain_sample{idx}.png')
    plt.show()

    # 真实源: 波形图（仅显示C3、Cz、C4通道）
    motor_channels = ['C3', 'Cz', 'C4']
    available_central_channels = [ch for ch in motor_channels if ch in evoked.ch_names]
    if available_central_channels:
        central_evoked = evoked.pick_channels(available_central_channels, ordered=True)
        fig_true_evoked = central_evoked.plot(show=False)
        fig_true_evoked.suptitle(f'模拟测试集{hand_name}: 真实EEG波形 (样本 {idx}, {available_central_channels}通道)')
        # 添加图例显示通道和颜色对应关系
        axes = fig_true_evoked.get_axes()
        if len(axes) > 0:
            axes[0].legend(available_central_channels, loc='upper right', fontsize=8)
        set_chinese_font_for_figure(fig_true_evoked)
        save_figure(fig_true_evoked, f'sim_test_{hand_label}_true_evoked_central_channels_sample{idx}.png')
        plt.show()
    else:
        print(f"错误: 模拟测试集{hand_name}中没有找到可用的中央通道")

    # 真实源: 脑地形图（仅显示C3、Cz、C4通道）
    valid_times_true = [t for t in plot_times if evoked.times[0] <= t <= evoked.times[-1]]
    if available_central_channels:
        # 使用之前创建的central_evoked
        fig_true_topomap = central_evoked.plot_topomap(times=valid_times_true, time_format='%0.1f s')
        fig_true_topomap.suptitle(f'模拟测试集{hand_name}: 真实EEG头皮地形图 (样本 {idx}, {available_central_channels}通道)')
        set_chinese_font_for_figure(fig_true_topomap)
        save_figure(fig_true_topomap, f'sim_test_{hand_label}_true_topomap_central_channels_sample{idx}.png')
        plt.show()
    else:
        print(f"错误: 模拟测试集{hand_name}中没有找到可用的中央通道进行topomap显示")

# BCI-IV-2a模拟设置 - 优化MI-EEG特性以更接近真实数据
# 修改模拟参数以更好地反映BCI-IV-2a特性，优化模拟数据的真实性
bci_settings = dict(
    duration_of_trial=4.0,      # BCI-IV-2a试次长度
    target_snr=2,
    number_of_sources=2,   # 源数量
    extents=(5, 15),         # 源范围
    beta_source=(1, 1.5),       # beta参数
    source_time_course='mi_eeg', # 改为MI-EEG模拟，包含ERD/ERS模式
    laterality='bilateral',  # 训练数据包含混合的左右手特征
)

# 样本数量
n_samples = 1000
# ========== 训练用模拟数据保存/加载 ==========
SIM_TRAIN_PATH = 'simulation_train.pkl'
if os.path.exists(SIM_TRAIN_PATH):
    print(f'检测到已保存的训练模拟数据，直接加载: {SIM_TRAIN_PATH}')
    with open(SIM_TRAIN_PATH, 'rb') as f:
        simulation = pickle.load(f)
else:
    print('未检测到训练模拟数据，开始生成...')
    simulation = Simulation(fwd, epochs.info, settings=bci_settings, verbose=True)
    simulation.simulate(n_samples=n_samples)
    with open(SIM_TRAIN_PATH, 'wb') as f:
        pickle.dump(simulation, f)
    print(f'训练模拟数据已保存到: {SIM_TRAIN_PATH}')

# 对模拟训练数据进行低通滤波以平滑信号
print("对模拟训练数据进行平滑处理 (40Hz低通滤波)...")
for i in range(len(simulation.eeg_data)):
    simulation.eeg_data[i].filter(l_freq=None, h_freq=40., verbose=False)

# 训练网络
# 选择适合BCI-IV-2a分析的网络模型
model_type = 'pinn'
print(f"使用PINN模型类型: {model_type}")
def get_model_name(model_type):
    """返回标准化模型名称（大写）"""
    return model_type.upper()

# ==================================================
#  选取激活最强顶点的实用函数 (需在首次使用前定义)
# ==================================================

def get_top_vertices(stc, n=2, time_point=0):
    """返回(1) data 索引列表 picks 供 show_traces 使用;
    (2) foci 列表, 每项为 (hemi, vertex_id) 供 add_foci 使用"""
    data_abs = np.abs(stc.data[:, time_point])
    sorted_idx = np.argsort(data_abs)[-n:]
    picks = sorted_idx.tolist()

    lh_vertices = stc.vertices[0]
    lh_len = len(lh_vertices)
    foci = []
    for idx in picks:
        if idx < lh_len:
            hemi = 'lh'
            vert = lh_vertices[idx]
        else:
            hemi = 'rh'
            vert = stc.vertices[1][idx - lh_len]
        foci.append((hemi, vert))
    return picks, foci

net = Net(fwd, verbose=2, model_type=model_type)

print("开始训练网络以解码BCI-IV-2a数据...")
# GPU优化训练参数
if gpus:
    # GPU训练：增大batch_size以充分利用GPU内存
    batch_size = 64  #
    epochs = 1000      # 可以增加epochs数，因为GPU训练更快
    print(f"🚀 使用GPU训练 - batch_size: {batch_size}, epochs: {epochs}")

# 使用with tf.device确保在GPU上训练
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    print("开始训练... (只显示每个epoch的loss和metrics)")
    start_time = time.time()
    net, history = net.fit(simulation, epochs=epochs, batch_size=batch_size, return_history=True)
    end_time = time.time()
    print(f"训练总耗时: {end_time - start_time:.2f} 秒")

# 检查训练后的模型类型
if hasattr(net, 'model'):
    print(f"训练后模型名称: {net.model.name}")
    # 检查是否为PINN模型(应该有两个输出)
    if hasattr(net.model, 'output') and isinstance(net.model.output, list) and len(net.model.output) == 2:
        print("成功使用PINN模型 - 具有双输出结构")
    else:
        print("警告: 可能未使用PINN模型 - 没有预期的双输出结构")

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

loss_dir = os.path.join(os.path.dirname(__file__), 'loss_curves')
os.makedirs(loss_dir, exist_ok=True)

plt.figure(figsize=(8, 6))
history_dict = history.history if hasattr(history, 'history') else history
train_loss = history_dict.get('loss', [])
val_loss = history_dict.get('val_loss', [])
plt.plot(smooth_curve(train_loss), label='训练损失')
if val_loss:
    plt.plot(smooth_curve(val_loss), label='验证损失')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('模型训练与验证损失曲线（平滑）')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
loss_curve_path = os.path.join(loss_dir, 'loss_curve.png')
plt.savefig(loss_curve_path, dpi=150)
plt.show()
plt.close()
print(f"已保存loss曲线至: {loss_curve_path}")

# ===================================================================
# 2. 模拟测试集源定位 - 左右手分别处理
# ===================================================================

print("\n===== 正在对模拟测试集数据进行源定位 (左右手分别处理) =====")
# 获取偶极子位置
_, _, pos, _ = util.unpack_fwd(fwd)  

# 分别对左右手模拟数据进行源定位预测
for hand_label, hand_name in zip(['left_hand', 'right_hand'], ['左手', '右手']):
    simulation_test_hand = simulation_test_dict[hand_label]
    print(f"\n--- 正在处理模拟测试集{hand_name}数据 ---")
    
    # 从模拟数据预测源
    source_hat_hand = net.predict(simulation_test_hand)

    # 计算评价指标
    print(f"计算{hand_name}模拟数据评价指标...")

    # -- 并行计算辅助函数 --
    def calculate_metrics_for_sample(i, source_hat, simulation_test, pos):
        """为单个样本计算所有评价指标"""
        y_true = simulation_test.source_data[i].data[:, 0]  # 取第一个时间点
        y_est = source_hat[i].data[:, 0]  # 取第一个时间点
        
        mle = eval_mean_localization_error(y_true, y_est, pos)
        mse = eval_mse(y_true, y_est)
        auc_close, auc_far = eval_auc(y_true, y_est, pos)
        
        return mle, mse, auc_close, auc_far

    # -- 使用joblib并行计算所有样本的指标 --
    results = Parallel(n_jobs=-1, verbose=1)(
        delayed(calculate_metrics_for_sample)(i, source_hat_hand, simulation_test_hand, pos)
        for i in range(len(source_hat_hand))
    )

    # -- 收集并行计算结果 --
    mle_values = [res[0] for res in results]
    mse_values = [res[1] for res in results]
    auc_close_values = [res[2] for res in results]
    auc_far_values = [res[3] for res in results]

    # 输出平均指标
    print(f"{hand_name}模拟数据平均定位误差 (MLE): {np.nanmean(mle_values):.2f} mm")
    print(f"{hand_name}模拟数据均方误差 (MSE): {np.nanmean(mse_values):.2e}")
    print(f"{hand_name}模拟数据AUC (近距离): {np.nanmean(auc_close_values):.4f}")
    print(f"{hand_name}模拟数据AUC (远距离): {np.nanmean(auc_far_values):.4f}")
    # 合并AUC并输出平均值
    all_auc_values = auc_close_values + auc_far_values
    print(f"{hand_name}模拟数据AUC (平均): {np.nanmean(all_auc_values):.4f}")

    # --- 可视化预测结果 ---
    idx = 0
    print(f"\n--- 正在显示: 模拟测试集{hand_name} - esinet预测源 (样本 {idx}) ---")
    source_pred = source_hat_hand[idx]

    evoked_pred = util.get_eeg_from_source(source_pred, fwd, epochs_stripped.info, tmin=0)

    # 预测源: 3D脑图
    picks, foci = get_top_vertices(source_pred)
    brain_pred = source_pred.plot(**plot_params, time_viewer=True)
    # brain_pred.show_traces(picks=picks) # MNE API变动: .show_traces() 已被移除
    for hemi, vert in foci:
        brain_pred.add_foci(vert, coords_as_verts=True, hemi=hemi, color='red', scale_factor=1.0)
    add_text_with_chinese(brain_pred, 0.1, 0.9, f'模拟测试集{hand_name}: esinet预测源 (样本 {idx})', 'title', font_size=14)
    save_brain_screenshot(brain_pred, f'sim_test_{hand_label}_pred_brain_sample{idx}.png')
    plt.show()

    # 预测源: 波形图 (只显示C3、Cz、C4通道)
    motor_channels = ['C3', 'Cz', 'C4']
    available_pred_channels = [ch for ch in motor_channels if ch in evoked_pred.ch_names]
    if available_pred_channels:
        evoked_pred_motor = evoked_pred.pick_channels(available_pred_channels, ordered=True)
        fig_pred_evoked = evoked_pred_motor.plot(show=False)
        fig_pred_evoked.suptitle(f'模拟测试集{hand_name}: 预测EEG波形 (样本 {idx}) - 运动皮层通道')
        # 添加图例显示通道和颜色对应关系
        axes = fig_pred_evoked.get_axes()
        if len(axes) > 0:
            axes[0].legend(available_pred_channels, loc='upper right', fontsize=8)
        set_chinese_font_for_figure(fig_pred_evoked)
        save_figure(fig_pred_evoked, f'sim_test_{hand_label}_pred_evoked_motor_sample{idx}.png')
        plt.show()

    # 预测源: 脑地形图 (只显示C3、Cz、C4通道)
    valid_times_pred = [t for t in plot_times if evoked_pred.times[0] <= t <= evoked_pred.times[-1]]
    if available_pred_channels:
        fig_pred_topomap = evoked_pred_motor.plot_topomap(times=valid_times_pred, time_format='%0.1f s')
        fig_pred_topomap.suptitle(f'模拟测试集{hand_name}: 预测EEG头皮地形图 (样本 {idx}) - 运动皮层通道')
        set_chinese_font_for_figure(fig_pred_topomap)
        save_figure(fig_pred_topomap, f'sim_test_{hand_label}_pred_topomap_motor_sample{idx}.png')
        plt.show()


# ===================================================================
# 3. 真实数据源定位
# ===================================================================
# 使用训练好的网络预测源
print("\n===== 正在对真实BCI-IV-2a数据进行源定位 (左右手分别处理) =====")
for hand_label, hand_name in zip(['left_hand', 'right_hand'], ['左手', '右手']):
    print(f"\n--- 正在处理: {hand_name} ---")
    hand_epochs = epochs_stripped[hand_label]
    stc = net.predict(hand_epochs.average())[0]
    
    # 添加真实数据集源定位后平均波形显示
    print(f"正在显示：真实数据集{hand_name}平均EEG波形（C3、C4、Cz通道）- 源定位后")
    central_channels = ['C3', 'Cz', 'C4']
    available_central_channels = [ch for ch in central_channels if ch in hand_epochs.ch_names]
    if available_central_channels:
        real_central_evoked_after = hand_epochs.average().pick_channels(available_central_channels, ordered=True)
        fig_real_avg_evoked_after = real_central_evoked_after.plot(verbose=0, show=False)
        fig_real_avg_evoked_after.suptitle(f'真实数据集{hand_name}平均EEG波形（{available_central_channels}通道）- 源定位后')
        # 添加图例显示通道和颜色对应关系
        axes = fig_real_avg_evoked_after.get_axes()
        if len(axes) > 0:
            axes[0].legend(available_central_channels, loc='upper right', fontsize=8)
        set_chinese_font_for_figure(fig_real_avg_evoked_after)
        save_figure(fig_real_avg_evoked_after, f'real_{hand_label}_averaged_central_channels_after_source_loc.png')
        plt.show()
    else:
        print(f"错误: 没有找到可用的中央通道")

    # ================== 真实数据集 - 预测前（模型输入） - EEG平均波形图 ==================
    # 数据来源: 真实数据集（BCI-IV-2a）
    # 图类型: EEG平均时域波形图 (只显示C3、Cz、C4通道)
    # 作用: 展示真实数据集在该类别下的EEG信号时域特征
    motor_channels = ['C3', 'Cz', 'C4']
    available_real_channels = [ch for ch in motor_channels if ch in hand_epochs.ch_names]
    if available_real_channels:
        hand_epochs_motor = hand_epochs.copy().pick_channels(available_real_channels, ordered=True)
        fig_real_evoked = hand_epochs_motor.average().plot(show=False)
        fig_real_evoked.suptitle(f'真实数据: {hand_name}EEG信号 - 运动皮层通道')
        # 添加图例显示通道和颜色对应关系
        axes = fig_real_evoked.get_axes()
        if len(axes) > 0:
            axes[0].legend(available_real_channels, loc='upper right', fontsize=8)
        set_chinese_font_for_figure(fig_real_evoked)
        save_figure(fig_real_evoked, f'real_data_{hand_label}_evoked_motor.png')
        plt.show()


    # ================== 真实数据集 - 预测前（模型输入） - 头皮地形图 ==================
    # 数据来源: 真实数据集（BCI-IV-2a）
    # 图类型: EEG头皮电位地形图 (只显示C3、Cz、C4通道)
    # 作用: 展示真实数据集在该类别下的EEG信号空间分布
    valid_times = [t for t in plot_times if hand_epochs.times[0] <= t <= hand_epochs.times[-1]]
    if len(valid_times) < len(plot_times):
        print(f"警告: 部分时间点超出{hand_name}BCI-IV-2a信号范围，仅绘制有效时间点: {valid_times}")
    if available_real_channels:
        fig_real_topomap = hand_epochs_motor.average().plot_topomap(times=valid_times, time_format='%0.1f s')
        fig_real_topomap.suptitle(f'真实数据: {hand_name}BCI-IV-2a地形图 - 运动皮层通道')
        set_chinese_font_for_figure(fig_real_topomap)
        save_figure(fig_real_topomap, f'real_data_{hand_label}_topomap_motor.png')
        plt.show()


    # ================== 真实数据集 - 预测后（模型输出） - 3D脑源分布图 ==================
    # 数据来源: 真实数据集（BCI-IV-2a），模型预测结果（PINN/FC等）
    # 图类型: 3D脑源分布图
    # 作用: 展示模型对真实数据的源空间预测分布
    picks, foci = get_top_vertices(stc)
    brain_real_pred = stc.plot(**plot_params, time_viewer=True)
    # brain_real_pred.show_traces(picks=picks, colors=['orange', 'cyan']) # MNE API变动: .show_traces() 已被移除
    for hemi, vert in foci:
        brain_real_pred.add_foci(vert, coords_as_verts=True, hemi=hemi, color='red', scale_factor=1.0)
    add_text_with_chinese(brain_real_pred, 0.1, 0.9, f'{get_model_name(model_type)}对{hand_name}BCI-IV-2a数据的预测', 'title', font_size=14)
    save_brain_screenshot(brain_real_pred, f'real_data_{hand_label}_pred_brain.png')
    plt.show()


    # ================== 真实数据集 - 预测后（模型输出） - 预测源重建EEG波形图 ==================
    # 数据来源: 真实数据集（BCI-IV-2a），模型预测结果反投到EEG空间
    # 图类型: 预测源重建EEG平均波形图 (只显示C3、Cz、C4通道)
    # 作用: 展示模型预测源反投到传感器空间后的EEG信号时域特征
    evoked_esi = util.get_eeg_from_source(stc, fwd, hand_epochs.info, tmin=0.)
    available_esi_channels = [ch for ch in motor_channels if ch in evoked_esi.ch_names]
    if available_esi_channels:
        evoked_esi_motor = evoked_esi.pick_channels(available_esi_channels, ordered=True)
        fig_esi_evoked = evoked_esi_motor.plot(show=False)
        fig_esi_evoked.suptitle(f'重建信号: {hand_name} (来自esinet预测源) - 运动皮层通道')
        # 添加图例显示通道和颜色对应关系
        axes = fig_esi_evoked.get_axes()
        if len(axes) > 0:
            axes[0].legend(available_esi_channels, loc='upper right', fontsize=8)
        set_chinese_font_for_figure(fig_esi_evoked)
        save_figure(fig_esi_evoked, f'real_data_{hand_label}_pred_evoked_motor.png')
        plt.show()


    # ================== 真实数据集 - 预测后（模型输出） - 预测源重建EEG头皮地形图 ==================
    # 数据来源: 真实数据集（BCI-IV-2a），模型预测结果反投到EEG空间
    # 图类型: 预测源重建EEG头皮地形图 (只显示C3、Cz、C4通道)
    # 作用: 展示模型预测源反投到传感器空间后的EEG信号空间分布
    valid_times_esi = [t for t in plot_times if evoked_esi.times[0] <= t <= evoked_esi.times[-1]]
    if len(valid_times_esi) < len(plot_times):
        print(f"警告: 部分时间点超出预测信号范围，仅绘制有效时间点: {valid_times_esi}")
    if available_esi_channels:
        fig_esi_topomap = evoked_esi_motor.plot_topomap(times=valid_times_esi, time_format='%0.1f s')
        fig_esi_topomap.suptitle(f'重建信号: {hand_name}BCI-IV-2a地形图 (来自esinet预测源) - 运动皮层通道')
        set_chinese_font_for_figure(fig_esi_topomap)
        save_figure(fig_esi_topomap, f'real_data_{hand_label}_pred_topomap_motor.png')
        plt.show()


print("\nBCI-IV-2a源定位分析完成！")
