# 使用esinet预测BCI-IV-2a数据集的源定位分析
# 本教程使用BCI-IV-2a运动想象EEG数据集

import mne
import numpy as np
from copy import deepcopy
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

# 固定随机种子，保证仿真数据可复现
np.random.seed(42)
random.seed(42)

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
else:
    print("❌ 未检测到GPU设备，将使用CPU训练")

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

if not epochs_list:
    raise ValueError("未能成功加载任何BCI-IV-2a数据文件，请检查数据路径和文件格式")

# 使用第一个被试的数据进行演示
epochs, epochs_plot = epochs_list[0]
print(f"使用第一个被试的数据进行分析，包含 {len(epochs)} 个试次")

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

# 1. 原始数据可视化
print("正在显示：原始EEG平均波形（所有试次）")
fig_raw_evoked = epochs_plot.average().plot(verbose=0)
fig_raw_evoked.suptitle('原始EEG平均波形（所有试次）')
save_figure(fig_raw_evoked, 'raw_evoked_alltrials.png')

raw_times = [0.5, 1.0, 2.0, 3.0]
last_time = epochs_plot.average().times[-1]
if last_time not in raw_times:
    raw_times.append(last_time)
raw_times = sorted(set([t for t in raw_times if epochs_plot.average().times[0] <= t <= last_time]))
print(f"正在显示：原始EEG头皮地形图（所有试次），时间点：{raw_times}")
fig_raw_topomap = epochs_plot.average().plot_topomap(times=raw_times, time_format='%0.1f s')
fig_raw_topomap.suptitle(f'原始EEG头皮地形图（所有试次）- 时间点：{raw_times}')
save_figure(fig_raw_topomap, 'raw_topomap_alltrials_times_' + '_'.join([f'{t:.3f}s' for t in raw_times]) + '.png')
set_chinese_font_for_figure(fig_raw_topomap)
# 单独保存每个时间点的topomap
for t in raw_times:
    fig_t = epochs_plot.average().plot_topomap(times=[t], time_format='%0.1f s')
    save_figure(fig_t, f'raw_topomap_alltrials_time{t:.3f}s.png')
    plt.close(fig_t)

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

# BCI-IV-2a模拟设置
# 修改模拟参数以更好地反映BCI-IV-2a特性
bci_settings = dict(
    duration_of_trial=4.0,      # BCI-IV-2a试次长度
    target_snr=5,      
    number_of_sources=1,   # 源数量
    extents=(5, 15),            # 源范围
    beta_source=(1, 1.5),       # beta参数
    source_time_course='sine'   # 使用正弦波模式
)

# 样本数量
n_samples = 1000
print("开始模拟BCI-IV-2a数据...")
simulation = Simulation(fwd, epochs.info, settings=bci_settings, verbose=True)
simulation.simulate(n_samples=n_samples)

# 可视化模拟数据
idx = 0
print("绘制模拟的BCI-IV-2a地形图...")
# 检查可用的时间范围
print(f"模拟数据的时间范围: {simulation.eeg_data[idx].times.min()} 到 {simulation.eeg_data[idx].times.max()} 秒")
# 统一可视化时间点
plot_times = [0.5, 1.0, 2.0, 3.0, 4.0]
valid_times = [t for t in plot_times if simulation.eeg_data[idx].times[0] <= t <= simulation.eeg_data[idx].times[-1]]
# 2. 模拟数据可视化
# 动态获取最后一个可用时间点，确保无越界
sim_times = [0.5, 1.0, 2.0, 3.0]
sim_last_time = simulation.eeg_data[idx].average().times[-1]
if sim_last_time not in sim_times:
    sim_times.append(sim_last_time)
sim_times = sorted(set([t for t in sim_times if simulation.eeg_data[idx].times[0] <= t <= sim_last_time]))
print(f"正在显示：模拟EEG头皮地形图，样本{idx}，时间点：{sim_times}")
fig_sim_topomap = simulation.eeg_data[idx].average().plot_topomap(times=sim_times, time_format='%0.1f s')
fig_sim_topomap.suptitle(f'模拟EEG头皮地形图-样本{idx}-时间点：{sim_times}')
save_figure(fig_sim_topomap, f'sim_topomap_sample{idx}_times_' + '_'.join([f'{t:.3f}s' for t in sim_times]) + '.png')
set_chinese_font_for_figure(fig_sim_topomap)
print(f"正在显示：模拟源空间分布（3D大脑图），样本{idx}")
brain_sim = simulation.source_data[idx].plot(**plot_params, initial_time=simulation.source_data[idx].times[0], time_viewer=False)
add_text_with_chinese(brain_sim, 0.1, 0.9, f'模拟源空间分布-样本{idx}', 'title', font_size=14)
save_brain_screenshot(brain_sim, f'sim_brain_sample{idx}.png')

# 训练网络
# 选择适合BCI-IV-2a分析的网络模型
model_type = 'pinn'
print(f"使用PINN模型类型: {model_type}")
net = Net(fwd, verbose=2, model_type=model_type)

print("开始训练网络以解码BCI-IV-2a数据...")
# GPU优化训练参数
if gpus:
    # GPU训练：增大batch_size以充分利用GPU内存
    batch_size = 16  # 从4增加到16
    epochs = 10      # 可以增加epochs数，因为GPU训练更快
    print(f"🚀 使用GPU训练 - batch_size: {batch_size}, epochs: {epochs}")
else:
    # CPU训练：保持较小的batch_size
    batch_size = 4
    epochs = 5
    print(f"🐌 使用CPU训练 - batch_size: {batch_size}, epochs: {epochs}")

# 使用with tf.device确保在GPU上训练
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    print("开始训练... (只显示每个epoch的loss和metrics)")
    start_time = time.time()
    net.fit(simulation, epochs=epochs, batch_size=batch_size)
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

# 评估
# 使用具有BCI-IV-2a特性的数据进行测试
bci_test_settings = dict(
    duration_of_trial=4.0,  # 与训练数据保持一致
    target_snr=target_snr,
    number_of_sources=(1, 2),  # 保持与训练数据一致
    extents=(5, 15),
    beta_source=(1, 1.5),
    source_time_course='sine'
)

n_samples = 10
simulation_test = Simulation(fwd, epochs_stripped.info, settings=bci_test_settings, verbose=True)
simulation_test.simulate(n_samples=n_samples)

# 从BCI-IV-2a预测源
source_hat = net.predict(simulation_test)

# 计算评价指标
print("计算评价指标...")
_, _, pos, _ = util.unpack_fwd(fwd)  # 获取偶极子位置

# 为每个样本计算评价指标
mle_values = []
mse_values = []
auc_close_values = []
auc_far_values = []

for i in range(len(source_hat)):
    # 获取真实源和预测源数据
    y_true = simulation_test.source_data[i].data[:, 0]  # 取第一个时间点
    y_est = source_hat[i].data[:, 0]  # 取第一个时间点
    
    # 计算平均定位误差
    mle = eval_mean_localization_error(y_true, y_est, pos)
    mle_values.append(mle)
    
    # 计算均方误差
    mse = eval_mse(y_true, y_est)
    mse_values.append(mse)
    
    # 计算AUC
    auc_close, auc_far = eval_auc(y_true, y_est, pos)
    auc_close_values.append(auc_close)
    auc_far_values.append(auc_far)

# 输出平均指标
print(f"平均定位误差 (MLE): {np.nanmean(mle_values):.2f} mm")
print(f"均方误差 (MSE): {np.nanmean(mse_values):.8f}")
print(f"AUC (近距离): {np.nanmean(auc_close_values):.4f}")
print(f"AUC (远距离): {np.nanmean(auc_far_values):.4f}")

# 可视化样本
# 真实值
idx = 0
source = simulation_test.source_data[idx]
# 绘制模拟BCI-IV-2a源
print("绘制模拟BCI-IV-2a源...")
a = source.plot(**plot_params)
add_text_with_chinese(a, 0.1, 0.9, 'BCI-IV-2a 真实源', 'title', font_size=14)

# 绘制模拟BCI-IV-2a
print("绘制模拟BCI-IV-2a信号...")
evoked = simulation_test.eeg_data[idx].average()
fig_evoked = evoked.plot()
plt.title('模拟BCI-IV-2a信号')
set_chinese_font_for_figure(fig_evoked)  # 应用中文字体设置

# 查看测试数据的时间范围
print(f"测试数据的时间范围: {simulation_test.eeg_data[idx].times.min()} 到 {simulation_test.eeg_data[idx].times.max()} 秒")
# 统一可视化时间点
valid_times = [t for t in plot_times if simulation_test.eeg_data[idx].times[0] <= t <= simulation_test.eeg_data[idx].times[-1]]
if len(valid_times) < len(plot_times):
    print(f"警告: 部分时间点超出测试数据范围，仅绘制有效时间点: {valid_times}")
fig = evoked.plot_topomap(times=valid_times, time_format='%0.1f s')
fig.suptitle('模拟BCI-IV-2a地形图')
set_chinese_font_for_figure(fig)  # 应用中文字体设置

# 绘制esinet预测
print("绘制esinet预测源...")
b = source_hat[idx].plot(**plot_params)
add_text_with_chinese(b, 0.1, 0.9, 'BCI-IV-2a 预测源', 'title', font_size=14)

# 绘制预测的EEG
print("绘制预测BCI-IV-2a信号...")
evoked_hat = util.get_eeg_from_source(source_hat[idx], fwd, epochs_stripped.info, tmin=0)
fig_evoked_hat = evoked_hat.plot()
plt.title('预测BCI-IV-2a信号')
set_chinese_font_for_figure(fig_evoked_hat)  # 应用中文字体设置

# 修正plot_topomap参数 - 确保times在有效范围内
valid_times = [t for t in plot_times if evoked_hat.times[0] <= t <= evoked_hat.times[-1]]
if len(valid_times) < len(plot_times):
    print(f"警告: 部分时间点超出预测信号范围，仅绘制有效时间点: {valid_times}")
fig = evoked_hat.plot_topomap(times=valid_times, time_format='%0.1f s')
fig.suptitle('预测BCI-IV-2a地形图')
set_chinese_font_for_figure(fig)  # 应用中文字体设置

# 真实数据源定位
# 使用训练好的网络预测源
print("使用训练好的网络预测真实BCI-IV-2a数据源...")
# 选择左手运动想象的数据进行测试
left_epochs = epochs_stripped['left_hand']
stc = net.predict(left_epochs.average())[0]

# 绘制预测源
brain = stc.plot(**plot_params)
add_text_with_chinese(brain, 0.1, 0.9, 'esinet对左手BCI-IV-2a数据的预测', 'title', font_size=14)

# 绘制真实EEG
left_epochs.plot()
plt.title('左手BCI-IV-2a信号')
# 修正plot_topomap参数 - 确保times在有效范围内
valid_times = [t for t in plot_times if left_epochs.times[0] <= t <= left_epochs.times[-1]]
if len(valid_times) < len(plot_times):
    print(f"警告: 部分时间点超出左手BCI-IV-2a信号范围，仅绘制有效时间点: {valid_times}")
fig = left_epochs.average().plot_topomap(times=valid_times, time_format='%0.1f s')
fig.suptitle('左手BCI-IV-2a地形图')
set_chinese_font_for_figure(fig)

# 绘制预测的EEG
evoked_esi = util.get_eeg_from_source(stc, fwd, left_epochs.info, tmin=0.)
fig_evoked_esi = evoked_esi.plot()
plt.title('从预测源重建的BCI-IV-2a信号')
set_chinese_font_for_figure(fig_evoked_esi)  # 应用中文字体设置

# 修正plot_topomap参数 - 确保times在有效范围内
valid_times = [t for t in plot_times if evoked_esi.times[0] <= t <= evoked_esi.times[-1]]
if len(valid_times) < len(plot_times):
    print(f"警告: 部分时间点超出预测源重建信号范围，仅绘制有效时间点: {valid_times}")
fig = evoked_esi.plot_topomap(times=valid_times, time_format='%0.1f s')
fig.suptitle('从预测源重建的BCI-IV-2a地形图')
set_chinese_font_for_figure(fig)  # 应用中文字体设置

# 使用eLORETA进行比较分析
print("使用eLORETA进行BCI-IV-2a信号的源定位...")
method = "eLORETA"
snr = 3.
lambda2 = 1. / snr ** 2

# 创建一个副本用于eLORETA分析，并添加平均参考
left_epochs_eeg_ref = left_epochs.copy()
left_epochs_eeg_ref.set_eeg_reference(projection=True)  # 添加EEG平均参考投影仪

# 计算协方差矩阵
noise_cov = mne.compute_covariance(
    left_epochs_eeg_ref, tmax=0.5, method=['shrunk', 'empirical'], rank=None, verbose=False)

# 创建逆算子
inverse_operator = mne.minimum_norm.make_inverse_operator(
    left_epochs_eeg_ref.info, fwd, noise_cov, loose='auto', depth=None, fixed=True, 
    verbose=False)
    
# 应用逆算子
stc_elor, residual = mne.minimum_norm.apply_inverse(
    left_epochs_eeg_ref.average(), 
    inverse_operator, 
    lambda2,
    method=method, 
    return_residual=True, 
    verbose=False
)

# 绘制源空间结果
brain = np.abs(stc_elor).plot(**plot_params)
add_text_with_chinese(brain, 0.1, 0.9, 'eLORETA对BCI-IV-2a数据的分析', 'title', font_size=14)

# 绘制预测的EEG
evoked_elor = util.get_eeg_from_source(stc_elor, fwd, left_epochs_eeg_ref.info, tmin=0.)
fig_evoked_elor = evoked_elor.plot()
plt.title('eLORETA重建的BCI-IV-2a信号')
set_chinese_font_for_figure(fig_evoked_elor)  # 应用中文字体设置

# 修正plot_topomap参数 - 确保times在有效范围内
valid_times = [t for t in plot_times if evoked_elor.times[0] <= t <= evoked_elor.times[-1]]
if len(valid_times) < len(plot_times):
    print(f"警告: 部分时间点超出eLORETA重建信号范围，仅绘制有效时间点: {valid_times}")
fig = evoked_elor.plot_topomap(times=valid_times, time_format='%0.1f s')
fig.suptitle('eLORETA重建的BCI-IV-2a地形图')
set_chinese_font_for_figure(fig)  # 应用中文字体设置

print("BCI-IV-2a源定位分析完成！")

# 3. 测试数据可视化
# 绘制测试模拟EEG波形
print(f"正在显示：测试模拟EEG平均波形，样本{idx}")
fig_test_evoked = evoked.plot()
fig_test_evoked.suptitle(f'测试模拟EEG平均波形-样本{idx}')
set_chinese_font_for_figure(fig_test_evoked)
# 绘制测试模拟EEG地形图
print(f"正在显示：测试模拟EEG头皮地形图，样本{idx}，时间点：{sim_times}")
fig_test_topomap = evoked.plot_topomap(times=sim_times, time_format='%0.1f s')
fig_test_topomap.suptitle(f'测试模拟EEG头皮地形图-样本{idx}-时间点：{sim_times}')
set_chinese_font_for_figure(fig_test_topomap)

# 4. esinet预测相关
# 绘制esinet预测源空间分布（3D大脑图）
print(f"正在显示：esinet预测源空间分布（3D大脑图），样本{idx}")
brain_pred = source_hat[idx].plot(**plot_params)
add_text_with_chinese(brain_pred, 0.1, 0.9, f'esinet预测源空间分布-样本{idx}', 'title', font_size=14)
# 绘制esinet预测EEG波形
print(f"正在显示：esinet预测EEG平均波形，样本{idx}")
fig_pred_evoked = evoked_hat.plot()
fig_pred_evoked.suptitle(f'esinet预测EEG平均波形-样本{idx}')
set_chinese_font_for_figure(fig_pred_evoked)
# 绘制esinet预测EEG地形图
print(f"正在显示：esinet预测EEG头皮地形图，样本{idx}，时间点：{sim_times}")
fig_pred_topomap = evoked_hat.plot_topomap(times=sim_times, time_format='%0.1f s')
fig_pred_topomap.suptitle(f'esinet预测EEG头皮地形图-样本{idx}-时间点：{sim_times}')
set_chinese_font_for_figure(fig_pred_topomap)

# 5. 真实数据源定位
# 绘制左手运动想象EEG波形
print(f"正在显示：左手运动想象EEG平均波形")
fig_left_evoked = left_epochs.plot()
fig_left_evoked.suptitle('左手运动想象EEG平均波形')
set_chinese_font_for_figure(fig_left_evoked)
# 绘制左手运动想象EEG地形图
left_times = [0.5, 1.0, 2.0, 3.0]
left_last_time = left_epochs.average().times[-1]
if left_last_time not in left_times:
    left_times.append(left_last_time)
left_times = sorted(set([t for t in left_times if left_epochs.times[0] <= t <= left_last_time]))
print(f"正在显示：左手运动想象EEG头皮地形图，时间点：{left_times}")
fig_left_topomap = left_epochs.average().plot_topomap(times=left_times, time_format='%0.1f s')
fig_left_topomap.suptitle(f'左手运动想象EEG头皮地形图-时间点：{left_times}')
set_chinese_font_for_figure(fig_left_topomap)
# 绘制esinet对真实数据的预测源空间分布
print(f"正在显示：esinet对真实数据预测源空间分布（3D大脑图）")
brain_real_pred = stc.plot(**plot_params)
add_text_with_chinese(brain_real_pred, 0.1, 0.9, 'esinet对真实数据预测源空间分布', 'title', font_size=14)
# 绘制从预测源重建的EEG波形
print(f"正在显示：esinet预测源重建EEG平均波形")
fig_esi_evoked = evoked_esi.plot()
fig_esi_evoked.suptitle('esinet预测源重建EEG平均波形')
set_chinese_font_for_figure(fig_esi_evoked)
# 绘制从预测源重建的EEG地形图
print(f"正在显示：esinet预测源重建EEG头皮地形图，时间点：{left_times}")
fig_esi_topomap = evoked_esi.plot_topomap(times=left_times, time_format='%0.1f s')
fig_esi_topomap.suptitle(f'esinet预测源重建EEG头皮地形图-时间点：{left_times}')
set_chinese_font_for_figure(fig_esi_topomap)

# 6. eLORETA对比分析
# 绘制eLORETA分析得到的源空间分布
print(f"正在显示：eLORETA源空间分布（3D大脑图）")
brain_elor = np.abs(stc_elor).plot(**plot_params)
add_text_with_chinese(brain_elor, 0.1, 0.9, 'eLORETA源空间分布', 'title', font_size=14)
# 绘制eLORETA重建的EEG波形
print(f"正在显示：eLORETA重建EEG平均波形")
fig_elor_evoked = evoked_elor.plot()
fig_elor_evoked.suptitle('eLORETA重建EEG平均波形')
set_chinese_font_for_figure(fig_elor_evoked)
# 绘制eLORETA重建的EEG地形图
print(f"正在显示：eLORETA重建EEG头皮地形图，时间点：{left_times}")
fig_elor_topomap = evoked_elor.plot_topomap(times=left_times, time_format='%0.1f s')
fig_elor_topomap.suptitle(f'eLORETA重建EEG头皮地形图-时间点：{left_times}')
set_chinese_font_for_figure(fig_elor_topomap)

