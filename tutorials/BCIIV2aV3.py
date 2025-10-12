# 使用esinet预测BCI-IV-2a数据集的源定位分析
# 无Dirichlet 与 Robin边界条件

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
import pickle
# 强制设置PyVista 3D后端
import mne
mne.viz.set_3d_backend('pyvista')

# 固定随机种子，保证仿真数据可复现
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
plot_params = dict(surface='inflated', cortex="low_contrast", hemi='both', verbose=0, background='white', foreground='black')

# 加载BCI-IV-2a数据
print("加载BCI-IV-2a运动想象EEG数据集...")

# BCI-IV-2a数据路径
bci_data_path = r"E:/pycharm/PINN/cursor-gPINN/BCI2a"

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

# 新增：批量保存单个方法不同视角的3D脑图截图
def save_stc_views(stc, method_name, sample_idx, views=['lateral', 'medial', 'dorsal', 'ventral']):
    for view in views:
        brain = stc.plot(surface='inflated', hemi='both', view=view, time_viewer=False, subjects_dir=mne.get_config('SUBJECTS_DIR'), size=(800, 600), background='white', foreground='black', clim='auto', colorbar=False)
        filename = f"{method_name}_sample{sample_idx}_{view}.png"
        save_brain_screenshot(brain, filename)
        brain.close()

def plot_stc_grid_views(stc, grid_layout, subjects_dir, base_plot_params=None, figsize_per_subplot=(4, 4), title=None, output_filename=None):
    """
    将源定位结果(stc)的多个指定视角绘制到一个网格布局的组合图中。

    参数
    ----------
    stc : mne.SourceEstimate
        要可视化的源数据。
    grid_layout : list of lists of dict
        定义网格布局和每个子图的视角参数。
        外层列表的每个元素代表图中的一行。
        内层列表的每个元素是一个字典，包含 'view'（视角名）、'hemi'（半球）等参数。
        如果某个位置想留空，可以用 None 占位。
    subjects_dir : str
        FreeSurfer 的 SUBJECTS_DIR 路径。
    base_plot_params : dict, optional
        通用于所有子图 stc.plot() 的默认参数。
    figsize_per_subplot : tuple, optional
        每个子图的 matplotlib 英寸大小 (width, height)。
    title : str, optional
        整个图窗的标题。
    output_filename : str, optional
        保存图像的文件名。如果提供，则保存图像并关闭图窗；否则，图窗保持打开状态。
    """
    import matplotlib.pyplot as plt
    import mne

    if base_plot_params is None:
        base_plot_params = dict(surface='inflated', cortex='low_contrast', size=(300,300),
                                background='white', foreground='black', time_viewer=False,
                                show_traces=False, colorbar=False)

    if not grid_layout or not isinstance(grid_layout, list) or not all(isinstance(row, list) for row in grid_layout):
        print("错误: grid_layout 必须是一个列表的列表。")
        return

    n_rows = len(grid_layout)
    if n_rows == 0:
        print("错误: grid_layout 不能为空。")
        return
    n_cols = max(len(row) for row in grid_layout) if n_rows > 0 else 0
    if n_cols == 0:
        print("错误: grid_layout 行不能为空。")
        return

    total_width = n_cols * figsize_per_subplot[0]
    total_height = n_rows * figsize_per_subplot[1]
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(total_width, total_height), squeeze=False)

    # 遍历每个子图，分别渲染不同视角
    for r_idx, row_config in enumerate(grid_layout):
        for c_idx, view_config in enumerate(row_config):
            ax = axes[r_idx, c_idx]
            if view_config is None:
                ax.axis('off')
                continue

            # 合并基础参数和当前子图参数（去除view和hemi，后续单独处理）
            current_plot_params = base_plot_params.copy()
            hemi = view_config.get('hemi', 'both')
            view = view_config.get('view', None)
            # 只保留stc.plot()支持的参数
            # stc.plot() 支持 hemi，但不支持 view
            current_plot_params['hemi'] = hemi

            # 新建Brain对象
            brain = stc.plot(subjects_dir=subjects_dir, **current_plot_params)
            # 切换到指定视角
            if view is not None:
                brain.show_view(view)
            # 截图
            img = brain.screenshot()
            brain.close()

            # 绘制到matplotlib子图
            ax.imshow(img)
            ax.axis('off')
        # 如果当前行的列数少于n_cols，关闭剩余的axes
        for c_idx_remaining in range(len(row_config), n_cols):
            axes[r_idx, c_idx_remaining].axis('off')

    if title:
        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96 if title else 1])
    else:
        fig.tight_layout()

    if output_filename:
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"创建目录: {output_dir}")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"组合视图已保存至: {output_filename}")
        plt.close(fig)
    else:
        pass

# ================== 真实数据集 - 左右手分别 - 预测前（真值） - EEG平均波形图 ==================
# 数据来源: 真实数据集（BCI-IV-2a），左右手分别，真值（未经过模型预测）
# 图类型: EEG平均时域波形图
# 作用: 展示真实数据集左右手分别的EEG平均波形
print("正在显示：左右手平均EEG波形（运动相关通道）")

# 定义与左右手运动相关的通道
central_channels = ['C3', 'Cz', 'C4']

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
            fig_avg_evoked.suptitle(f'原始EEG平均波形: {hand_name}（{available_central_channels}通道）')
            # 添加图例显示通道和颜色对应关系
            axes = fig_avg_evoked.get_axes()
            if len(axes) > 0:
                axes[0].legend(available_central_channels, loc='upper right', fontsize=8)
            set_chinese_font_for_figure(fig_avg_evoked)
            save_figure(fig_avg_evoked, f'raw_{hand_label}_averaged_central_channels_evoked.png')
        else:
            print(f"错误: 没有找到可用的中央通道")

# ================== 真实数据集 - 左右手分别 - 预测前（真值） - 头皮地形图 ==================
# 数据来源: 真实数据集（BCI-IV-2a），左右手分别，真值（未经过模型预测）
# 图类型: EEG头皮电位地形图
# 作用: 展示真实数据集左右手分别在不同时间点的头皮空间分布
raw_times = [0.5, 1.0, 2.0, 3.0]
print(f"正在显示：左右手原始EEG头皮地形图，时间点：{raw_times}")

# 分别处理左手和右手的头皮地形图
for hand_label, hand_name in zip(['left_hand', 'right_hand'], ['左手', '右手']):
    hand_epochs = epochs_plot[hand_label]
    if len(hand_epochs) > 0:
        averaged_evoked = hand_epochs.average()
        
        # 获取有效时间点
        last_time = averaged_evoked.times[-1]
        if last_time not in raw_times:
            raw_times.append(last_time)
        valid_times = sorted(set([t for t in raw_times if averaged_evoked.times[0] <= t <= last_time]))
        
        if len(valid_times) < len(raw_times):
            print(f"警告: 部分时间点超出 {hand_name} 信号范围，仅绘制有效时间点: {valid_times}")

        fig_raw_topomap = averaged_evoked.plot_topomap(times=valid_times, time_format='%0.1f s')
        fig_raw_topomap.suptitle(f'原始EEG头皮地形图 - {hand_name} - 时间点：{valid_times}')
        set_chinese_font_for_figure(fig_raw_topomap)
        save_figure(fig_raw_topomap, f'raw_{hand_label}_topomap_times_' + '_'.join([f'{t:.3f}s' for t in valid_times]) + '.png')
        
        # ============== 原始EEG头皮电位地形图 (单个时间点，用于保存) ==============
        # 单独保存每个时间点的topomap
        for t in valid_times:
            fig_t = averaged_evoked.plot_topomap(times=[t], time_format='%0.1f s')
            save_figure(fig_t, f'raw_{hand_label}_topomap_time{t:.3f}s.png')
            plt.close(fig_t)
    else:
        print(f"警告: {hand_name}数据中没有样本")

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
    number_of_sources=2,   # 源数量
    extents=(5, 15),          # 源范围
    beta_source=(1, 1.5),       # beta参数
    source_time_course='sine'   # 使用正弦波模式
)

# 样本数量
n_samples = 1000
print("开始模拟BCI-IV-2a数据...")

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

# 可视化模拟数据
idx = 0
print("绘制训练模拟的BCI-IV-2a地形图...")
# 检查可用的时间范围
print(f"模拟训练数据的时间范围: {simulation.eeg_data[idx].times.min()} 到 {simulation.eeg_data[idx].times.max()} 秒")
# 统一可视化时间点
plot_times = [0.5, 1.0, 2.0, 3.0, 4.0]
valid_times = [t for t in plot_times if simulation.eeg_data[idx].times[0] <= t <= simulation.eeg_data[idx].times[-1]]

# ================== 模拟数据集 - 训练集 - 预测前（真值） - 头皮地形图 ==================
# 数据来源: 模拟数据集（simulation），训练集，真值（未经过模型预测）
# 图类型: EEG头皮电位地形图
# 数据来源: 模拟生成的EEG数据 (simulation.eeg_data[idx].average() - 用于模型训练的第idx个模拟样本)
# 模型/方法: 无 (此为模拟产生的EEG信号，其源活动是已知的，见下图)
# 目的/含义: 展示模拟引擎产生的、具有特定信噪比和已知源活动的EEG信号在头皮上的空间分布情况。
# 所属阶段: 模拟数据生成 (用于后续模型训练)
# 备注: 此图对应 print 输出 "正在显示：模拟EEG头皮地形图，样本{idx}，时间点：{sim_times}" 和图标题 "模拟EEG头皮地形图-样本{idx}-时间点：{sim_times}"。idx通常为0。
# ===============================================================================
# 2. 模拟数据可视化
# 动态获取最后一个可用时间点，确保无越界
sim_times = [0.5, 1.0, 2.0, 3.0]
sim_last_time = simulation.eeg_data[idx].average().times[-1]
if sim_last_time not in sim_times:
    sim_times.append(sim_last_time)
sim_times = sorted(set([t for t in sim_times if simulation.eeg_data[idx].times[0] <= t <= sim_last_time]))
print(f"正在显示：训练模拟EEG头皮地形图，样本{idx}，时间点：{sim_times}")
fig_sim_topomap = simulation.eeg_data[idx].average().plot_topomap(times=sim_times, time_format='%0.1f s')
fig_sim_topomap.suptitle(f'训练模拟EEG头皮地形图-样本{idx}-时间点：{sim_times}')
save_figure(fig_sim_topomap, f'sim_topomap_sample{idx}_times_' + '_'.join([f'{t:.3f}s' for t in sim_times]) + '.png')
set_chinese_font_for_figure(fig_sim_topomap)

# ================== 模拟数据集 - 训练集 - 预测前（真值） - 3D脑源分布图 ==================
# 数据来源: 模拟数据集（simulation），训练集，真值（未经过模型预测）
# 图类型: 3D脑源分布图
# 作用: 可视化训练模拟数据的真实脑源分布
print(f"正在显示：训练模拟源空间分布（3D大脑图），样本{idx}")
brain_sim_train = simulation.source_data[idx].plot(**plot_params, initial_time=simulation.source_data[idx].times[0], time_viewer=False)
add_text_with_chinese(brain_sim_train, 0.1, 0.9, f'训练模拟源空间分布-样本{idx}', 'title', font_size=14)
save_brain_screenshot(brain_sim_train, f'sim_train_brain_sample{idx}.png')
brain_sim_train.close()

# ================== 模拟数据集 - 训练集 - 预测前（真值） - 3D脑源分布多视角峰值图 ==================
# 数据来源: 模拟数据集（simulation），训练集，真值（未经过模型预测）
# 图类型: 3D脑源分布多视角峰值图
# 作用: 展示训练模拟数据峰值时刻的脑源分布多视角
source_train = simulation.source_data[idx]
peak_vertex, peak_time_idx = source_train.get_peak(hemi=None, tmin=None, tmax=None, mode='abs', vert_as_index=True, time_as_index=True)
peak_time = source_train.times[peak_time_idx]
print(f"训练集样本{idx}源活动峰值时间: {peak_time:.3f}s, 顶点: {peak_vertex}")
brain_train_peak = source_train.plot(**plot_params, initial_time=peak_time, time_viewer=False)
add_text_with_chinese(brain_train_peak, 0.1, 0.9, f'训练集真值源-样本{idx}-峰值{peak_time:.2f}s', 'title', font_size=14)
for view in ['lateral', 'medial', 'dorsal', 'ventral']:
    brain_train_peak.show_view(view)
    save_brain_screenshot(brain_train_peak, f'sim_train_brain_sample{idx}_peak_{peak_time:.2f}s_{view}.png')
brain_train_peak.close()

plt.close('all')

# 选择适合BCI-IV-2a分析的网络模型
model_type = 'pinn'  # 可选: 'pinn', 'fc', 'lstm', 'cnn', 'convdip'

def get_model_name(model_type):
    """返回标准化模型名称（大写）"""
    return model_type.upper()

print(f"使用{get_model_name(model_type)}模型类型: {model_type}")
net = Net(fwd, verbose=2, model_type=model_type)

print("开始训练网络以解码BCI-IV-2a数据...")
# GPU优化训练参数
if gpus:
    # GPU训练：增大batch_size以充分利用GPU内存
    batch_size = 64
    epochs = 1000     # 可以增加epochs数，因为GPU训练更快
    learning_rate = 0.001
    patience = 15
    print(f"🚀 使用GPU训练 - batch_size: {batch_size}, epochs: {epochs}")

# 使用with tf.device确保在GPU上训练
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    print("开始训练... (只显示每个epoch的loss和metrics)")
    start_time = time.time()
    net, history = net.fit(simulation, patience=patience,epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, return_history=True)
    end_time = time.time()
    print(f"训练总耗时: {end_time - start_time:.2f} 秒")

# 检查训练后的模型类型
if hasattr(net, 'model'):
    print(f"训练后模型名称: {net.model.name}")

# ====== 新增：保存loss曲线 ======
import matplotlib.pyplot as plt
import os

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
# 根据实际参数生成文件名：model_type_number_of_sources_target_snr
loss_filename = f"{model_type}_{bci_settings['number_of_sources']}_{bci_settings['target_snr']}.png"
loss_curve_path = os.path.join(loss_dir, loss_filename)
plt.savefig(loss_curve_path, dpi=150)
plt.close()
print(f"已保存loss曲线至: {loss_curve_path}")

# 评估
# 使用具有BCI-IV-2a特性的数据进行测试
bci_test_settings = dict(
    duration_of_trial=4.0,  # 与训练数据保持一致
    target_snr=5,
    number_of_sources=2,  # 保持与训练数据一致
    extents=(5, 15),
    beta_source=(1, 1.5),
    source_time_course='sine'
)

n_samples = 100

# ========== 测试用模拟数据保存/加载 ==========
SIM_TEST_PATH = 'simulation_test.pkl'
if os.path.exists(SIM_TEST_PATH):
    print(f'检测到已保存的测试模拟数据，直接加载: {SIM_TEST_PATH}')
    with open(SIM_TEST_PATH, 'rb') as f:
        simulation_test = pickle.load(f)
else:
    print('未检测到测试模拟数据，开始生成...')
    simulation_test = Simulation(fwd, epochs_stripped.info, settings=bci_test_settings, verbose=True)
    simulation_test.simulate(n_samples=n_samples)
    with open(SIM_TEST_PATH, 'wb') as f:
        pickle.dump(simulation_test, f)
    print(f'测试模拟数据已保存到: {SIM_TEST_PATH}')

# GPU优化的评估阶段
print("🚀 开始GPU优化评估...")
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    start_eval_time = time.time()

    # GPU优化的预测
    print("  正在进行GPU批量预测...")
    source_hat = net.predict(simulation_test)

    # 计算评价指标
    print("  正在计算评价指标...")
    _, _, pos, _ = util.unpack_fwd(fwd)  # 获取偶极子位置

    # GPU加速的评价指标计算
    mle_values = []
    mse_values = []
    auc_close_values = []
    auc_far_values = []

    # 批量处理评价指标以提高效率
    batch_size_eval = 32
    for batch_start in range(0, len(source_hat), batch_size_eval):
        batch_end = min(batch_start + batch_size_eval, len(source_hat))

        # 批量计算指标
        for i in range(batch_start, batch_end):
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

    end_eval_time = time.time()
    print(f"  GPU评估耗时: {end_eval_time - start_eval_time:.2f} 秒")

# 输出平均指标
print(f"平均定位误差 (MLE): {np.nanmean(mle_values):.2f} mm")
print(f"均方误差 (MSE): {np.nanmean(mse_values):.2e}")
# 合并AUC并输出平均值
all_auc_values = auc_close_values + auc_far_values
print(f"AUC (平均): {np.nanmean(all_auc_values):.4f}")

# 可视化样本
# 真实值
idx = 0
source = simulation_test.source_data[idx]

# 测试集真值源点分布多视角组合图可视化与保存
print(f"绘制测试模拟源空间分布（多视角组合图），样本{idx}")
example_grid_layout = [
    [ {'view': 'lateral', 'hemi': 'lh'}, {'view': 'dorsal', 'hemi': 'both'}, {'view': 'lateral', 'hemi': 'rh'} ],
    [ {'view': 'medial', 'hemi': 'lh'}, {'view': 'ventral', 'hemi': 'both'}, {'view': 'medial', 'hemi': 'rh'} ]
]
custom_base_params = dict(
    surface='inflated', cortex='low_contrast', size=(300, 300),
    background='white', foreground='black', time_viewer=False,
    show_traces=False, colorbar=False, verbose=0
)
plot_stc_grid_views(
    stc=simulation_test.source_data[idx],
    grid_layout=example_grid_layout,
    subjects_dir=subjects_dir,
    base_plot_params=custom_base_params,
    figsize_per_subplot=(3.5, 3.5),
    title=f"测试模拟源空间分布-样本{idx} 多视角组合图",
    output_filename=os.path.join(FIG_DIR, f'sim_test_brain_sample{idx}_composite.png')
)

# ============== 测试集真值源多视角峰值3D脑图 ==============
source_test = simulation_test.source_data[idx]
peak_vertex, peak_time_idx = source_test.get_peak(hemi=None, tmin=None, tmax=None, mode='abs', vert_as_index=True, time_as_index=True)
peak_time = source_test.times[peak_time_idx]
print(f"测试集样本{idx}源活动峰值时间: {peak_time:.3f}s, 顶点: {peak_vertex}")
brain_test_peak = source_test.plot(**plot_params, initial_time=peak_time, time_viewer=False)
add_text_with_chinese(brain_test_peak, 0.1, 0.9, f'测试集真值源-样本{idx}-峰值{peak_time:.2f}s', 'title', font_size=14)
for view in ['lateral', 'medial', 'dorsal', 'ventral']:
    brain_test_peak.show_view(view)
    save_brain_screenshot(brain_test_peak, f'sim_test_brain_sample{idx}_peak_{peak_time:.2f}s_{view}.png')
brain_test_peak.close()

# ================== 模拟数据集 - 测试集 - 预测前（真值） - 3D脑源分布图 ==================
# 数据来源: 模拟数据集（simulation_test），测试集，真值（未经过模型预测）
# 作用: 作为模型评估的基准
source = simulation_test.source_data[idx]
print(f"绘制{get_model_name(model_type)}模拟测试源（真值）...")
a = source.plot(**plot_params)
add_text_with_chinese(a, 0.1, 0.9, f'{get_model_name(model_type)}预测源空间分布-样本{idx}', 'title', font_size=14)

# ================== 模拟数据集 - 测试集 - 预测前（真值） - EEG平均波形图 ==================
# 数据来源: 模拟数据集（simulation_test），测试集，真值（未经过模型预测）
# 作用: 展示真实EEG信号的时域特征
print("绘制模拟测试BCI-IV-2a信号（真值）...")
evoked = simulation_test.eeg_data[idx].average()
fig_evoked = evoked.plot()
plt.title('模拟测试BCI-IV-2a信号（真值）')
set_chinese_font_for_figure(fig_evoked)

# ============== 图7: 模拟测试数据的EEG头皮地形图 (评估用，"真值"，样本idx) ==============
# 图类型: EEG头皮电位地形图
# 数据来源: 模拟测试数据集中的EEG数据 (simulation_test.eeg_data[idx].average() - 第idx个模拟测试样本的平均EEG信号)
# 模型/方法: 无 (此为模拟测试数据中产生的"真实"EEG信号)
# 目的/含义: 展示用于评估模型性能的模拟测试EEG信号在头皮上的空间分布特征。
# 所属阶段: 模型评估 (基于模拟测试数据) - "真值"可视化
# 备注: 图标题为 "模拟BCI-IV-2a地形图"。
# =================================================================================================
# 查看测试数据的时间范围
print(f"测试数据的时间范围: {simulation_test.eeg_data[idx].times.min()} 到 {simulation_test.eeg_data[idx].times.max()} 秒")
# 统一可视化时间点
valid_times = [t for t in plot_times if simulation_test.eeg_data[idx].times[0] <= t <= simulation_test.eeg_data[idx].times[-1]]
if len(valid_times) < len(plot_times):
    print(f"警告: 部分时间点超出测试数据范围，仅绘制有效时间点: {valid_times}")
fig = evoked.plot_topomap(times=valid_times, time_format='%0.1f s')
fig.suptitle('模拟测试BCI-IV-2a地形图')
set_chinese_font_for_figure(fig)  # 应用中文字体设置

# ================== 模拟数据集 - 测试集 - 预测后（模型输出） - 3D脑源分布图 ==================
# 数据来源: 模拟数据集（simulation_test），测试集，模型预测结果（PINN/FC等）
# 作用: 展示模型对模拟数据的预测效果
print(f"绘制{get_model_name(model_type)}模拟测试源（模型预测结果）...")
brain_pred = source_hat[idx].plot(**plot_params)
add_text_with_chinese(brain_pred, 0.1, 0.9, f'{get_model_name(model_type)}预测结果-样本{idx}', 'title', font_size=14)
save_brain_screenshot(brain_pred, f'{get_model_name(model_type).lower()}_predict_brain_sample{idx}.png')

# ================== 模拟数据集 - 测试集 - 预测后（模型输出） - 重建EEG平均波形图 ==================
# 数据来源: 模拟数据集（simulation_test），测试集，模型预测结果反投到EEG空间
# 作用: 展示模型预测源反投到传感器空间后的EEG信号时域特征
print("绘制模拟测试BCI-IV-2a信号（模型预测结果反投EEG）...")
evoked_hat = util.get_eeg_from_source(source_hat[idx], fwd, simulation_test.eeg_data[idx].info, tmin=0.)
fig_evoked_hat = evoked_hat.plot()
plt.title('模拟测试BCI-IV-2a信号（模型预测结果反投EEG）')
set_chinese_font_for_figure(fig_evoked_hat)

# ================== 模拟数据集 - 测试集 - 预测后（模型输出） - 重建EEG头皮地形图 ==================
# 数据来源: 模拟数据集（simulation_test），测试集，模型预测结果反投到EEG空间
# 作用: 展示模型预测源反投到传感器空间后的EEG信号在头皮上的空间分布
valid_times_hat = [t for t in plot_times if evoked_hat.times[0] <= t <= evoked_hat.times[-1]]
if len(valid_times_hat) < len(plot_times):
    print(f"警告: 部分时间点超出模型预测反投EEG信号范围，仅绘制有效时间点: {valid_times_hat}")
fig_hat_topomap = evoked_hat.plot_topomap(times=valid_times_hat, time_format='%0.1f s')
fig_hat_topomap.suptitle('模拟测试BCI-IV-2a地形图（模型预测结果反投EEG）')
set_chinese_font_for_figure(fig_hat_topomap)

# 真实数据源定位
# 使用训练好的网络预测源
print("使用训练好的网络预测真实BCI-IV-2a数据源...（左右手均分析）")
for hand_label, hand_name in zip(['left_hand', 'right_hand'], ['左手', '右手']):
    hand_epochs = epochs_stripped[hand_label]
    stc = net.predict(hand_epochs.average())[0]

    # ================== 真实数据集 - 预测前（模型输出） - EEG平均波形图 ==================
    # 数据来源: 真实数据集（BCI-IV-2a），模型预测结果（PINN/FC等）
    # 图类型: EEG平均时域波形图
    # 作用: 展示真实数据集在该类别下的EEG信号时域特征
    hand_epochs.plot()
    plt.title(f'{hand_name}BCI-IV-2a信号')

    # ================== 真实数据集 - 预测前（模型输出） - 头皮地形图 ==================
    # 数据来源: 真实数据集（BCI-IV-2a），模型预测结果（PINN/FC等）
    # 图类型: EEG头皮电位地形图
    # 作用: 展示真实数据集在该类别下的EEG信号空间分布
    valid_times = [t for t in plot_times if hand_epochs.times[0] <= t <= hand_epochs.times[-1]]
    if len(valid_times) < len(plot_times):
        print(f"警告: 部分时间点超出{hand_name}BCI-IV-2a信号范围，仅绘制有效时间点: {valid_times}")
    fig = hand_epochs.average().plot_topomap(times=valid_times, time_format='%0.1f s')
    fig.suptitle(f'{hand_name}BCI-IV-2a地形图')
    set_chinese_font_for_figure(fig)

    # ================== 真实数据集 - 预测后（模型输出） - 3D脑源分布图 ==================
    # 数据来源: 真实数据集（BCI-IV-2a），模型预测结果（PINN/FC等）
    # 图类型: 3D脑源分布图
    # 作用: 展示模型对真实数据的源空间预测分布
    brain = stc.plot(**plot_params)
    add_text_with_chinese(brain, 0.1, 0.9, f'{get_model_name(model_type)}对{hand_name}BCI-IV-2a数据的预测', 'title', font_size=14)
 

    # ================== 真实数据集 - 预测后（模型输出） - 预测源重建EEG波形图 ==================
    # 数据来源: 真实数据集（BCI-IV-2a），模型预测结果反投到EEG空间
    # 图类型: 预测源重建EEG平均波形图
    # 作用: 展示模型预测源反投到传感器空间后的EEG信号时域特征
    evoked_esi = util.get_eeg_from_source(stc, fwd, hand_epochs.info, tmin=0.)
    fig_evoked_esi = evoked_esi.plot()
    plt.title(f'从预测源重建的{hand_name}BCI-IV-2a信号')
    set_chinese_font_for_figure(fig_evoked_esi)

    # ================== 真实数据集 - 预测后（模型输出） - 预测源重建EEG头皮地形图 ==================
    # 数据来源: 真实数据集（BCI-IV-2a），模型预测结果反投到EEG空间
    # 图类型: 预测源重建EEG头皮地形图
    # 作用: 展示模型预测源反投到传感器空间后的EEG信号空间分布
    valid_times = [t for t in plot_times if evoked_esi.times[0] <= t <= evoked_esi.times[-1]]
    if len(valid_times) < len(plot_times):
        print(f"警告: 部分时间点超出预测信号范围，仅绘制有效时间点: {valid_times}")
    fig = evoked_esi.plot_topomap(times=valid_times, time_format='%0.1f s')
    fig.suptitle(f'预测{hand_name}BCI-IV-2a地形图')
    set_chinese_font_for_figure(fig)

    # ============== 多视角组合图演示 ==============
    if stc is not None:
        print(f"正在使用 {get_model_name(model_type)} 模型预测的{hand_name}数据 stc 生成组合视图...")
        example_grid_layout = [
            [ {'view': 'lateral', 'hemi': 'lh'}, {'view': 'dorsal', 'hemi': 'both'}, {'view': 'lateral', 'hemi': 'rh'} ],
            [ {'view': 'medial', 'hemi': 'lh'}, {'view': 'ventral', 'hemi': 'both'}, {'view': 'medial', 'hemi': 'rh'} ]
        ]
        custom_base_params = dict(
            surface='inflated', cortex='low_contrast', size=(300, 300),
            background='white', foreground='black', time_viewer=False,
            show_traces=False, colorbar=False, verbose=0
        )
        plot_stc_grid_views(
            stc=stc,
            grid_layout=example_grid_layout,
            subjects_dir=subjects_dir,
            base_plot_params=custom_base_params,
            figsize_per_subplot=(3.5, 3.5),
            title=f"{get_model_name(model_type)}对{hand_name}数据的多视角组合图",
            output_filename=os.path.join(FIG_DIR, f'{get_model_name(model_type).lower()}_{hand_label}_real_data_composite_views.png')
        )
    else:
        print(f"警告: 未找到 {hand_name} stc 对象，无法生成组合视图。")

print("BCI-IV-2a源定位分析完成！")

