# 使用MNE方法对BCI-IV-2a数据集进行源定位分析
# 本教程使用BCI-IV-2a运动想象EEG数据集，实现最小范数估计(Minimum Norm Estimate)源定位

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
import contextlib
import pickle

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
plot_params = dict(surface='inflated', cortex="low_contrast", hemi='both', verbose=0,
                  background='white', foreground='black')

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

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# ================== 主流程 ==================

# 1. 加载真实数据
bci_data_path = r"E:/pycharm/PINN/cursor-gPINN/BCI2a"
epochs_list = load_bci_iv_2a_data(bci_data_path)
if not epochs_list:
    raise ValueError("未能成功加载任何BCI-IV-2a数据文件，请检查数据路径和文件格式")
epochs, epochs_plot = epochs_list[0]

# 2. 可视化真实数据集（所有试次）
fig_raw_evoked = epochs_plot.average().plot(verbose=0)
fig_raw_evoked.suptitle('原始EEG平均波形（所有试次）')
save_figure(fig_raw_evoked, 'raw_evoked_alltrials.png')
raw_times = [0.5, 1.0, 2.0, 3.0]
last_time = epochs_plot.average().times[-1]
if last_time not in raw_times:
    raw_times.append(last_time)
raw_times = sorted(set([t for t in raw_times if epochs_plot.average().times[0] <= t <= last_time]))
fig_raw_topomap = epochs_plot.average().plot_topomap(times=raw_times, time_format='%0.1f s')
fig_raw_topomap.suptitle(f'原始EEG头皮地形图（所有试次）- 时间点：{raw_times}')
save_figure(fig_raw_topomap, 'raw_topomap_alltrials_times_' + '_'.join([f'{t:.3f}s' for t in raw_times]) + '.png')
set_chinese_font_for_figure(fig_raw_topomap)

# 3. 创建前向模型
subjects_dir = str(mne.datasets.sample.data_path() / 'subjects')
subject = 'fsaverage'
src_fname = os.path.join(subjects_dir, subject, 'bem', f'{subject}-oct-6-src.fif')
if not os.path.exists(src_fname):
    src = mne.setup_source_space(subject, spacing='oct3', subjects_dir=subjects_dir, add_dist=False, verbose=True)
    mne.write_source_spaces(src_fname, src, overwrite=True)
else:
    src = mne.read_source_spaces(src_fname, verbose=True)
bem_fname = os.path.join(subjects_dir, subject, 'bem', f'{subject}-5120-5120-5120-bem-sol.fif')
bem = mne.read_bem_solution(bem_fname, verbose=True)
trans_fname = os.path.join(subjects_dir, subject, 'bem', f'{subject}-trans.fif')
trans = mne.read_trans(trans_fname, verbose=True)
fwd = mne.make_forward_solution(epochs.info, trans=trans, src=src, bem=bem, meg=False, eeg=True, mindist=5.0, verbose=True)
fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True, verbose=False)
mne.set_config('SUBJECTS_DIR', subjects_dir)
epochs_stripped = epochs.copy().load_data()

# ========== 测试用模拟数据保存/加载 ==========
SIM_TEST_PATH = '2个源10dB测试.pkl'
if os.path.exists(SIM_TEST_PATH):
    print(f'检测到已保存的测试模拟数据，直接加载: {SIM_TEST_PATH}')
    with open(SIM_TEST_PATH, 'rb') as f:
        simulation_test = pickle.load(f)
else:
    print('未检测到测试模拟数据，开始生成...')
    bci_test_settings = dict(
        duration_of_trial=4.0,  # 与训练数据保持一致
        target_snr=2,
        number_of_sources=1,  # 保持与训练数据一致
        extents=(5, 15),
        beta_source=(1, 1.5),
        source_time_course='sine'
    )
    n_samples = 100
    simulation_test = Simulation(fwd, epochs_stripped.info, settings=bci_test_settings, verbose=True)
    simulation_test.simulate(n_samples=n_samples)
    with open(SIM_TEST_PATH, 'wb') as f:
        pickle.dump(simulation_test, f)
    print(f'测试模拟数据已保存到: {SIM_TEST_PATH}')

# 后续MNE分析全部基于simulation_test
eeg_data = simulation_test.eeg_data
source_data = simulation_test.source_data
n_samples = len(eeg_data)

# 4. 可视化模拟数据（第一个样本，源定位前）
idx = 0
plot_params = dict(surface='inflated', cortex="low_contrast", hemi='both', verbose=0,
                  background='white', foreground='black')
plot_times = [0.5, 1.0, 2.0, 3.0, 4.0]
sim_times = [t for t in plot_times if eeg_data[idx].times[0] <= t <= eeg_data[idx].times[-1]]
fig_sim_topomap = eeg_data[idx].average().plot_topomap(times=sim_times, time_format='%0.1f s')
fig_sim_topomap.suptitle(f'训练模拟EEG头皮地形图-样本{idx}-时间点：{sim_times}')
save_figure(fig_sim_topomap, f'sim_topomap_sample{idx}_times_' + '_'.join([f'{t:.3f}s' for t in sim_times]) + '.png')
set_chinese_font_for_figure(fig_sim_topomap)

# 新增：仿真样本真值源空间分布可视化（与BCIIV2a.py一致）
source = simulation_test.source_data[idx]
print(f"绘制仿真样本真值源空间分布-样本{idx} ...")
a = source.plot(**plot_params)
add_text_with_chinese(a, 0.1, 0.9, f'仿真真值源空间分布-样本{idx}', 'title', font_size=14)

# 循环外防呆处理所有仿真样本的baseline和average reference，屏蔽MNE日志
for i in range(n_samples):
    epochs = eeg_data[i]
    with suppress_stdout():
        if not hasattr(epochs, '_baseline_applied') or not epochs._baseline_applied:
            epochs.apply_baseline((None, 0.5))
            epochs._baseline_applied = True
        if not epochs.info['projs']:
            epochs.set_eeg_reference('average', projection=True)
            epochs.apply_proj()

# 6. MNE对模拟数据集

print("\n===== MNE对模拟数据的源定位分析=====")
method = "MNE"
snr = 3.0  # MNE通常使用较高的SNR
lambda2 = 1. / snr ** 2

mne_mle_values = []
mne_mse_values = []
mne_auc_close_values = []
mne_auc_far_values = []

_, _, pos, _ = util.unpack_fwd(fwd)

for i in range(n_samples):
    sim_epochs = eeg_data[i]
    sim_epochs_ref = sim_epochs.copy()
    # 强制转换为float64并重新构造EpochsArray，确保类型兼容
    _data = sim_epochs_ref.get_data()
    if _data.dtype != np.float64:
        _data = _data.astype(np.float64)
    sim_epochs_ref = mne.EpochsArray(
        _data, sim_epochs_ref.info, events=sim_epochs_ref.events,
        event_id=sim_epochs_ref.event_id, tmin=sim_epochs_ref.tmin, verbose=False
    )
    # 屏蔽apply_baseline所有stdout输出
    with suppress_stdout():
        sim_epochs_ref.apply_baseline((None, 0.5))
    noise_cov_sim = mne.compute_covariance(
        sim_epochs_ref, tmax=0.5, method=['shrunk', 'empirical'], rank=None, verbose=False)
    inverse_operator_sim = mne.minimum_norm.make_inverse_operator(
        sim_epochs_ref.info, fwd, noise_cov_sim, loose='auto', depth=None, fixed=True, verbose=False)
    stc_mne_sim, _ = mne.minimum_norm.apply_inverse(
        sim_epochs_ref.average(), inverse_operator_sim, lambda2,
        method=method, return_residual=True, verbose=False)

    true_source_stc = source_data[i]
    _, peak_time_idx = true_source_stc.get_peak(hemi=None, tmin=None, tmax=None, mode='abs', vert_as_index=True, time_as_index=True)
    mne_y_true = true_source_stc.data[:, peak_time_idx]
    if peak_time_idx < stc_mne_sim.data.shape[1]:
        mne_y_est = stc_mne_sim.data[:, peak_time_idx]
    else:
        mne_y_est = stc_mne_sim.data[:, -1]

    mne_mle_values.append(eval_mean_localization_error(mne_y_true, mne_y_est, pos))
    mne_mse_values.append(eval_mse(mne_y_true, mne_y_est))
    auc_close, auc_far = eval_auc(mne_y_true, mne_y_est, pos)
    mne_auc_close_values.append(auc_close)
    mne_auc_far_values.append(auc_far)

print(f"MNE 平均定位误差 (MLE): {np.nanmean(mne_mle_values):.2f} mm")
print(f"MNE 均方误差 (MSE): {np.nanmean(mne_mse_values):.2e}")
all_mne_auc_values = mne_auc_close_values + mne_auc_far_values
print(f"MNE AUC (平均): {np.nanmean(all_mne_auc_values):.4f}")

# ========== 新增：MNE第一个样本的可视化（增加滤波以平滑信号） ===========
idx = 0
# 准备第一个样本用于可视化
sim_epochs_orig = eeg_data[idx]
# 强制转换为float64并重新构造EpochsArray，确保类型兼容以解决compute_covariance的ValueError
_data = sim_epochs_orig.get_data()
if _data.dtype != np.float64:
    _data = _data.astype(np.float64)
sim_epochs_vis = mne.EpochsArray(
    _data, sim_epochs_orig.info, events=sim_epochs_orig.events,
    event_id=sim_epochs_orig.event_id, tmin=sim_epochs_orig.tmin, verbose=False
)

# 确保应用了基线校正
with suppress_stdout():
    sim_epochs_vis.apply_baseline((None, 0.5))
    sim_epochs_vis._baseline_applied = True

# 获取单试次的Evoked对象
evoked_single = sim_epochs_vis.average()
# 对单试次EEG信号应用低通滤波以平滑可视化
evoked_single.filter(l_freq=None, h_freq=40.0, fir_design='firwin', verbose=False)

# 使用此单试次计算噪声协方差和逆算子
noise_cov_vis = mne.compute_covariance(
    sim_epochs_vis, tmax=0.5, method=['shrunk', 'empirical'], rank=None, verbose=False)
inverse_operator_vis = mne.minimum_norm.make_inverse_operator(
    sim_epochs_vis.info, fwd, noise_cov_vis, loose='auto', depth=None, fixed=True, verbose=False)

# 使用滤波后的Evoked数据进行逆解
stc_mne_sim, _ = mne.minimum_norm.apply_inverse(
    evoked_single, inverse_operator_vis, lambda2,
    method=method, return_residual=True, verbose=False)

# 对源定位结果也进行低通滤波，以平滑脑区激活时序图
stc_mne_sim.filter(l_freq=None, h_freq=40.0, fir_design='firwin', verbose=False)

# ================== MNE重建源空间分布多视角组合图（静态截图） ==================
# 数据来源: MNE重建结果，模拟测试集，样本idx
# 图类型: 3D脑源分布多视角组合图（静态）
# 作用: 展示MNE重建源空间分布的多视角特征
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
    stc=stc_mne_sim,
    grid_layout=example_grid_layout,
    subjects_dir=subjects_dir,
    base_plot_params=custom_base_params,
    figsize_per_subplot=(3.5, 3.5),
    title=f"MNE重建源空间分布-样本{idx} 多视角组合图",
    output_filename=os.path.join(FIG_DIR, f'MNE_brain_sample{idx}_composite.png')
)

# ================== MNE重建源空间分布峰值多视角截图（静态截图） ==================
# 数据来源: MNE重建结果，模拟测试集，样本idx
# 图类型: 3D脑源分布峰值多视角截图（静态）
# 作用: 展示MNE重建源空间分布在峰值时刻的多视角特征
peak_vertex, peak_time_idx = stc_mne_sim.get_peak(hemi=None, tmin=None, tmax=None, mode='abs', vert_as_index=True, time_as_index=True)
peak_time = stc_mne_sim.times[peak_time_idx]
print(f"MNE重建样本{idx}源活动峰值时间: {peak_time:.3f}s, 顶点: {peak_vertex}")
brain_mne_peak = stc_mne_sim.plot(**plot_params, initial_time=peak_time, time_viewer=True, subjects_dir=subjects_dir)
add_text_with_chinese(brain_mne_peak, 0.1, 0.9, f'MNE重建源-样本{idx}-峰值{peak_time:.2f}s', 'title', font_size=14)
# 不自动关闭窗口，用户可交互操作

# ================== MNE反投EEG波形图 ==================
# 数据来源: MNE重建源，反投到EEG空间，模拟测试集，样本idx
# 图类型: EEG平均波形图
# 作用: 展示MNE重建源反投到传感器空间后的EEG信号时域特征
evoked_mne = util.get_eeg_from_source(stc_mne_sim, fwd, eeg_data[idx].info, tmin=0.)
fig_evoked_mne = evoked_mne.plot()
plt.title('MNE重建的模拟BCI-IV-2a信号')
set_chinese_font_for_figure(fig_evoked_mne)

# ================== MNE反投EEG头皮地形图 ==================
# 数据来源: MNE重建源，反投到EEG空间，模拟测试集，样本idx
# 图类型: EEG头皮地形图
# 作用: 展示MNE重建源反投到传感器空间后的EEG信号空间分布
plot_times = [0.5, 1.0, 2.0, 3.0, 4.0]
valid_times_mne = [t for t in plot_times if evoked_mne.times[0] <= t <= evoked_mne.times[-1]]
if len(valid_times_mne) < len(plot_times):
    print(f"警告: 部分时间点超出MNE重建信号范围，仅绘制有效时间点: {valid_times_mne}")
fig_mne_topomap = evoked_mne.plot_topomap(times=valid_times_mne, time_format='%0.1f s')
fig_mne_topomap.suptitle('MNE重建的模拟BCI-IV-2a地形图')
set_chinese_font_for_figure(fig_mne_topomap)

# ================== MNE对真实数据集的源定位分析 ==================
print("\n===== MNE对真实BCI-IV-2a数据的源定位分析=====")

# 使用MNE方法预测真实BCI-IV-2a数据源
print("使用MNE方法预测真实BCI-IV-2a数据源...（左右手均分析）")
for hand_label, hand_name in zip(['left_hand', 'right_hand'], ['左手', '右手']):
    hand_epochs = epochs_stripped[hand_label]
    
    # 准备数据用于MNE分析
    hand_epochs_ref = hand_epochs.copy()
    # 强制转换为float64并重新构造EpochsArray，确保类型兼容
    _data = hand_epochs_ref.get_data()
    if _data.dtype != np.float64:
        _data = _data.astype(np.float64)
    hand_epochs_ref = mne.EpochsArray(
        _data, hand_epochs_ref.info, events=hand_epochs_ref.events,
        event_id=hand_epochs_ref.event_id, tmin=hand_epochs_ref.tmin, verbose=False
    )
    # 屏蔽apply_baseline所有stdout输出
    with suppress_stdout():
        hand_epochs_ref.apply_baseline((None, 0.5))
        # 设置EEG平均参考投影器
        if not hand_epochs_ref.info['projs']:
            hand_epochs_ref.set_eeg_reference('average', projection=True)
            hand_epochs_ref.apply_proj()
    
    # 计算噪声协方差
    noise_cov_real = mne.compute_covariance(
        hand_epochs_ref, tmax=0.5, method=['shrunk', 'empirical'], rank=None, verbose=False)
    
    # 创建逆算子
    inverse_operator_real = mne.minimum_norm.make_inverse_operator(
        hand_epochs_ref.info, fwd, noise_cov_real, loose='auto', depth=None, fixed=True, verbose=False)
    
    # 应用MNE
    stc_mne_real, _ = mne.minimum_norm.apply_inverse(
        hand_epochs_ref.average(), inverse_operator_real, lambda2,
        method=method, return_residual=True, verbose=False)

    # ================== 真实数据集 - 预测前（模型输出） - EEG平均波形图 ==================
    # 数据来源: 真实数据集（BCI-IV-2a），模型预测结果（MNE等）
    # 图类型: EEG平均时域波形图
    # 作用: 展示真实数据集在该类别下的EEG信号时域特征
    hand_epochs.plot()
    plt.title(f'{hand_name}BCI-IV-2a信号')

    # ================== 真实数据集 - 预测前（模型输出） - 头皮地形图 ==================
    # 数据来源: 真实数据集（BCI-IV-2a），模型预测结果（MNE等）
    # 图类型: EEG头皮电位地形图
    # 作用: 展示真实数据集在该类别下的EEG信号空间分布
    plot_times = [0.5, 1.0, 2.0, 3.0]
    valid_times = [t for t in plot_times if hand_epochs.times[0] <= t <= hand_epochs.times[-1]]
    if len(valid_times) < len(plot_times):
        print(f"警告: 部分时间点超出{hand_name}BCI-IV-2a信号范围，仅绘制有效时间点: {valid_times}")
    fig = hand_epochs.average().plot_topomap(times=valid_times, time_format='%0.1f s')
    fig.suptitle(f'{hand_name}BCI-IV-2a地形图')
    set_chinese_font_for_figure(fig)

    # ================== 真实数据集 - 预测后（模型输出） - 3D脑源分布图 ==================
    # 数据来源: 真实数据集（BCI-IV-2a），模型预测结果（MNE等）
    # 图类型: 3D脑源分布图
    # 作用: 展示模型对真实数据的源空间预测分布
    brain = stc_mne_real.plot(**plot_params)
    add_text_with_chinese(brain, 0.1, 0.9, f'MNE对{hand_name}BCI-IV-2a数据的预测', 'title', font_size=14)
    save_brain_screenshot(brain, f'mne_{hand_label}_real_data_brain.png')

    # ================== 真实数据集 - 预测后（模型输出） - 预测源重建EEG波形图 ==================
    # 数据来源: 真实数据集（BCI-IV-2a），模型预测结果反投到EEG空间
    # 图类型: 预测源重建EEG平均波形图
    # 作用: 展示模型预测源反投到传感器空间后的EEG信号时域特征
    evoked_esi = util.get_eeg_from_source(stc_mne_real, fwd, hand_epochs.info, tmin=0.)
    fig_evoked_esi = evoked_esi.plot()
    plt.title(f'从MNE预测源重建的{hand_name}BCI-IV-2a信号')
    set_chinese_font_for_figure(fig_evoked_esi)

    # ================== 真实数据集 - 预测后（模型输出） - 预测源重建EEG头皮地形图 ==================
    # 数据来源: 真实数据集（BCI-IV-2a），模型预测结果反投到EEG空间
    # 图类型: 预测源重建EEG头皮地形图
    # 作用: 展示模型预测源反投到传感器空间后的EEG信号空间分布
    valid_times = [t for t in plot_times if evoked_esi.times[0] <= t <= evoked_esi.times[-1]]
    if len(valid_times) < len(plot_times):
        print(f"警告: 部分时间点超出预测信号范围，仅绘制有效时间点: {valid_times}")
    fig = evoked_esi.plot_topomap(times=valid_times, time_format='%0.1f s')
    fig.suptitle(f'MNE预测{hand_name}BCI-IV-2a地形图')
    set_chinese_font_for_figure(fig)

    # ============== 多视角组合图演示 ==============
    if stc_mne_real is not None:
        print(f"正在使用 MNE 模型预测的{hand_name}数据 stc 生成组合视图...")
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
            stc=stc_mne_real,
            grid_layout=example_grid_layout,
            subjects_dir=subjects_dir,
            base_plot_params=custom_base_params,
            figsize_per_subplot=(3.5, 3.5),
            title=f"MNE对{hand_name}数据的多视角组合图",
            output_filename=os.path.join(FIG_DIR, f'mne_{hand_label}_real_data_composite_views.png')
        )
    else:
        print(f"警告: 未找到 {hand_name} stc 对象，无法生成组合视图。")

print("BCI-IV-2a MNE源定位分析完成！")

