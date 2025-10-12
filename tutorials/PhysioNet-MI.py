# 使用esinet预测MI-EEG数据集的源定位分析
# 本教程使用MNE-Python中的运动想象EEG数据集（eegbci）

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
from mne.datasets import eegbci  # 导入MI-EEG数据集

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

# 设置绘图参数
plot_params = dict(surface='inflated', cortex="low_contrast", hemi='both', verbose=0)

# 加载MI-EEG数据
print("加载PhysioNet运动想象EEG数据集...")

# 直接指定数据路径
custom_path = r"E:\桌面\数据集\PhysioNet-MI"

# 选择受试者和实验条件（run 6: 左/右手想象）
subject = 1
runs = [6]  # 只使用左右手运动想象的数据

# 直接从自定义路径构建文件路径
files = []
for run in runs:
    # 构建文件路径，PhysioNet-MI标准格式是S001/S001R06.edf
    file_path = os.path.join(custom_path, f"S{subject:03d}", f"S{subject:03d}R{run:02d}.edf")
    files.append(file_path)

# 读取并预处理数据
raw = mne.io.read_raw_edf(files[0], preload=True)
eegbci.standardize(raw)  # 应用标准化处理

# 设置标准1020电极系统
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)

# 获取事件并检查实际事件标签
events, event_id = mne.events_from_annotations(raw)
print("数据集中的事件标签:", event_id)

# 标准PhysioNet-MI数据集中，T1通常对应左手，T2对应右手
# 但具体映射可能因数据集版本而异，所以我们打印出所有标签
if 'T1' in event_id and 'T2' in event_id:
    event_id_map = {'left_hand': event_id['T1'], 'right_hand': event_id['T2']}
else:
    # 如果不是标准标签，可能需要按照顺序映射
    labels = list(event_id.keys())
    if len(labels) >= 3:  # 通常T0是休息状态，T1和T2是左右手
        event_id_map = {'left_hand': event_id[labels[1]], 'right_hand': event_id[labels[2]]}
    else:
        print("警告: 无法确定左右手事件标签，使用前两个可用标签")
        event_id_map = {'condition_1': event_id[labels[0]], 'condition_2': event_id[labels[1]]}

print("使用以下事件映射:", event_id_map)

# 创建epochs
tmin, tmax = 0., 4.0  # 运动想象通常在提示后开始，持续几秒钟
baseline = None  # 不进行baseline校正，因为数据已经过滤
epochs = mne.Epochs(raw, events, event_id_map, tmin, tmax, proj=True,
                   baseline=baseline, verbose=0, preload=True)

# 打印通道信息
print(f"数据集中的EEG通道: {len(epochs.ch_names)}")
print(f"所有可用EEG通道: {epochs.ch_names}")

# 可视化样本数据
epochs.average().plot(verbose=0)
fig = epochs.average().plot_topomap(times=[0.5, 1.0, 2.0, 3.0], time_format='%0.1f s')
set_chinese_font_for_figure(fig)
fig.suptitle('运动想象EEG地形图')
print("加载了MI-EEG数据集")

# 获取样本数据集的subjects_dir
print("设置FreeSurfer样本数据集路径...")
subjects_dir = str(mne.datasets.sample.data_path() / 'subjects')
subject = 'fsaverage'  # 使用fsaverage或sample主题

# 创建适合PhysioNet-MI数据的前向模型
print("创建PhysioNet-MI数据的前向模型...")

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

print(f"成功创建了适合PhysioNet-MI数据的前向模型，包含 {len(fwd['sol']['data'])} 个源点")

# 设置subjects_dir供后续可视化使用
mne.set_config('SUBJECTS_DIR', subjects_dir)

# 创建一个epochs的副本供后续使用
epochs_stripped = epochs.copy().load_data()

# 模拟MI-EEG数据
# 运动想象EEG通常在mu(8-12Hz)和beta(13-30Hz)频带显示ERD/ERS模式
# 首先，我们计算EEG数据的信噪比（SNR）
target_snr = util.calc_snr_range(epochs, baseline_span=(0.0, 0.5), data_span=(0.5, 3.5))
print(f'目标SNR为 {target_snr:.2f}')

# MI-EEG模拟设置
# 修改模拟参数以更好地反映MI-EEG特性
mi_settings = dict(
    duration_of_trial=4.0,      # 缩短时间窗口以减少内存使用
    target_snr=target_snr,      
    number_of_sources=(1, 2),   # 减少源数量以减小复杂度
    extents=(5, 15),            # 适当减小源范围
    beta_source=(1, 1.5),       # 减小beta参数以简化模型
    source_time_course='sine'   # 使用正弦波模式
)

# 减少样本数量以减少内存使用
n_samples = 1000  # 从1000减少到200
print("开始模拟MI-EEG数据...")
simulation = Simulation(fwd, epochs.info, settings=mi_settings, verbose=True)
simulation.simulate(n_samples=n_samples)

# 可视化模拟数据
idx = 0
print("绘制模拟的MI-EEG地形图...")
# 修复plot_topomap参数 - 删除title参数并修正times范围
# 检查可用的时间范围
print(f"模拟数据的时间范围: {simulation.eeg_data[idx].times.min()} 到 {simulation.eeg_data[idx].times.max()} 秒")
# 使用更合适的时间点 - 确保在有效范围内
times = [0.0, 0.3, 0.6]  # 确保在模拟数据范围(0.0-0.992)内
fig = simulation.eeg_data[idx].average().plot_topomap(times=times, time_format='%0.1f s')
fig.suptitle('模拟MI-EEG地形图')
set_chinese_font_for_figure(fig)  # 应用中文字体设置
simulation.source_data[idx].plot(**plot_params, initial_time=simulation.source_data[idx].times[idx], time_viewer=False)

# 训练网络
# 选择适合MI-EEG分析的网络模型
model_type = 'pinn'
print(f"使用PINN模型类型: {model_type}")
net = Net(fwd, verbose=1, model_type=model_type)

print("开始训练网络以解码MI-EEG数据...")
# 添加batch_size参数以减小批次大小
net.fit(simulation, epochs=5, batch_size=4)  # 减少epochs和batch_size

# 检查训练后的模型类型
if hasattr(net, 'model'):
    print(f"训练后模型名称: {net.model.name}")
    # 检查是否为PINN模型(应该有两个输出)
    if isinstance(net.model.output, list) and len(net.model.output) == 2:
        print("成功使用PINN模型 - 具有双输出结构")
    else:
        print("警告: 可能未使用PINN模型 - 没有预期的双输出结构")

# 评估
# 使用具有MI-EEG特性的数据进行测试
mi_test_settings = dict(
    duration_of_trial=4.0,  # 与训练数据保持一致
    target_snr=target_snr,
    number_of_sources=(1, 2),  # 保持与训练数据一致
    extents=(5, 15),
    beta_source=(1, 1.5),
    source_time_course='sine'
)


n_samples = 10
simulation_test = Simulation(fwd, epochs.info, settings=mi_test_settings, verbose=True)
simulation_test.simulate(n_samples=n_samples)

# 从MI-EEG预测源
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
# 绘制模拟MI-EEG源
print("绘制模拟MI-EEG源...")
a = source.plot(**plot_params)
add_text_with_chinese(a, 0.1, 0.9, 'MI-EEG 真实源', 'title', font_size=14)

# 绘制模拟MI-EEG
print("绘制模拟MI-EEG信号...")
evoked = simulation_test.eeg_data[idx].average()
fig_evoked = evoked.plot()
plt.title('模拟MI-EEG信号')
set_chinese_font_for_figure(fig_evoked)  # 应用中文字体设置

# 查看测试数据的时间范围
print(f"测试数据的时间范围: {simulation_test.eeg_data[idx].times.min()} 到 {simulation_test.eeg_data[idx].times.max()} 秒")
# 修复plot_topomap参数 - 确保times在有效范围内
times = [0.0, 0.3, 0.6]  # 使用更合适的时间点
fig = evoked.plot_topomap(times=times, time_format='%0.1f s')
fig.suptitle('模拟MI-EEG地形图')
set_chinese_font_for_figure(fig)  # 应用中文字体设置

# 绘制esinet预测
print("绘制esinet预测源...")
b = source_hat[idx].plot(**plot_params)
add_text_with_chinese(b, 0.1, 0.9, 'MI-EEG 预测源', 'title', font_size=14)

# 绘制预测的EEG
print("绘制预测MI-EEG信号...")
evoked_hat = util.get_eeg_from_source(source_hat[idx], fwd, epochs_stripped.info, tmin=0)
fig_evoked_hat = evoked_hat.plot()
plt.title('预测MI-EEG信号')
set_chinese_font_for_figure(fig_evoked_hat)  # 应用中文字体设置

# 修复plot_topomap参数 - 确保times在有效范围内
fig = evoked_hat.plot_topomap(times=times, time_format='%0.1f s')
fig.suptitle('预测MI-EEG地形图')
set_chinese_font_for_figure(fig)  # 应用中文字体设置

# 真实数据源定位
# 使用训练好的网络预测源
print("使用训练好的网络预测真实MI-EEG数据源...")
# 选择左手运动想象的数据进行测试
left_epochs = epochs['left_hand']
stc = net.predict(left_epochs.average())[0]

# 绘制预测源
brain = stc.plot(**plot_params)
add_text_with_chinese(brain, 0.1, 0.9, 'esinet对左手MI-EEG数据的预测', 'title', font_size=14)

# 绘制真实EEG
left_epochs.plot()
plt.title('左手MI-EEG信号')
# 修复plot_topomap参数 - 确保在有效时间范围内
fig = left_epochs.average().plot_topomap(times=[0.5, 1.0, 2.0, 3.0], time_format='%0.1f s')
fig.suptitle('左手MI-EEG地形图')
set_chinese_font_for_figure(fig)

# 绘制预测的EEG
evoked_esi = util.get_eeg_from_source(stc, fwd, left_epochs.info, tmin=0.)
fig_evoked_esi = evoked_esi.plot()
plt.title('从预测源重建的MI-EEG信号')
set_chinese_font_for_figure(fig_evoked_esi)  # 应用中文字体设置

# 修复plot_topomap参数 - 删除title参数
fig = evoked_esi.plot_topomap(times=[0.5, 1.0, 2.0, 3.0], time_format='%0.1f s')
fig.suptitle('从预测源重建的MI-EEG地形图')
set_chinese_font_for_figure(fig)  # 应用中文字体设置

# 使用eLORETA进行比较分析
print("使用eLORETA进行MI-EEG信号的源定位...")
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
add_text_with_chinese(brain, 0.1, 0.9, 'eLORETA对MI-EEG数据的分析', 'title', font_size=14)

# 绘制预测的EEG
evoked_elor = util.get_eeg_from_source(stc_elor, fwd, left_epochs_eeg_ref.info, tmin=0.)
fig_evoked_elor = evoked_elor.plot()
plt.title('eLORETA重建的MI-EEG信号')
set_chinese_font_for_figure(fig_evoked_elor)  # 应用中文字体设置

# 修复plot_topomap参数 - 删除title参数
fig = evoked_elor.plot_topomap(times=[0.5, 1.0, 2.0, 3.0], time_format='%0.1f s')
fig.suptitle('eLORETA重建的MI-EEG地形图')
set_chinese_font_for_figure(fig)  # 应用中文字体设置

print("MI-EEG源定位分析完成！")

