# 使用esinet预测样本ERP数据集的单一时间帧源：Brainstorm听觉数据。
# 本教程基于mne-python教程

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

subjects_dir = os.path.join(os.path.normpath(mne.datasets.sample.data_path()).replace("\\", "/"), 'subjects')
plot_params = dict(surface='inflated', cortex="low_contrast", hemi='both', verbose=0)
mne.set_config('SUBJECTS_DIR', subjects_dir)

# 加载数据
# 与mne-python教程一样，我们首先需要加载一些样本数据

data_path = os.path.join(os.path.normpath(mne.datasets.sample.data_path()).replace("\\", "/"))
raw_fname = os.path.join(data_path, 'MEG', 'sample',  
                    'sample_audvis_filt-0-40_raw.fif')

raw = mne.io.read_raw_fif(raw_fname, verbose=0)  # 已经具有平均参考
events = mne.find_events(raw, stim_channel='STI 014', verbose=0)

event_id = dict(aud_l=1)  # 事件触发器和条件
tmin = -0.2  # 每个时间段的开始（触发前200毫秒）
tmax = 0.5  # 每个时间段的结束（触发后500毫秒）
# raw.info['bads'] = ['MEG 2443', 'EEG 053']  # 没有带EEG的坏通道
baseline = (None, 0)  # 表示从第一个瞬间到t = 0
reject = dict(eeg=150e-6, eog=150e-6)  # 使用EEG而不是MEG的拒绝标准

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=('eeg', 'eog'), baseline=baseline, reject=reject,  # 使用eeg而不是grad
                    verbose=0, preload=True)

epochs.drop_channels('EOG 061')
fname_fwd = os.path.join(os.path.normpath(data_path).replace("\\", "/"), "MEG", "sample", "sample_audvis-eeg-oct-6-fwd.fif")
fwd = mne.read_forward_solution(fname_fwd, verbose=0)


epochs_stripped = epochs.copy().load_data().pick_types(eeg=True) # 使用eeg=True而不是meg=True
fwd = fwd.pick_channels(epochs_stripped.ch_names)
fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                                    use_cps=True, verbose=0)

# 可视化样本数据
epochs.average().plot(verbose=0)
epochs

# 模拟数据
# 人工神经网络需要训练数据来学习如何根据M/EEG数据预测脑电活动（源）。
# 首先，我们计算EEG数据的信噪比（SNR），以便调整我们的模拟。

target_snr = util.calc_snr_range(epochs, baseline_span=(-0.2, 0.0), data_span=(0.05, 0.2))
print(f'目标SNR为 {target_snr:.2f}')

# 接下来，我们可以使用包的默认设置执行模拟。如果此单元运行时间过长，请将n_samples更改为较小的整数。
# 注意，对于出版级别的逆解决方案，你应该将训练样本数量增加到100,000。

settings = dict(duration_of_trial=0.01, target_snr=target_snr)
n_samples = 1000
simulation = Simulation(fwd, epochs.info, settings=settings, verbose=True)
simulation.simulate(n_samples=n_samples)

# 可视化模拟数据
# 让我们可视化模拟数据，看看它是否正常。
# 您可以更改idx为另一个整数来可视化不同的样本

idx = 0

simulation.eeg_data[idx].average().plot_topomap([0.])
simulation.source_data[idx].plot(**plot_params, initial_time=simulation.source_data[idx].times[idx], time_viewer=False)

# 训练网络
# Net类包含我们的神经网络。
# 使用上面创建的模拟，我们可以训练神经网络。
# 这可能需要几分钟，取决于您的计算机。为了获得最佳结果，您应该将epochs数量增加到100或更多。

model_type = 'FC'  # 也可以是'LSTM'或'ConvDip'
net = Net(fwd, verbose=1, model_type=model_type)  # 初始化神经网络对象
net.fit(simulation, epochs=10)  # 使用我们模拟的eeg和源数据训练网络。

# 评估
# 评估是一个两步过程。
# 1.模拟神经网络尚未见过的一些测试数据。
# 与训练数据不同，这种模拟数据还将具有时间维度，如"duration_of_trial"参数所示。
# 2.对这些数据进行预测并直观检查结果。

settings = dict(duration_of_trial=0.02, target_snr=target_snr, number_of_sources=(1, 10), extents=(2, 40),)
n_samples = 10
simulation_test = Simulation(fwd, epochs.info, settings=settings, verbose=True)
simulation_test.simulate(n_samples=n_samples)


# 从EEG预测源
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
# 绘制模拟源
a = source.plot(**plot_params)
a.add_text(0.1, 0.9, '真实值', 'title',
               font_size=14)
# 绘制模拟EEG
evoked = simulation_test.eeg_data[idx].average()
evoked.plot()
evoked.plot_topomap([0.0,])

# 绘制esinet预测
b = source_hat[idx].plot(**plot_params)
b.add_text(0.1, 0.9, '预测', 'title',
               font_size=14)

# 绘制预测的EEG
evoked_hat = util.get_eeg_from_source(source_hat[idx], fwd, epochs_stripped.info, tmin=0)
evoked_hat.plot()
evoked_hat.plot_topomap([0.0,])

# 真实数据源定位
# 使用ANN
# 预测源
stc = net.predict(epochs.average())[0]
# 绘制预测源
brain = stc.plot(**plot_params)
brain.add_text(0.1, 0.9, 'esinet对听觉数据的预测', 'title',
               font_size=14)
# 绘制真实EEG
epochs.load_data()

epochs.pick(["eeg",]).average().plot()
epochs.pick(["eeg",]).average().plot_topomap()

# 绘制预测的EEG
evoked_esi = util.get_eeg_from_source(stc, fwd, epochs.pick_types(eeg=True).info, tmin=0.)
evoked_esi.plot()
evoked_esi.plot_topomap()

# 使用eLORETA
method = "eLORETA"
snr = 3.
lambda2 = 1. / snr ** 2
noise_cov = mne.compute_covariance(
    epochs, tmax=0., method=['shrunk', 'empirical'], rank=None, verbose=False)

inverse_operator = mne.minimum_norm.make_inverse_operator(
    evoked.info, fwd, noise_cov, loose='auto', depth=None, fixed=True, 
    verbose=False)
    
stc_elor, residual = mne.minimum_norm.apply_inverse(epochs.average(), inverse_operator, lambda2,
                              method=method, return_residual=True, verbose=False)
brain = np.abs(stc_elor).plot(**plot_params)
brain.add_text(0.1, 0.9, 'eLORETA对听觉数据的分析', 'title',
               font_size=14)
# 绘制预测的EEG
epochs.load_data()
evoked_elor = util.get_eeg_from_source(stc_elor, fwd, epochs.pick_types(eeg=True).info, tmin=0.)
evoked_elor.plot()
evoked_elor.plot_topomap()

