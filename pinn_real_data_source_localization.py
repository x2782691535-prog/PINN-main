#!/usr/bin/env python3
"""
PINN源定位脚本 - 用于真实数据集
使用esinet.net.py中的PINN模型对源定位真实数据集进行源定位分析

数据集：E:\pycharm\PINN\PINN-main\源定位真实数据集
该数据集包含7个被试的高密度EEG数据和颅内电刺激ground truth
"""

import numpy as np
import os
import sys
import mne
import random
import time
import glob
import shutil

# 在导入TensorFlow之前设置环境变量，抑制警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 抑制INFO和WARNING信息
import tensorflow as tf

# 关键修复：重置随机种子以确保每次运行产生不同的结果
def reset_random_seeds():
    """重置所有随机种子以确保真正的随机性"""
    current_time = int(time.time() * 1000000) % 1000000  # 使用微秒级时间戳
    np.random.seed(current_time)
    random.seed(current_time + 1)
    tf.random.set_seed(current_time + 2)
    print(f"🎲 重置随机种子: numpy={current_time}, random={current_time + 1}, tf={current_time + 2}")

# 立即重置随机种子
reset_random_seeds()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
import pickle
from pathlib import Path

# 添加esinet路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'esinet'))

# 导入esinet模块
try:
    from esinet.net import Net
    from esinet.simulation import Simulation
    from esinet.util.util import source_to_sourceEstimate
    from esinet import util
    print("✅ esinet模块导入成功")
except ImportError as e:
    print(f"❌ esinet导入失败: {e}")
    sys.exit(1)

# 全局定义修复版本的Simulation类，解决pickle序列化问题
class FixedSimulation(Simulation):
    def generate_balanced_sources(self, number_of_sources):
        """修复版本：使用连续的源空间索引而不是原始顶点编号"""
        # 获取实际可用的源点数量
        n_sources_total = self.fwd['nsource']
        
        if number_of_sources == 1:
            # 对于单源，随机选择一个安全的索引
            safe_idx = np.random.randint(0, n_sources_total)
            return np.array([safe_idx])
        else:
            # 对于多源，确保所有索引都在安全范围内
            safe_indices = np.random.choice(
                n_sources_total, 
                size=number_of_sources, 
                replace=False
            )
            return safe_indices

# 设置TensorFlow
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True) if tf.config.experimental.list_physical_devices('GPU') else None

# 数据集路径
DATASET_PATH = r"E:\pycharm\PINN\PINN-main\源定位真实数据集"

class RealDataPINNLocalizer:
    """使用PINN进行真实数据源定位的类"""
    
    def __init__(self, dataset_path=DATASET_PATH):
        self.dataset_path = Path(dataset_path)
        self.subjects = [f"sub-{i:02d}" for i in range(1, 8)]  # sub-01 到 sub-07
        self.eeg_data = {}
        self.ground_truth_positions = {}
        self.forward_models = {}
        self.electrode_positions = {}
        self.pinn_model = None
        self.model_type = 'pinn'  # 默认使用PINN模型，可选: fc, lstm, cnn
        self.source_time_course = 'pulse'  # 默认使用脉冲波形，可选: sine, random
        
    def load_forward_model(self, subject):
        """加载前向模型"""
        fwd_path = self.dataset_path / "derivatives" / "sourcemodelling" / subject / "fwd" / f"{subject}_fwd.fif"
        if fwd_path.exists():
            try:
                fwd = mne.read_forward_solution(str(fwd_path))
                print(f"✅ 成功加载 {subject} 的前向模型")
                return fwd
            except Exception as e:
                print(f"❌ 加载 {subject} 前向模型失败: {e}")
                return None
        else:
            print(f"❌ 前向模型文件不存在: {fwd_path}")
            return None
    
    def load_eeg_data(self, subject, run=1):
        """加载EEG数据"""
        eeg_path = self.dataset_path / "derivatives" / "epochs" / subject / "eeg" / f"{subject}_task-seegstim_run-{run:02d}_epochs.npy"
        
        if eeg_path.exists():
            try:
                eeg_data = np.load(str(eeg_path))
                print(f"✅ 成功加载 {subject} run-{run:02d} EEG数据，形状: {eeg_data.shape}")
                return eeg_data
            except Exception as e:
                print(f"❌ 加载 {subject} EEG数据失败: {e}")
                return None
        else:
            print(f"❌ EEG数据文件不存在: {eeg_path}")
            return None
    
    def load_ground_truth_positions(self, subject):
        """加载ground truth电极位置"""
        ieeg_path = self.dataset_path / "derivatives" / "epochs" / subject / "ieeg" / f"{subject}_task-seegstim_space-T1w_electrodes.tsv"
        
        if ieeg_path.exists():
            try:
                import pandas as pd
                electrodes_df = pd.read_csv(str(ieeg_path), sep='\t')
                positions = electrodes_df[['x', 'y', 'z']].values
                electrode_names = electrodes_df['name'].values
                print(f"✅ 成功加载 {subject} ground truth位置，数量: {len(positions)}")
                return positions, electrode_names
            except Exception as e:
                print(f"❌ 加载 {subject} ground truth位置失败: {e}")
                return None, None
        else:
            print(f"❌ Ground truth文件不存在: {ieeg_path}")
            return None, None
    
    def load_electrode_positions(self, subject):
        """加载EEG电极位置"""
        elec_path = self.dataset_path / "derivatives" / "epochs" / subject / "eeg" / f"{subject}_task-seegstim_electrodes.tsv"
        
        if elec_path.exists():
            try:
                import pandas as pd
                electrodes_df = pd.read_csv(str(elec_path), sep='\t')
                positions = electrodes_df[['x', 'y', 'z']].values
                electrode_names = electrodes_df['name'].values
                print(f"✅ 成功加载 {subject} EEG电极位置，数量: {len(positions)}")
                return positions, electrode_names
            except Exception as e:
                print(f"❌ 加载 {subject} EEG电极位置失败: {e}")
                return None, None
        else:
            print(f"❌ EEG电极位置文件不存在: {elec_path}")
            return None, None
    
    def get_stimulation_info(self, subject, run=1):
        """获取刺激信息"""
        stim_path = self.dataset_path / "derivatives" / "epochs" / subject / "eeg" / f"{subject}_task-seegstim_run-{run:02d}_epochs.json"
        
        if stim_path.exists():
            try:
                with open(str(stim_path), 'r') as f:
                    stim_info = json.load(f)
                print(f"✅ 成功加载 {subject} run-{run:02d} 刺激信息")
                return stim_info
            except Exception as e:
                print(f"❌ 加载 {subject} 刺激信息失败: {e}")
                return None
        else:
            print(f"❌ 刺激信息文件不存在: {stim_path}")
            return None
    
    def create_aligned_info_object(self, subject):
        """创建与真实数据集对齐的info对象"""
        try:
            # 加载EEG电极位置
            electrode_positions, electrode_names = self.load_electrode_positions(subject)
            if electrode_positions is None:
                return None
            
            # 创建info对象
            ch_names = electrode_names.tolist()
            sfreq = 1000  # 假设采样率为1000Hz，根据实际数据调整
            ch_types = ['eeg'] * len(ch_names)
            
            info = mne.create_info(ch_names, sfreq, ch_types)
            
            # 设置电极位置
            montage = mne.channels.make_dig_montage(
                ch_pos=dict(zip(ch_names, electrode_positions)),
                coord_frame='head'
            )
            info.set_montage(montage)
            
            print(f"✅ 创建info对象成功，通道数: {len(ch_names)}")
            return info
            
        except Exception as e:
            print(f"❌ 创建info对象失败: {e}")
            return None
    
    def calculate_target_snr_from_real_data(self, subject, runs=[1, 2, 3]):
        """从真实数据计算目标SNR"""
        try:
            snr_values = []
            
            for run in runs:
                eeg_data = self.load_eeg_data(subject, run)
                if eeg_data is None:
                    continue
                
                # 计算SNR（简化版本）
                if eeg_data.ndim == 3:  # (trials, channels, time)
                    # 假设前25%时间为基线，后75%为信号
                    baseline_end = eeg_data.shape[2] // 4
                    baseline = eeg_data[:, :, :baseline_end]
                    signal = eeg_data[:, :, baseline_end:]
                    
                    baseline_power = np.mean(np.var(baseline, axis=2))
                    signal_power = np.mean(np.var(signal, axis=2))
                    
                    if baseline_power > 0:
                        snr = 10 * np.log10(signal_power / baseline_power)
                        snr_values.append(snr)
            
            if snr_values:
                target_snr = np.mean(snr_values)
                print(f"   - 从真实数据计算的目标SNR: {target_snr:.2f} dB")
                return max(target_snr, 1.0)  # 确保SNR不小于1
            else:
                print("   - 无法计算SNR，使用默认值: 5 dB")
                return 5.0
                
        except Exception as e:
            print(f"❌ 计算SNR失败: {e}")
            return 5.0
    
    def create_aligned_simulation_settings(self, subject):
        """创建与真实数据集对齐的模拟设置"""
        try:
            print("🔧 分析真实数据特征...")
            
            # 1. 从真实数据计算SNR
            target_snr = self.calculate_target_snr_from_real_data(subject)
            
            # 2. 分析刺激特征
            stim_info = self.get_stimulation_info(subject, run=1)
            if stim_info and 'Description' in stim_info:
                # 解析刺激描述，例如 "Stimulation of channel K13-14 1mA"
                description = stim_info['Description']
                print(f"   - 刺激描述: {description}")
            
            # 3. 获取数据时间特征
            eeg_data = self.load_eeg_data(subject, run=1)
            if eeg_data is not None:
                if eeg_data.ndim == 3:
                    trial_duration = eeg_data.shape[2] / 1000.0  # 假设1000Hz采样率
                else:
                    trial_duration = 0.5  # 默认值
            else:
                trial_duration = 0.5
            
            print(f"   - 估计试次时长: {trial_duration:.3f} 秒")
            
            # 4. 创建针对颅内电刺激的模拟设置（只使用允许的参数）
            aligned_settings = dict(
                method='standard',
                duration_of_trial=trial_duration,
                sample_frequency=1000,  # 与真实数据一致
                target_snr=target_snr,
                number_of_sources=1,  # 颅内电刺激通常是单点源
                extents=(1, 3),      # 较小的源范围，模拟电极刺激
                amplitudes=(1, 10),   # 适中的幅度范围
                shapes='gaussian',    # 高斯分布，模拟电刺激的局部性
                beta_source=(0.5, 1.0),  # 较低的beta值，模拟电刺激特征
                beta=(1, 2),         # 噪声的频谱特征
                beta_noise=(1, 2),   # 噪声beta值
                source_spread='spherical',  # 球形扩散，模拟电刺激
                source_number_weighting=False,  # 不使用源数量加权
                source_time_course='random',  # 使用允许的时间过程
            )
            
            print("✅ 模拟设置创建完成")
            print(f"   - SNR: {target_snr:.2f} dB")
            print(f"   - 试次时长: {trial_duration:.3f} 秒")
            print(f"   - 采样频率: {aligned_settings['sample_frequency']} Hz")
            print(f"   - 源数量: {aligned_settings['number_of_sources']}")
            print(f"   - 源范围: {aligned_settings['extents']} mm")
            print(f"   - 源形状: {aligned_settings['shapes']}")
            
            return aligned_settings
            
        except Exception as e:
            print(f"❌ 创建模拟设置失败: {e}")
            return None
    
    def setup_pinn_model_like_bciiv2a(self, fwd, subject, n_samples=1000):
        """完全按照BCIIV2a.py的方式设置PINN模型"""
        try:
            print("🔧 按照BCIIV2a.py方式创建模拟数据...")
            
            # 1. 关键修复：按照BCIIV2a.py转换前向模型为固定方向
            print("🔧 转换前向模型为固定方向（仿BCIIV2a.py）...")
            if not fwd['surf_ori']:
                print("   - 检测到自由方向前向模型，转换为固定方向")
                fwd_fixed = mne.convert_forward_solution(
                    fwd, surf_ori=True, force_fixed=True,
                    use_cps=True, verbose=False
                )
                print(f"   - 转换后源点数: {fwd_fixed['nsource']}")
            else:
                print("   - 前向模型已是固定方向")
                fwd_fixed = fwd
            
            # 2. 创建对齐的info对象
            info = self.create_aligned_info_object(subject)
            if info is None:
                return None
            
            # 3. 根据真实数据集特征设置模拟参数
            # 从之前的输出可知：真实数据形状 (38, 256, 2081)
            # 2081个时间点，如果采样率1000Hz = 2.081秒
            real_data_settings = dict(
                duration_of_trial=0.5,      # 大幅减少时长，避免内存问题
                sample_frequency=250,       # 降低采样率，减少内存占用
                target_snr=5,               # 适中的SNR
                number_of_sources=1,        # 颅内电刺激通常是单源
                extents=(5, 15),           # 适中的源范围
                beta_source=(1, 1.5),      # 适中的beta参数
                source_time_course=self.source_time_course   # 根据参数选择时间过程
            )
            
            print(f"✅ 模拟设置（仿BCIIV2a.py）: {real_data_settings}")
            
            # 4. 检查是否存在已保存的模拟数据（仿BCIIV2a.py的缓存机制）
            SIM_TRAIN_PATH = f'{subject}_real_data_simulation_fixed.pkl'
            
            # 检查是否存在模拟数据，如果不存在才重新生成
            if os.path.exists(SIM_TRAIN_PATH):
                print(f'📁 发现已保存的模拟数据，直接加载: {SIM_TRAIN_PATH}')
                try:
                    with open(SIM_TRAIN_PATH, 'rb') as f:
                        simulation = pickle.load(f)
                    print(f'✅ 成功加载训练模拟数据')
                except Exception as e:
                    print(f'❌ 加载模拟数据失败: {e}，将重新生成')
                    simulation = None
            else:
                print('🔄 未检测到训练模拟数据，开始生成...')
                simulation = None
            
            if simulation is None:
                # 再次重置随机种子以确保模拟数据的随机性
                reset_random_seeds()
                
                # 使用全局定义的修复版本Simulation类，大幅减少样本数量避免内存问题
                print("   - 生成300个样本（避免内存不足）")
                simulation = FixedSimulation(fwd_fixed, info, settings=real_data_settings, verbose=False)
                simulation.simulate(n_samples=300)  # 大幅减少样本数量
                
                # 保存模拟数据（仿BCIIV2a.py）
                with open(SIM_TRAIN_PATH, 'wb') as f:
                    pickle.dump(simulation, f)
                print(f'✅ 训练模拟数据已保存到: {SIM_TRAIN_PATH}')
            
            # 5. 初始化模型（根据model_type参数）
            print(f"🔧 初始化{self.model_type.upper()}模型...")

            # 根据model_type参数选择模型
            if self.model_type.lower() == 'pinn':
                # 修复：不传递model_type='pinn'，使用默认的'auto'，避免触发特殊的PINN编译逻辑
                self.pinn_model = Net(fwd_fixed, use_pinn=True,
                                     physics_weight=0.4, rescale_sources='rms')
                if not hasattr(self.pinn_model, 'dropout'):
                    self.pinn_model.dropout = 0.2
                self.pinn_model._build_pinn_model()

            elif self.model_type.lower() == 'fc':
                # FC模型：关闭物理约束
                self.pinn_model = Net(fwd_fixed, model_type='fc', use_pinn=False,
                                     n_dense_layers=3, n_dense_units=200, rescale_sources='rms')
                if not hasattr(self.pinn_model, 'dropout'):
                    self.pinn_model.dropout = 0.2
                self.pinn_model._build_fc_model()

            elif self.model_type.lower() == 'lstm':
                # LSTM模型：关闭物理约束
                self.pinn_model = Net(fwd_fixed, model_type='lstm', use_pinn=False,
                                     n_lstm_layers=2, n_lstm_units=32, n_dense_units=200, rescale_sources='rms')
                if not hasattr(self.pinn_model, 'dropout'):
                    self.pinn_model.dropout = 0.2
                self.pinn_model._build_temporal_model()

            elif self.model_type.lower() == 'cnn':
                # CNN模型：关闭物理约束
                self.pinn_model = Net(fwd_fixed, model_type='cnn', use_pinn=False,
                                     n_filters=64, n_lstm_units=32, rescale_sources='rms')
                if not hasattr(self.pinn_model, 'dropout'):
                    self.pinn_model.dropout = 0.2
                self.pinn_model._build_cnn_model()
            else:
                raise ValueError(f"不支持的模型类型: {self.model_type}")
            
            # 保存固定方向的前向模型供后续使用
            self.fwd_fixed = fwd_fixed
            
            print(f"✅ {self.model_type.upper()}模型设置完成（仿BCIIV2a.py方式）")
            return simulation
            
        except Exception as e:
            print(f"❌ {self.model_type.upper()}模型设置失败: {e}")
            print(f"错误详情: {str(e)}")
            return None
    
    def train_pinn_model(self, simulation, epochs=100, batch_size=32, learning_rate=0.001, patience=7):
        """训练神经网络模型"""
        try:
            print(f"🚀 开始训练{self.model_type.upper()}模型...")
            print(f"   - 训练样本数: {simulation.n_samples}")
            print(f"   - 训练轮数: {epochs}")
            print(f"   - 批次大小: {batch_size}")
            print(f"   - 学习率: {learning_rate}")
            
            # 使用esinet的训练方法 (TensorFlow后端)
            self.pinn_model.fit(
                simulation, 
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                patience=patience
            )
            
            print(f"✅ {self.model_type.upper()}模型训练完成")
            return True
            
        except Exception as e:
            print(f"❌ {self.model_type.upper()}模型训练失败: {e}")
            return False
    
    def perform_source_localization(self, eeg_data, fwd):
        """执行源定位（保留原始神经网络方法）"""
        try:
            print(f"🎯 执行{self.model_type.upper()}源定位...")
            # 预处理EEG数据
            if eeg_data.ndim == 3:  # (trials, channels, time)
                # 仅做试验平均，去掉时间平均！
                # eeg_data = np.mean(eeg_data, axis=2)  # ←删除此行
                eeg_data = np.mean(eeg_data, axis=0)  # 试验平均，(channels, time)
                print(f"   - 试验平均后形状: {eeg_data.shape}")
            elif eeg_data.ndim == 2:  # (channels, time) 或 (trials, channels)
                if eeg_data.shape[1] > eeg_data.shape[0]:  # (channels, time)
                    print(f"   - 输入原始为(通道,时间)，不再做平均，形状: {eeg_data.shape}")
                else:  # (trials, channels)
                    eeg_data = np.mean(eeg_data, axis=0)  # 试验平均
                    print(f"   - (trials,channels) 试验平均->(channels,): {eeg_data.shape}")
            # 确保数据形状正确 for Evoked
            if eeg_data.ndim == 1:
                eeg_data = eeg_data.reshape(1, -1)
            print(f"   - 进Evoked前shape: {eeg_data.shape}")
            
            # 使用神经网络进行源重建 (TensorFlow后端)
            # 创建MNE Evoked对象，因为esinet.predict期望MNE对象
            # 关键修复：使用前向模型中的通道名，而不是自己生成的通道名
            print(f"   - 前向模型通道数: {len(self.fwd_fixed.ch_names)}")
            print(f"   - 前向模型通道名示例: {self.fwd_fixed.ch_names[:5]}")
            
            # 核心修正：只比较通道数，不动时间长度！
            if eeg_data.shape[0] != len(self.fwd_fixed.ch_names):
                print(f"   ⚠️ 通道数不匹配: EEG数据有{eeg_data.shape[0]}，模型期望{len(self.fwd_fixed.ch_names)}")
                if eeg_data.shape[0] > len(self.fwd_fixed.ch_names):
                    eeg_data = eeg_data[:len(self.fwd_fixed.ch_names), :]
                    print(f"   - 截取为{len(self.fwd_fixed.ch_names)}通道")
                else:
                    n_times = eeg_data.shape[1]
                    padding = np.zeros((len(self.fwd_fixed.ch_names) - eeg_data.shape[0], n_times))
                    eeg_data = np.vstack([eeg_data, padding])
                    print(f"   - 填充为{len(self.fwd_fixed.ch_names)}通道")

            ch_names = self.fwd_fixed.ch_names
            info = mne.create_info(ch_names, 1000, 'eeg')
            # 直接创建Evoked对象，无需转置：shape=(n_channels, n_times)
            evoked = mne.EvokedArray(eeg_data, info, tmin=0)
            
            # 添加详细的调试信息和错误追踪
            print(f"   - Evoked对象创建成功")
            print(f"   - Evoked数据形状: {evoked.data.shape}")
            print(f"   - Evoked通道数: {len(evoked.ch_names)}")
            print(f"   - {self.model_type.upper()}模型类型: {type(self.pinn_model)}")
            print(f"   - 前向模型通道名匹配检查: {evoked.ch_names[:5]} vs {self.fwd_fixed.ch_names[:5]}")
            
            # 尝试预测并捕获详细错误
            try:
                print("   - 开始调用 pinn_model.predict()...")
                print(f"   - 输入evoked数据的前5个通道的均值: {np.mean(evoked.data[:5, :], axis=1)}")
                # evoked对象创建后, evoked.data.shape通常是(channels, times)，需要增加batch
                data = evoked.data
                # 训练同款缩放：逐时间点减均值并按标准差标准化（CAR + 标准化）
                data = np.array(data, dtype=np.float32, copy=True)
                data -= np.mean(data, axis=0, keepdims=True)
                std = np.std(data, axis=0, keepdims=True)
                std[std == 0] = 1.0
                data /= std
                print(f"   - 归一化后data形状: {data.shape}")
                
                if data.ndim == 2:
                    data = np.expand_dims(data, axis=0)  # (channels, times) -> (1, channels, times)
                elif data.ndim == 1:
                    data = data[np.newaxis, :, np.newaxis]
                
                print(f"   - 扩展维度后data形状: {data.shape}")
                
                # 关键修复：PINN模型期望(batch, time, channels)，需要转置
                # 从(1, channels, times)转换为(1, times, channels)
                data = np.transpose(data, (0, 2, 1))
                print(f"   - 转置后data形状: {data.shape}")
                
                # 直接调用底层模型，兼容PINN双输出(dict)
                outputs = self.pinn_model.model.predict(data, verbose=0)
                if isinstance(outputs, dict):
                    source_estimate = outputs.get('main_output')
                    if isinstance(source_estimate, (list, tuple, np.ndarray)):
                        source_estimate = source_estimate[0]  # (time, n_sources)
                else:
                    # 单输出模型: (batch, time, n_sources)
                    source_estimate = outputs[0]
                print("   - predict() 调用成功")
            except Exception as e:
                print(f"   ❌ 详细错误信息: {str(e)}")
                print(f"   ❌ 错误类型: {type(e)}")
                import traceback
                print(f"   ❌ 完整错误堆栈:")
                traceback.print_exc()
                
                # 尝试检查 evoked 对象的内部状态
                print(f"   - 调试信息:")
                print(f"     * evoked.data 类型: {type(evoked.data)}")
                print(f"     * evoked.data 形状: {evoked.data.shape}")
                print(f"     * evoked.data 数据类型: {evoked.data.dtype}")
                print(f"     * evoked.info['nchan']: {evoked.info['nchan']}")
                
                raise e
            
            print(f"✅ {self.model_type.upper()}源定位完成")
            return source_estimate
            
        except Exception as e:
            print(f"❌ {self.model_type.upper()}源定位失败: {e}")
            return None
    
    def calculate_localization_error(self, predicted_sources, true_positions, source_positions):
        """返回当前run下MLE/AUC结果，不降维，聚焦最大响应时刻（如有多时间点，自动选最高激活）"""
        import numpy as np
        from sklearn.metrics import roc_auc_score, average_precision_score

        # 1. 获取最大响应时刻的源激活
        # 支持(predicted_sources).shape = (n_times, n_sources) or (n_sources,)
        if hasattr(predicted_sources, 'data'):
            data = predicted_sources.data
        else:
            data = np.asarray(predicted_sources)
        data = np.squeeze(data)

        if data.ndim == 2:
            # 每一时刻都max，选择time维度下总激活最大的那个时间点
            max_time_idx = np.argmax(np.max(np.abs(data), axis=1))
            src_activation = data[max_time_idx]  # (n_sources,)
            print(f"   [DEBUG] 当前run max_time_idx: {max_time_idx}, src_activation的max: {np.max(np.abs(src_activation))}")
        else:
            src_activation = data
            print(f"   [DEBUG] 当前run (单时间点) src_activation的max: {np.max(np.abs(src_activation))}")

        # 2. 预测点
        pred_idx = np.argmax(np.abs(src_activation))
        pred_pos = source_positions[pred_idx]

        # 3. GT点
        if len(true_positions.shape) == 1:
            gt_pos = true_positions[np.newaxis]
        else:
            gt_pos = true_positions
        # 计算所有真实点到预测点距离，MLE为最小距离（每run允许多个真点）
        dists = np.sqrt(np.sum((gt_pos - pred_pos) ** 2, axis=1)) * 1000
        mle = np.min(dists)

        # 4. 构造二元标签做AUC，认为距离最近的一个GT点是正，其余为负
        gt_mask = np.zeros(len(source_positions), dtype=int)
        # 最小距离点作为唯一正例
        closest_gt = np.argmin(np.linalg.norm(source_positions - gt_pos[np.argmin(dists)], axis=1))
        gt_mask[closest_gt] = 1

        pred_score_norm = np.abs(src_activation)
        if np.max(pred_score_norm) > 0:
            pred_score_norm = pred_score_norm / np.max(pred_score_norm)
        try:
            auc = roc_auc_score(gt_mask, pred_score_norm)
        except:
            auc = 0.5
        try:
            auc_pr = average_precision_score(gt_mask, pred_score_norm)
        except:
            auc_pr = 0.5

        return {'mle': float(mle),
                'auc': float(auc),
                'auc_pr': float(auc_pr),
                'pred_idx': int(pred_idx),
                'pred_pos': pred_pos.tolist(),
                'gt_pos': gt_pos[np.argmin(dists)].tolist()}
    
    def extract_run_gt_positions(self, stim_info, electrode_positions, electrode_names):
        """根据run的刺激信息提取该run专属的GT电极三维位置集合。
        支持多种字段：channels/channel/anodes/cathodes/Description。
        若解析失败，回退为全体电极（保持兼容）。"""
        try:
            if stim_info is None:
                return electrode_positions
            # 统一提取可能的电极名
            names = []
            # 显式字段
            for key in ['channels', 'channel', 'anodes', 'cathodes', 'stim_channels', 'stim']:
                v = stim_info.get(key)
                if isinstance(v, str):
                    names.append(v)
                elif isinstance(v, (list, tuple)):
                    for item in v:
                        if isinstance(item, str):
                            names.append(item)
            # 从描述里解析形如 "K13-14" 或含空格
            desc = stim_info.get('Description') or stim_info.get('description') or ''
            if isinstance(desc, str):
                # 预处理：去单位等
                desc_clean = desc.replace('Stimulation of channel', '').replace('channels', '').replace('channel', '')
                desc_clean = desc_clean.replace('电极', '').replace('刺激', '')
                # 分隔符尝试
                for token in desc_clean.replace(',', ' ').replace(';', ' ').split():
                    if any(ch.isalpha() for ch in token):
                        names.append(token)
            # 规范化：拆分连字符对，如 "K13-14"
            parsed = []
            for n in names:
                n = n.strip()
                if '-' in n and not n.strip().startswith('-') and not n.strip().endswith('-'):
                    left,right = n.split('-',1)
                    left = left.strip()
                    right = right.strip()
                    # 右半部可能只给数字，如 13-14 => 需要补全前缀
                    if left and right and right[0].isdigit() and not any(c.isalpha() for c in right):
                        # 提取左侧字母前缀
                        prefix = ''.join([c for c in left if c.isalpha()])
                        parsed.extend([left, f"{prefix}{right}"])
                    else:
                        parsed.extend([left, right])
                else:
                    parsed.append(n)
            # 去重并比对electrode_names
            name_set = set([p.strip() for p in parsed if p and p.strip()])
            if not name_set:
                return electrode_positions
            name_to_idx = {str(electrode_names[i]): i for i in range(len(electrode_names))}
            idxs = [name_to_idx[n] for n in name_set if n in name_to_idx]
            if not idxs:
                return electrode_positions
            return electrode_positions[idxs]
        except Exception:
            return electrode_positions
    
    def get_run_numbers(self, subject):
        """自动获取指定subject的run编号列表，按文件排序"""
        epoch_dir = os.path.join('源定位真实数据集', 'derivatives', 'epochs', subject, 'eeg')
        files = glob.glob(os.path.join(epoch_dir, f'{subject}_task-seegstim_run-*_epochs.npy'))
        run_numbers = []
        for file in files:
            fname = os.path.basename(file)
            parts = fname.split('_')
            for part in parts:
                if part.startswith('run-'):
                    try:
                        run_numbers.append(int(part[4:]))
                    except:
                        pass
        run_numbers = sorted(set(run_numbers))
        return run_numbers
    
    def run_analysis(self, subject="sub-01"):
        print(f"🚀 开始分析被试: {subject}")
        print("="*50)

        run_numbers = self.get_run_numbers(subject)
        print(f"   - 检测到 {len(run_numbers)} 个run: {run_numbers}")

        # 1. 加载前向模型
        fwd = self.load_forward_model(subject)
        if fwd is None:
            return None
        # 2. 设置模型
        print(f"🧠 使用{self.model_type.upper()}方法（仿BCIIV2a.py模拟数据生成）")
        simulation = self.setup_pinn_model_like_bciiv2a(fwd, subject, n_samples=100)
        if simulation is None:
            return None
        # 3. 获取源空间位置
        source_positions = self.fwd_fixed['src'][0]['rr'][self.fwd_fixed['src'][0]['vertno']]
        if len(self.fwd_fixed['src']) > 1:
            source_positions = np.vstack([
                source_positions,
                self.fwd_fixed['src'][1]['rr'][self.fwd_fixed['src'][1]['vertno']]
            ])
        print(f"   - 源空间点数: {len(source_positions)}")

        # 4. 加载ground truth位置
        true_positions, electrode_names = self.load_ground_truth_positions(subject)
        if true_positions is None:
            return None
        # 5. 训练PINN模型
        success = self.train_pinn_model(simulation, epochs=2000, batch_size=16, patience=15)
        if not success:
            return None
        # 6. 所有run统一作为testing set批量统计
        results = []
        for run in run_numbers:
            print(f"\n📊 处理 run-{run:02d}...")
            eeg_data = self.load_eeg_data(subject, run)
            if eeg_data is None:
                continue
            stim_info = self.get_stimulation_info(subject, run)
            source_estimate = self.perform_source_localization(eeg_data, self.fwd_fixed)
            if source_estimate is None:
                continue
            # 为该run构造专属GT
            run_true_positions = self.extract_run_gt_positions(stim_info, true_positions, electrode_names)
            run_metrics = self.calculate_localization_error(
                source_estimate, run_true_positions, source_positions)
            print(f"[DEBUG] run-{run:02d}: pred_idx={run_metrics['pred_idx']} pred_pos={run_metrics['pred_pos']} gt_pos={run_metrics['gt_pos']}")
            # 打印峰值激活信息：
            # 注：src_activation的max值已在calculate_localization_error打印
            results.append({
                'run': run,
                'metrics': run_metrics,
                'stimulation_info': stim_info
            })
        # 7. 统一输出和保存整体均值/方差，所有run
        if results:
            mles = [r['metrics']['mle'] for r in results]
            aucs = [r['metrics']['auc'] for r in results]
            auc_prs = [r['metrics']['auc_pr'] for r in results]
            avg_mle = np.mean(mles)
            std_mle = np.std(mles)
            avg_auc = np.mean(aucs)
            avg_auc_pr = np.mean(auc_prs)
            # 计算最小MLE与最大AUC
            min_mle_idx = int(np.argmin(mles))
            min_mle = mles[min_mle_idx]
            min_mle_run = results[min_mle_idx]['run']
            max_auc_idx = int(np.argmax(aucs))
            max_auc = aucs[max_auc_idx]
            max_auc_run = results[max_auc_idx]['run']

            print(f"\n📈 {subject} 测试集评估结果 ({self.model_type.upper()}, all runs):")
            print(f"   - Testing samples (runs): {len(results)}")
            print(f"   - MLE: {avg_mle:.2f} ± {std_mle:.2f} mm")
            print(f"   - AUC: {avg_auc:.4f}")
            print(f"   - PR-AUC: {avg_auc_pr:.4f}")
            print(f"   - Min MLE: {min_mle:.2f} mm (run-{min_mle_run:02d})")
            print(f"   - Max AUC: {max_auc:.4f} (run-{max_auc_run:02d})")

            # 保存统一结构
            self.save_results(results, avg_mle, std_mle)
            return results
        else:
            print(f"❌ {subject} 没有成功的analysis结果")
            return None
    
    def save_results(self, results, avg_mle, std_mle):
        """保存分析结果 (已适配新结构)"""
        try:
            results_dir = Path("results_pinn_real_data")
            results_dir.mkdir(exist_ok=True)
            results_file = results_dir / f"{self.model_type}_localization_results_{results[0]['run']}.json"
            save_data = {
                'testing_samples': len(results),
                'average_mle_mm': float(avg_mle),
                'std_mle_mm': float(std_mle),
                'detailed': []
            }
            for result in results:
                m = result['metrics']
                save_data['detailed'].append({
                    'run': result['run'],
                    'mle_mm': m['mle'],
                    'auc': m['auc'],
                    'pr_auc': m['auc_pr'],
                    'pred_pos': m['pred_pos'],
                    'gt_pos': m['gt_pos'],
                    'stimulation_info': result['stimulation_info'],
                })
            with open(results_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            print(f"✅ 结果已保存到: {results_file}")
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")


if __name__ == "__main__":
    # ============ 配置参数（直接修改这里） ============
    SUBJECT = 'sub-01'              # 被试ID: sub-01 到 sub-07
    MODEL_TYPE = 'fc'             # 模型类型: 'pinn', 'fc', 'lstm', 'cnn'
    SOURCE_TIME_COURSE = 'pulse'    # 源时间过程: 'pulse'(脉冲), 'sine'(正弦), 'random'(随机)
    # ==============================================

    localizer = RealDataPINNLocalizer(DATASET_PATH)
    localizer.model_type = MODEL_TYPE
    localizer.source_time_course = SOURCE_TIME_COURSE

    print(f"\n{'='*60}")
    print(f"  模型类型: {MODEL_TYPE.upper()}")
    print(f"  源时间过程: {SOURCE_TIME_COURSE}")
    print(f"  被试: {SUBJECT}")
    print(f"{'='*60}\n")

    localizer.run_analysis(SUBJECT)
