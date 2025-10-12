# BCI-IV-2a左右手分类V3.py
# 使用PINN提取空域特征 + CNN提取时域特征的混合模型
# 实现MI-EEG信号的左右手分类

import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, Dataset, DataLoader, Subset
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
import mne
from mne.channels import make_standard_montage
from mne.io import RawArray
from mne.forward import make_forward_solution
import gc
import torch.nn.functional as F
import time
from scipy import signal
import copy
import contextlib
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib字体为微软黑体，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# =================================================================================
# PINN源定位网络模块
# =================================================================================
class PINNSourceLocalization(nn.Module):
    """
    物理信息神经网络(PINN)源定位模块
    适配真实数据集，不依赖真实源位置标签
    """
    def __init__(self, input_dim, n_sources, leadfield, device='cuda'):
        super(PINNSourceLocalization, self).__init__()
        self.input_dim = input_dim
        self.n_sources = n_sources
        self.device = device
        
        # 引线场矩阵 (n_channels, n_sources)
        self.leadfield = leadfield.to(device)
        
        # 深度特征提取网络（类似BCIIV2a.py中的TimeDistributed Dense层）
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Dropout(0.25),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Dropout(0.25),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Dropout(0.25)
        ).to(device)
        
        # 源定位映射层（从深度特征映射到源空间）
        self.source_mapping = nn.Linear(512, n_sources).to(device)
        
        # 物理约束损失计算器
        self.physics_loss_calculator = self._create_physics_loss_calculator()
        
    def _create_physics_loss_calculator(self):
        """创建物理约束损失计算器"""
        class PhysicsLossCalculator(nn.Module):
            def __init__(self, leadfield, device):
                super().__init__()
                self.leadfield = leadfield
                self.device = device
                
                # 根据论文公式(4)的权重系数
                self.lambda_data = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
                self.lambda_phys = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
                self.lambda_reg = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
                
                # 正则化参数 (公式7)
                self.alpha_D = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
                self.alpha_R = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
                self.gamma = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
                
            def forward(self, source_activations, target_eeg=None):
                """
                计算完整的PINN损失函数，基于论文公式(4)
                L_total = λ_data * L_data + λ_phys * L_phys + λ_reg * L_reg
                """
                batch_size, n_sources = source_activations.shape
                
                # 1. Data Fidelity Loss (公式5) - 数据保真度损失
                L_data = self._compute_data_fidelity_loss(source_activations, target_eeg)
                
                # 2. Physics-Constrained Loss (公式6) - 物理约束损失
                L_phys = self._compute_physics_constrained_loss(source_activations, target_eeg)
                
                # 3. Auxiliary Regularization Loss (公式7) - 辅助正则化损失
                L_reg = self._compute_auxiliary_regularization_loss(source_activations)
                
                # 总损失 (公式4)
                total_loss = (torch.abs(self.lambda_data) * L_data + 
                            torch.abs(self.lambda_phys) * L_phys + 
                            torch.abs(self.lambda_reg) * L_reg)
                
                return total_loss
                
            def _compute_data_fidelity_loss(self, source_activations, target_eeg):
                """
                计算数据保真度损失 (公式5)
                L_data = 1 - cos(J_pred, J_true) = 1 - (J_pred · J_true) / (||J_pred||_2 × ||J_true||_2)
                """
                if target_eeg is None:
                    return torch.tensor(0.0, device=self.device)
                
                # 通过前向模型重建EEG信号
                J_pred = torch.matmul(source_activations, self.leadfield.t())  # 预测的电流密度
                J_true = target_eeg  # 真实的电流密度（这里用目标EEG近似）
                
                # 计算余弦相似度
                dot_product = torch.sum(J_pred * J_true, dim=1)
                norm_pred = torch.norm(J_pred, p=2, dim=1)
                norm_true = torch.norm(J_true, p=2, dim=1)
                
                # 避免除零
                norm_pred = torch.clamp(norm_pred, min=1e-8)
                norm_true = torch.clamp(norm_true, min=1e-8)
                
                cosine_sim = dot_product / (norm_pred * norm_true)
                cosine_sim = torch.clamp(cosine_sim, min=-1.0, max=1.0)
                
                # 数据保真度损失
                L_data = 1.0 - torch.mean(cosine_sim)
                return L_data

            def _compute_physics_constrained_loss(self, source_activations, target_eeg):
                """
                计算物理约束损失 (公式6)
                L_phys = ||φ_meas - L * J_pred||_2^2
                """
                if target_eeg is None:
                    return torch.tensor(0.0, device=self.device)
                
                # φ_meas: 测量的头皮电位
                phi_meas = target_eeg
                
                # L * J_pred: 通过引线场矩阵和预测的源活动计算的头皮电位
                phi_pred = torch.matmul(source_activations, self.leadfield.t())
                
                # 计算L2范数的平方
                L_phys = torch.mean((phi_meas - phi_pred) ** 2)
                return L_phys

            def _compute_auxiliary_regularization_loss(self, source_activations):
                """
                计算辅助正则化损失 (公式7)
                L_reg = α_D ||φ - g||_2^2 + α_R ||∂φ/∂n + γφ - h||_2^2
                """
                # 简化版本的正则化损失
                # 第一项：Dirichlet边界条件的正则化
                L1 = torch.abs(self.alpha_D) * torch.mean(source_activations ** 2)
                
                # 第二项：Robin边界条件的正则化（简化为梯度正则化）
                # 计算源活动的空间梯度（近似）
                if source_activations.shape[1] > 1:
                    grad_phi = source_activations[:, 1:] - source_activations[:, :-1]
                    robin_term = grad_phi + torch.abs(self.gamma) * source_activations[:, :-1]
                    L2 = torch.abs(self.alpha_R) * torch.mean(robin_term ** 2)
                else:
                    L2 = torch.tensor(0.0, device=self.device)
                
                L_reg = L1 + L2
                return L_reg
        
        return PhysicsLossCalculator(self.leadfield, self.device)
    
    def forward(self, x, target_eeg=None):
        """
        前向传播
        Args:
            x: 输入EEG数据 (batch_size, n_channels, n_timepoints)
            target_eeg: 目标EEG信号用于重建损失计算 (batch_size, n_channels)
        Returns:
            dict: 包含源激活和损失的字典
        """
        batch_size, n_channels, n_timepoints = x.shape
        
        # 时间维度平均，得到空间特征
        spatial_features = torch.mean(x, dim=2)  # (batch_size, n_channels)
        
        # 深度特征提取
        deep_features = self.feature_extractor(spatial_features)
        
        # 源定位映射
        source_activations = self.source_mapping(deep_features)
        
        # 计算物理约束损失
        physics_loss = self.physics_loss_calculator(source_activations, target_eeg)
        
        return {
            'source_activations': source_activations,
            'deep_features': deep_features,
            'physics_loss': physics_loss,
            'spatial_features': spatial_features
        }


# =================================================================================
# CNN时域特征提取模块
# =================================================================================
class CNNTemporalExtractor(nn.Module):
    """
    CNN时域特征提取器
    专门设计用于捕获EEG信号的时间动态特征
    """
    def __init__(self, in_channels=22, seq_length=1000, dropout_rate=0.35):
        super(CNNTemporalExtractor, self).__init__()
        
        # 平衡的时域卷积网络 - 防过拟合优化
        self.temporal_conv = nn.Sequential(
            # 第一层：捕获短时特征
            nn.Conv1d(in_channels, 20, kernel_size=5, padding=2),
            nn.BatchNorm1d(20),
            nn.LeakyReLU(0.1),
            nn.AvgPool1d(kernel_size=4),  # -> (batch, 20, 250)
            nn.Dropout(dropout_rate * 0.8),
            
            # 第二层：捕获中等时间尺度特征
            nn.Conv1d(20, 40, kernel_size=3, padding=1),
            nn.BatchNorm1d(40),
            nn.LeakyReLU(0.1),
            nn.AvgPool1d(kernel_size=5),  # -> (batch, 40, 50)
            nn.Dropout(dropout_rate),
            
            # 第三层：捕获长时特征
            nn.Conv1d(40, 80, kernel_size=3, padding=1),
            nn.BatchNorm1d(80),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool1d(5),  # -> (batch, 80, 5)
            nn.Dropout(dropout_rate),
        )
        
        self.flatten = nn.Flatten()
        self.output_dim = 80 * 5  # 400 (减少特征维度防止过拟合)
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入EEG数据 (batch_size, n_channels, n_timepoints)
        Returns:
            提取的时域特征 (batch_size, output_dim)
        """
        temporal_features = self.temporal_conv(x)
        temporal_features = self.flatten(temporal_features)
        return temporal_features


# =================================================================================
# PINN+CNN混合模型
# =================================================================================
class PINN_CNN_BCI(nn.Module):
    """
    PINN+CNN混合模型用于BCI-IV-2a左右手分类
    - PINN分支：提取空域特征
    - CNN分支：提取时域特征
    - 特征融合：自适应融合两种特征
    - 分类器：最终分类输出
    """
    def __init__(self, leadfield, fwd_model, epochs_info, in_channels=22, seq_length=1000, 
                 num_classes=2, dropout_rate=0.35, use_pinn=True, sfreq=250):
        super(PINN_CNN_BCI, self).__init__()
        
        self.in_channels = in_channels
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.use_pinn = use_pinn
        self.sfreq = sfreq
        
        device = leadfield.device if isinstance(leadfield, torch.Tensor) else 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 存储前向模型信息
        self.leadfield = leadfield
        self.fwd_model = fwd_model
        self.epochs_info = epochs_info
        
        # CNN时域特征提取分支
        self.cnn_extractor = CNNTemporalExtractor(
            in_channels=in_channels,
            seq_length=seq_length,
            dropout_rate=dropout_rate
        ).to(device)
        
        # PINN空域特征提取分支
        if self.use_pinn:
            n_sources = leadfield.shape[1]
            self.pinn_extractor = PINNSourceLocalization(
                input_dim=in_channels,
                n_sources=n_sources,
                leadfield=leadfield,
                device=device
            ).to(device)
            
            # 特征融合层
            fusion_input_dim = self.cnn_extractor.output_dim + 512  # CNN特征 + PINN深度特征
        else:
            fusion_input_dim = self.cnn_extractor.output_dim
        
        # 平衡的特征融合网络 - 防过拟合优化
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(200, 100),
            nn.BatchNorm1d(100),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(100, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        ).to(device)
        
        # 分类器头
        self.classifier = nn.Linear(64, num_classes).to(device)
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入EEG数据 (batch_size, n_channels, n_timepoints)
        Returns:
            dict: 包含分类结果和中间特征的字典
        """
        batch_size = x.shape[0]
        
        # CNN时域特征提取
        cnn_features = self.cnn_extractor(x)
        
        # PINN空域特征提取
        pinn_outputs = None
        physics_loss = torch.tensor(0.0, device=x.device)
        
        if self.use_pinn:
            # 计算空间平均作为重建目标
            target_eeg = torch.mean(x, dim=2)  # (batch_size, n_channels)
            target_eeg = F.normalize(target_eeg, p=2, dim=1)
            
            # PINN前向传播
            pinn_outputs = self.pinn_extractor(x, target_eeg)
            physics_loss = pinn_outputs['physics_loss']
            pinn_features = pinn_outputs['deep_features']
            
            # 特征融合
            combined_features = torch.cat([cnn_features, pinn_features], dim=1)
        else:
            combined_features = cnn_features
        
        # 特征融合处理
        fused_features = self.feature_fusion(combined_features)
        
        # 分类预测
        logits = self.classifier(fused_features)
        
        # 返回结果
        result = {
            'logits': logits,
            'cnn_features': cnn_features,
            'fused_features': fused_features,
            'physics_loss': physics_loss
        }
        
        if pinn_outputs is not None:
            result.update({
                'source_activations': pinn_outputs['source_activations'],
                'pinn_features': pinn_outputs['deep_features'],
                'spatial_features': pinn_outputs['spatial_features']
            })
        
        return result


# =================================================================================
# 数据加载与预处理（基于V2版本）
# =================================================================================
class BCI2aDataset(Dataset):
    """BCI-IV-2a数据集加载器"""
    def __init__(self, data_dir, subjects, session_type='T', transform=None, repeat_factor=1):
        """
        初始化BCI2a数据集加载器
        
        参数:
        data_dir: 数据集根目录
        subjects: 要加载的受试者编号列表 (1-9)
        session_type: 会话类型，T或E (仅用于兼容，实际未使用)
        transform: 数据转换函数
        repeat_factor: 数据重复系数，用于扩充数据集，默认为1（不扩充）
        """
        self.data = []
        self.labels = []
        self.transform = transform
        self.apply_augmentation = False
        self.repeat_factor = repeat_factor

        for subj in subjects:
            # 直接加载S{subj}.mat文件
            file_path = os.path.join(data_dir, f'S{subj}.mat')
                
            if not os.path.exists(file_path):
                print(f"警告: 文件 {file_path} 不存在，跳过此受试者")
                continue
                
            try:
                mat_data = sio.loadmat(file_path)
                
                # 检查rawdata和label键是否存在
                if 'rawdata' not in mat_data or 'label' not in mat_data:
                    print(f"警告: 文件 {file_path} 中没有rawdata或label键，跳过此文件")
                    continue
                
                # 获取数据和标签
                rawdata = mat_data['rawdata']  # 形状: (1000, 22, 576)
                labels = mat_data['label']     # 形状: (576, 1)
                
                # 转换数据格式: (1000, 22, 576) -> (576, 22, 1000)
                rawdata = np.transpose(rawdata, (2, 1, 0))
                
                # 保持标签为1-4，只进行打平处理
                labels = labels.flatten().astype(np.int64)
                
                # 确保标签是1-4范围内
                if np.min(labels) < 1 or np.max(labels) > 4:
                    labels = np.clip(labels, 1, 4)
                
                # 检查标签是否有足够的类别 
                unique_labels = np.unique(labels)
                if len(unique_labels) < 4:
                    print(f"警告: 标签中没有所有四个类别，只有这些类别: {unique_labels}")
                
                # 标准化每个试验的每个通道
                for i in range(rawdata.shape[0]):  # 遍历所有试验
                    for j in range(rawdata.shape[1]):  # 遍历所有通道
                        # 对每个通道的时间序列进行标准化
                        scaler = StandardScaler()
                        channel_data = rawdata[i, j, :]
                        # 检查数据是否包含NaN或Inf
                        if np.isnan(channel_data).any() or np.isinf(channel_data).any():
                            # 用中值替换无效值
                            channel_data = np.nan_to_num(channel_data, nan=np.nanmedian(channel_data))
                        rawdata[i, j, :] = scaler.fit_transform(channel_data.reshape(-1, 1)).flatten()
                
                # 添加到数据集
                self.data.append(rawdata)
                self.labels.append(labels)
                
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {str(e)}")
                import traceback
                traceback.print_exc()

        if self.data and self.labels:  # 确保有数据被成功加载
            self.data = np.concatenate(self.data, axis=0)
            self.labels = np.concatenate(self.labels, axis=0)
            
            # 验证数据集
            self.validate_dataset()
            
            # 过滤数据，只保留左右手的数据（标签1和2）
            self.filter_left_right_hand_data()
            
            # 如果需要扩充数据集
            if self.repeat_factor > 1:
                self.expand_dataset()
            
            print(f"总共加载了 {len(self.labels)} 个样本，数据形状: {self.data.shape}")
        else:
            raise ValueError("没有成功加载任何数据。请检查数据集路径和格式。")
    
    def filter_left_right_hand_data(self):
        """过滤数据，只保留左右手运动想象的数据（标签1和2）"""
        # 创建掩码，选择标签为1或2的样本
        mask = np.logical_or(self.labels == 1, self.labels == 2)
        
        # 应用掩码，过滤数据和标签
        self.data = self.data[mask]
        self.labels = self.labels[mask]
        
        # 输出过滤后的数据分布
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        print(f"过滤后标签分布: {dict(zip(unique_labels, counts))}")
        print(f"只使用左手(标签1)和右手(标签2)的数据，共{len(self.labels)}个样本")
    
    def set_augmentation(self, enabled=True):
        """设置是否启用数据增强"""
        self.apply_augmentation = enabled
        return self
    
    def validate_dataset(self):
        """验证数据集完整性"""
        if len(self.data) != len(self.labels):
            raise ValueError(f"数据和标签数量不匹配: 数据{len(self.data)}个, 标签{len(self.labels)}个")
            
        # 检查标签分布
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        print(f"标签分布: {dict(zip(unique_labels, counts))}")
        
        # 检查数据中是否有NaN或Inf
        if np.isnan(self.data).any() or np.isinf(self.data).any():
            print("警告: 数据中包含NaN或Inf值，已尝试修复")
            self.data = np.nan_to_num(self.data)
    
    def expand_dataset(self):
        """
        扩充数据集，将每个样本扩充为repeat_factor倍。
        原始样本不增强，扩充的样本都会应用增强。
        """
        original_size = len(self.labels)
        
        # 创建扩展的索引和增强标记数组
        all_indices = []
        all_augmentation_flags = []
        
        for idx in range(original_size):
            # 添加原始样本（不增强）
            all_indices.append(idx)
            all_augmentation_flags.append(False)
            
            # 添加增强样本（应用增强）
            for _ in range(self.repeat_factor - 1):
                all_indices.append(idx)
                all_augmentation_flags.append(True)
        
        # 转换为numpy数组
        self.expanded_indices = np.array(all_indices)
        self.augmentation_flags = np.array(all_augmentation_flags)
        
        # 创建扩充的标签数组
        self.expanded_labels = self.labels.repeat(self.repeat_factor)
        
        print(f"数据集已扩充{self.repeat_factor}倍：原始样本 {original_size} 个，增强样本 {len(self.expanded_indices) - original_size} 个")
        print(f"扩充后总样本数：{len(self.expanded_indices)} 个")
    
    def apply_data_augmentation(self, data):
        """应用增强的数据增强策略 - 防过拟合"""
        # 1. 随机高斯噪声 (增强版)
        if np.random.rand() < 0.8:
            noise_level = np.random.uniform(0.005, 0.02)  # 增加噪声强度
            channel_noise_level = np.random.uniform(0.001, noise_level, (data.shape[0], 1))
            noise = np.random.normal(0, 1, data.shape) * channel_noise_level
            data = data + noise

        # 2. 随机幅度缩放 (增强版)
        if np.random.rand() < 0.7:
            scale_range = (0.8, 1.2)  # 扩大缩放范围
            channel_scales = np.random.uniform(scale_range[0], scale_range[1], (data.shape[0], 1))
            data = data * channel_scales

        # 3. 随机时间偏移 (增强版)
        if np.random.rand() < 0.6:
            max_shift = 25  # 增加最大偏移量
            shift = np.random.randint(-max_shift, max_shift + 1)
            if shift != 0:
                data_shifted = np.zeros_like(data)
                if shift > 0:
                    data_shifted[:, shift:] = data[:, :-shift]
                else:
                    data_shifted[:, :shift] = data[:, -shift:]
                data = data_shifted

        # 4. 新增：随机频域滤波
        if np.random.rand() < 0.4:
            # 随机应用低通或高通滤波
            from scipy import signal
            fs = 250  # 采样频率
            if np.random.rand() < 0.5:
                # 低通滤波
                cutoff = np.random.uniform(30, 50)
                b, a = signal.butter(4, cutoff/(fs/2), 'low')
            else:
                # 高通滤波
                cutoff = np.random.uniform(0.5, 2)
                b, a = signal.butter(4, cutoff/(fs/2), 'high')
            
            for ch in range(data.shape[0]):
                try:
                    data[ch, :] = signal.filtfilt(b, a, data[ch, :])
                except:
                    pass  # 如果滤波失败，跳过

        # 5. 新增：随机通道dropout
        if np.random.rand() < 0.3:
            n_dropout = np.random.randint(1, 4)  # 随机丢弃1-3个通道
            dropout_channels = np.random.choice(data.shape[0], n_dropout, replace=False)
            data[dropout_channels, :] = 0

        # 6. 新增：随机时间窗口masking
        if np.random.rand() < 0.3:
            mask_length = np.random.randint(10, 50)  # 随机mask长度
            start_idx = np.random.randint(0, max(1, data.shape[1] - mask_length))
            data[:, start_idx:start_idx + mask_length] *= 0.1  # 不完全置零，而是大幅衰减

        # 7. 新增：Mixup数据增强 (概率较低，避免过度混合)
        if np.random.rand() < 0.2:
            lambda_mix = np.random.beta(0.2, 0.2)  # Beta分布生成混合系数
            # 这里只做数据混合，标签混合在训练循环中处理
            if hasattr(self, '_mixup_data_cache') and len(self._mixup_data_cache) > 0:
                mix_data = self._mixup_data_cache[np.random.randint(len(self._mixup_data_cache))]
                data = lambda_mix * data + (1 - lambda_mix) * mix_data

        # 8. 新增：EMG伪迹模拟
        if np.random.rand() < 0.15:
            # 模拟肌电伪迹，主要影响边缘通道
            artifact_channels = np.random.choice(range(min(4, data.shape[0])), 
                                                size=np.random.randint(1, 3), replace=False)
            for ch in artifact_channels:
                artifact = np.random.normal(0, 0.3, data.shape[1]) * np.random.uniform(0.5, 2.0)
                data[ch, :] += artifact

        return data.astype(np.float32)
    
    def __len__(self):
        # 如果数据集被扩充，返回扩充后的长度
        if hasattr(self, 'expanded_indices') and self.repeat_factor > 1:
            return len(self.expanded_indices)
        # 否则返回原始长度
        return len(self.labels)

    def __getitem__(self, idx):
        """获取数据集中的一个样本"""
        # 如果数据集被扩充，使用expanded_indices获取实际索引
        if hasattr(self, 'expanded_indices') and self.repeat_factor > 1:
            real_idx = self.expanded_indices[idx]
            should_augment = self.augmentation_flags[idx]
            label = self.labels[real_idx]
        else:
            real_idx = idx
            should_augment = False
            label = self.labels[idx]
            
        # 创建副本以避免修改原始数据
        data = self.data[real_idx].copy().astype(np.float32)
        
        # 如果是扩充样本且启用了增强，应用随机增强方法
        if should_augment and self.apply_augmentation:
            data = self.apply_data_augmentation(data)
            
        # 应用额外的转换（如果有）
        if self.transform:
            data = self.transform(data)
            
        # 最终确保数据是float32类型
        data = data.astype(np.float32)
            
        # 返回原始标签（1或2），在训练和验证循环中统一处理映射
        return data, label


# =================================================================================
# 头模型构建（基于V2版本）
# =================================================================================
def build_head_model(subjects_dir, subject='fsaverage'):
    """
    构建与BCIIV2a.py完全一致的头模型和正向模型
    """
    import mne
    import os
    import numpy as np
    
    # 电极名称
    ch_names = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
    ]
    sfreq = 250.
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)
    raw = mne.io.RawArray(np.zeros((len(ch_names), 1000)), info)
    raw.set_montage(montage)

    # 源空间
    src_fname = os.path.join(subjects_dir, subject, 'bem', f'{subject}-oct-3-src.fif')
    if not os.path.exists(src_fname):
        src = mne.setup_source_space(subject, spacing='oct3', subjects_dir=subjects_dir, add_dist=False, verbose=True)
        mne.write_source_spaces(src_fname, src, overwrite=True)
    else:
        src = mne.read_source_spaces(src_fname, verbose=True)

    # BEM模型
    bem_fname = os.path.join(subjects_dir, subject, 'bem', f'{subject}-5120-5120-5120-bem-sol.fif')
    bem = mne.read_bem_solution(bem_fname, verbose=True)

    # 坐标变换
    trans_fname = os.path.join(subjects_dir, subject, 'bem', f'{subject}-trans.fif')
    trans = mne.read_trans(trans_fname, verbose=True)

    # 前向模型
    fwd = mne.make_forward_solution(
        raw.info, trans=trans, src=src, bem=bem,
        meg=False, eeg=True, mindist=5.0, verbose=True
    )
    fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True, verbose=False)
    print(f"成功创建了适合BCI-IV-2a数据的前向模型，包含 {fwd['sol']['data'].shape[1]} 个源点，{fwd['sol']['data'].shape[0]} 个EEG通道")
    return fwd


# =================================================================================
# 训练与评估函数
# =================================================================================
def train_model(data_dir, subject_id=None, num_epochs=500, batch_size=64, repeat_factor=3, 
                dropout_rate=0.35, learning_rate=0.001, physics_weight=0.3, use_pinn=True):
    """
    训练PINN+CNN混合模型函数
    """
    # 记录开始时间
    start_time = time.time()
    
    # 创建历史记录字典
    history = {
        'train_loss': [],
        'train_class_loss': [],
        'train_physics_loss': [], 
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 构建头模型
    if subject_id == 1:
        print("构建头模型...")
        subjects_dir = os.path.join(mne.get_config('SUBJECTS_DIR') or '', 'fsaverage', '..')
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            default_subjects_dir = os.path.abspath(os.path.join(script_dir, '../../freesurfer/subjects'))
        except NameError:
            default_subjects_dir = os.path.abspath(os.path.join('.', '../../freesurfer/subjects'))

        subjects_dir = default_subjects_dir if not os.path.exists(subjects_dir) else subjects_dir
        if not os.path.exists(os.path.join(subjects_dir, 'fsaverage')):
            print(f"错误: 找不到fsaverage目录于: {subjects_dir}")
            return [], (0,0,0)

        fwd = build_head_model(subjects_dir, subject='fsaverage')
        leadfield = torch.tensor(fwd['sol']['data'], dtype=torch.float32).to(device)
        
        # 创建epochs info
        ch_names = [
            'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
        ]
        sfreq = 250.
        epochs_info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        montage = mne.channels.make_standard_montage('standard_1020')
        epochs_info.set_montage(montage)
    else:
        # 对其他被试，使用静默模式构建头模型
        print(f"为被试 S{subject_id} 构建头模型...")
        subjects_dir = os.path.join(mne.get_config('SUBJECTS_DIR') or '', 'fsaverage', '..')
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            default_subjects_dir = os.path.abspath(os.path.join(script_dir, '../../freesurfer/subjects'))
        except NameError:
            default_subjects_dir = os.path.abspath(os.path.join('.', '../../freesurfer/subjects'))
        subjects_dir = default_subjects_dir if not os.path.exists(subjects_dir) else subjects_dir

        with open(os.devnull, 'w') as f, redirect_stdout(f):
            fwd = build_head_model(subjects_dir, subject='fsaverage')
            leadfield = torch.tensor(fwd['sol']['data'], dtype=torch.float32).to(device)
            
            # 创建epochs info
            ch_names = [
                'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
            ]
            sfreq = 250.
            epochs_info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
            montage = mne.channels.make_standard_montage('standard_1020')
            epochs_info.set_montage(montage)

    # 每个被试使用不同的随机种子，确保结果多样性
    # 注意：这里不重新设置全局种子，使用主函数中设置的种子
    # 让每个被试的随机性基于时间和被试ID
    subject_seed = (int(time.time() * 1000) + subject_id * 1000) % 100000
    # 只在数据加载和增强时使用被试特定的种子
    local_rng = np.random.RandomState(subject_seed)
        
    # 加载数据
    print(f"加载被试 S{subject_id} 的数据...")
    dataset = BCI2aDataset(data_dir, [subject_id], transform=None, repeat_factor=1)
    print(f"被试 S{subject_id} 原始数据集大小: {len(dataset)}个样本")
    
    # 分层划分训练集和测试集
    all_indices = np.arange(len(dataset))
    all_labels = np.array([dataset[i][1] for i in all_indices])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(all_indices, all_labels))
    train_indices = all_indices[train_idx]
    test_indices = all_indices[test_idx]
    
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    print(f"分层划分后：训练集 {len(train_dataset)}，测试集 {len(test_dataset)}")
    
    # 9折交叉验证
    kf = KFold(n_splits=9, shuffle=True, random_state=42)
    all_fold_results = []
    
    for fold, (train_val_idx, test_fold_idx) in enumerate(kf.split(train_indices)):
        print(f"\n--- 第 {fold+1}/9 折交叉验证 ---")
        
        # 获取当前折的训练验证索引
        current_train_indices = train_indices[train_val_idx]
        current_test_indices = train_indices[test_fold_idx]
        
        # 进一步划分训练和验证集
        current_labels = np.array([dataset[i][1] for i in current_train_indices])
        sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42+fold)
        train_idx_inner, val_idx_inner = next(sss_inner.split(current_train_indices, current_labels))
        
        fold_train_indices = current_train_indices[train_idx_inner]
        fold_val_indices = current_train_indices[val_idx_inner]
        
        print(f"折{fold+1}: 训练集 {len(fold_train_indices)}, 验证集 {len(fold_val_indices)}")
        
        # 创建数据增强的训练数据集
        fold_train_dataset = Subset(dataset, fold_train_indices)
        if repeat_factor > 1:
            # 创建增强版本的数据集
            augmented_dataset = BCI2aDataset(data_dir, [subject_id], transform=None, repeat_factor=repeat_factor)
            augmented_dataset.set_augmentation(True)
            # 只对训练集应用增强
            augmented_train_indices = []
            for idx in fold_train_indices:
                # 添加原始样本
                augmented_train_indices.append(idx)
                # 添加增强样本（通过重复索引实现）
                for _ in range(repeat_factor - 1):
                    augmented_train_indices.append(idx)
            fold_train_dataset = Subset(augmented_dataset, augmented_train_indices)
        
        fold_val_dataset = Subset(dataset, fold_val_indices)
        
        # 创建数据加载器
        train_loader = DataLoader(fold_train_dataset, batch_size=batch_size, shuffle=True, 
                                num_workers=0, pin_memory=True)
        val_loader = DataLoader(fold_val_dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=0, pin_memory=True)
        
        # 初始化模型
        model = PINN_CNN_BCI(
            leadfield=leadfield,
            fwd_model=fwd,
            epochs_info=epochs_info,
            in_channels=22,
            seq_length=1000,
            num_classes=2,
            dropout_rate=dropout_rate,
            use_pinn=use_pinn,
            sfreq=250
        ).to(device)
        
        # 损失函数和优化器 - 平衡优化
        # 使用标签平滑的交叉熵损失
        criterion = nn.CrossEntropyLoss(label_smoothing=0.15)  # 增加标签平滑强度
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)  # 增加权重衰减
        # 使用更保守的学习率调度
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=8, verbose=False)
        
        # 训练循环 - 更严格的早停
        best_val_acc = 0.0
        patience_counter = 0
        max_patience = 10  # 更严格的早停
        
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_class_loss = 0.0
            train_physics_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(device)
                # 将标签从1,2映射到0,1
                batch_labels = (batch_labels - 1).long().to(device)
                
                optimizer.zero_grad()
                
                # 前向传播
                outputs = model(batch_data)
                
                # 分类损失
                class_loss = criterion(outputs['logits'], batch_labels)
                
                # PINN物理损失（如果使用PINN）
                if use_pinn:
                    physics_loss = outputs['physics_loss']
                    # 总损失 = 分类损失 + 加权的PINN损失
                    total_loss = class_loss + physics_weight * physics_loss
                else:
                    physics_loss = torch.tensor(0.0, device=device)
                    total_loss = class_loss
                
                # 反向传播
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # 统计
                train_loss += total_loss.item()
                train_class_loss += class_loss.item()
                train_physics_loss += physics_loss.item()
                
                _, predicted = torch.max(outputs['logits'].data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_predictions = []
            val_true_labels = []
            
            with torch.no_grad():
                for batch_data, batch_labels in val_loader:
                    batch_data = batch_data.to(device)
                    batch_labels = (batch_labels - 1).long().to(device)
                    
                    outputs = model(batch_data)
                    class_loss = criterion(outputs['logits'], batch_labels)
                    
                    if use_pinn:
                        physics_loss = outputs['physics_loss']
                        total_loss = class_loss + physics_weight * physics_loss
                    else:
                        physics_loss = torch.tensor(0.0, device=device)
                        total_loss = class_loss
                    
                    val_loss += total_loss.item()
                    _, predicted = torch.max(outputs['logits'].data, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels).sum().item()
                    
                    val_predictions.extend(predicted.cpu().numpy())
                    val_true_labels.extend(batch_labels.cpu().numpy())
            
            # 计算准确率
            train_acc = 100.0 * train_correct / train_total
            val_acc = 100.0 * val_correct / val_total
            
            # 计算F1分数
            val_f1 = f1_score(val_true_labels, val_predictions, average='weighted')
            
            # 学习率调度 - 基于验证准确率
            scheduler.step(val_acc)
            
            # 早停检查 - 增强版
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 检查过拟合：训练准确率比验证准确率高太多
            if train_acc - val_acc > 12.0:  # 更严格的过拟合检测阈值
                patience_counter += 1
                if epoch > 15:  # 更早开始检查
                    print(f"检测到过拟合迹象: 训练准确率{train_acc:.2f}% vs 验证准确率{val_acc:.2f}%")
                
            if patience_counter >= max_patience:
                print(f"早停于第 {epoch+1} 轮，最佳验证准确率: {best_val_acc:.2f}%")
                break
            
            # 每20轮打印一次进度
            if (epoch + 1) % 20 == 0:
                if use_pinn:
                    avg_physics_loss = train_physics_loss / len(train_loader)
                    print(f"轮次 {epoch+1}/{num_epochs}: 训练准确率 {train_acc:.2f}%, 验证准确率 {val_acc:.2f}%, PINN损失 {avg_physics_loss:.6f}")
                else:
                    print(f"轮次 {epoch+1}/{num_epochs}: 训练准确率 {train_acc:.2f}%, 验证准确率 {val_acc:.2f}%")
        
        # 计算训练集的真实F1分数
        final_train_predictions = []
        final_train_labels = []
        
        model.eval()
        with torch.no_grad():
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(device)
                batch_labels = (batch_labels - 1).long().to(device)
                
                outputs = model(batch_data)
                _, predicted = torch.max(outputs['logits'].data, 1)
                
                final_train_predictions.extend(predicted.cpu().numpy())
                final_train_labels.extend(batch_labels.cpu().numpy())
        
        train_f1 = f1_score(final_train_labels, final_train_predictions, average='weighted')
        train_kappa = cohen_kappa_score(final_train_labels, final_train_predictions)
        val_kappa = cohen_kappa_score(val_true_labels, val_predictions)
        
        # 记录完整结果 (fold, train_acc, train_f1, train_kappa, val_acc, val_f1, val_kappa)
        all_fold_results.append((fold+1, train_acc, train_f1, train_kappa, best_val_acc, val_f1, val_kappa))
        
        # 清理内存
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        gc.collect()
    
    # 最终在测试集上评估（使用最后一折的模型架构重新训练）
    print("\n=== 最终测试集评估 ===")
    
    # 使用完整训练集重新训练模型
    if repeat_factor > 1:
        augmented_full_dataset = BCI2aDataset(data_dir, [subject_id], transform=None, repeat_factor=repeat_factor)
        augmented_full_dataset.set_augmentation(True)
        augmented_train_indices = []
        for idx in train_indices:
            augmented_train_indices.append(idx)
            for _ in range(repeat_factor - 1):
                augmented_train_indices.append(idx)
        final_train_dataset = Subset(augmented_full_dataset, augmented_train_indices)
    else:
        final_train_dataset = Subset(dataset, train_indices)
    
    final_test_dataset = Subset(dataset, test_indices)
    
    final_train_loader = DataLoader(final_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    final_test_loader = DataLoader(final_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # 重新初始化模型
    final_model = PINN_CNN_BCI(
        leadfield=leadfield,
        fwd_model=fwd,
        epochs_info=epochs_info,
        in_channels=22,
        seq_length=1000,
        num_classes=2,
        dropout_rate=dropout_rate,
        use_pinn=use_pinn,
        sfreq=250
    ).to(device)
    
    final_optimizer = optim.Adam(final_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 快速训练最终模型（使用较少轮次）
    final_model.train()
    for epoch in range(min(50, num_epochs)):
        for batch_data, batch_labels in final_train_loader:
            batch_data = batch_data.to(device)
            batch_labels = (batch_labels - 1).long().to(device)
            
            final_optimizer.zero_grad()
            outputs = final_model(batch_data)
            class_loss = criterion(outputs['logits'], batch_labels)
            
            if use_pinn:
                physics_loss = outputs['physics_loss']
                total_loss = class_loss + physics_weight * physics_loss
            else:
                physics_loss = torch.tensor(0.0, device=device)
                total_loss = class_loss
                
            total_loss.backward()
            final_optimizer.step()
    
    # 最终测试
    final_model.eval()
    test_predictions = []
    test_true_labels = []
    
    with torch.no_grad():
        for batch_data, batch_labels in final_test_loader:
            batch_data = batch_data.to(device)
            batch_labels = (batch_labels - 1).long().to(device)
            
            outputs = final_model(batch_data)
            _, predicted = torch.max(outputs['logits'].data, 1)
            
            test_predictions.extend(predicted.cpu().numpy())
            test_true_labels.extend(batch_labels.cpu().numpy())
    
    # 计算最终指标
    test_acc = 100.0 * accuracy_score(test_true_labels, test_predictions)
    test_f1 = f1_score(test_true_labels, test_predictions, average='weighted')
    test_kappa = cohen_kappa_score(test_true_labels, test_predictions)
    
    print(f"最终测试结果 - 准确率: {test_acc:.2f}%, F1: {test_f1:.4f}, Kappa: {test_kappa:.4f}")
    
    # 清理内存
    del final_model, final_optimizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return all_fold_results, (test_acc, test_f1, test_kappa)


# =================================================================================
# 主函数
# =================================================================================
if __name__ == '__main__':
    # ===================== 参数控制变量（方便修改） =====================
    # 训练参数 - 平衡准确率和过拟合控制
    BATCH_SIZE = 24          # 适中的批次大小，平衡训练稳定性和随机性
    NUM_EPOCHS = 100         # 适中的训练轮数，配合严格早停
    LEARNING_RATE = 0.0005   # 适中的学习率
    DROPOUT_RATE = 0.5       # 增强Dropout，防止过拟合
    PHYSICS_WEIGHT = 0.4     # 增加物理损失权重，增强正则化
    
    # 数据增强参数 - 平衡增强强度
    REPEAT_FACTOR = 6        # 适中的数据增强倍数
    
    # 模型参数
    USE_PINN = True  # 是否使用PINN空域特征提取
    
    # 随机性控制
    USE_FIXED_SEED = False  # True: 使用固定种子(42)确保结果可复现, False: 使用随机种子
    FIXED_SEED = 42        # 固定种子值
    
    # 数据路径
    DATA_DIR = r"E:\pycharm\PINN\PINN\data\BCI2a"  # 请根据实际情况修改路径
    
    # ===================== 程序开始执行 =====================
    print("=== 使用 PINN(空域) + CNN(时域) 混合模型进行BCI-IV-2a左右手分类 ===")
    print(f"训练参数配置:")
    print(f"  - 批次大小: {BATCH_SIZE}")
    print(f"  - 训练轮数: {NUM_EPOCHS}")
    print(f"  - 学习率: {LEARNING_RATE}")
    print(f"  - Dropout率: {DROPOUT_RATE}")
    print(f"  - 物理损失权重: {PHYSICS_WEIGHT}")
    print(f"  - 数据增强倍数: {REPEAT_FACTOR}倍 ({'不使用增强' if REPEAT_FACTOR == 1 else f'包含{REPEAT_FACTOR-1}倍增强数据'})")
    print(f"  - PINN模式: {'启用' if USE_PINN else '禁用'}")
    print(f"  - 数据路径: {DATA_DIR}")
    
    # 记录全局开始时间
    global_start_time = time.time()
    
    # 设置随机种子
    import random
    if USE_FIXED_SEED:
        current_seed = FIXED_SEED
        print(f"  - 随机种子: {current_seed} (固定种子，结果可复现)")
    else:
        current_seed = int(time.time()) % 10000  # 使用当前时间戳作为种子
        print(f"  - 随机种子: {current_seed} (随机种子，每次运行不同)")
    
    torch.manual_seed(current_seed)
    np.random.seed(current_seed)
    random.seed(current_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(current_seed)
        torch.cuda.manual_seed_all(current_seed)
        
    # 强制垃圾回收
    gc.collect()
    torch.cuda.empty_cache()
    
    # 检查数据目录结构
    if os.path.exists(DATA_DIR):
        # 检查是否存在所有受试者数据
        available_subjects = []
        for i in range(1, 10):
            file_path = os.path.join(DATA_DIR, f'S{i}.mat')
            if os.path.exists(file_path):
                available_subjects.append(i)
        
        print(f"\n检测到 {len(available_subjects)} 个可用受试者数据: S{available_subjects}")
        
        # 存储每个被试的结果
        subject_results = []
        testset_results = []
        
        for subject_id in available_subjects:
            print(f"\n=== 开始训练被试 S{subject_id} ===\n")
            results, testset_result = train_model(
                DATA_DIR, 
                subject_id=subject_id, 
                num_epochs=NUM_EPOCHS, 
                batch_size=BATCH_SIZE, 
                repeat_factor=REPEAT_FACTOR,
                dropout_rate=DROPOUT_RATE,
                learning_rate=LEARNING_RATE,
                physics_weight=PHYSICS_WEIGHT,
                use_pinn=USE_PINN
            )
            if not results: 
                continue

            arr = np.array(results)
            train_avg_acc = np.mean(arr[:,1])
            train_avg_f1 = np.mean(arr[:,2])
            train_avg_kappa = np.mean(arr[:,3])
            val_avg_acc = np.mean(arr[:,4])
            val_avg_f1 = np.mean(arr[:,5])
            val_avg_kappa = np.mean(arr[:,6])
            
            subject_results.append((subject_id, train_avg_acc, train_avg_f1, train_avg_kappa, 
                                  val_avg_acc, val_avg_f1, val_avg_kappa))
            testset_results.append((subject_id, *testset_result))
            
            # 打印当前被试的详细结果
            print(f"\n--- 被试 S{subject_id} 结果汇总 ---")
            print(f"9折交叉验证平均结果:")
            print(f"  训练集 - 准确率: {train_avg_acc:.2f}%, F1: {train_avg_f1:.4f}, Kappa: {train_avg_kappa:.4f}")
            print(f"  验证集 - 准确率: {val_avg_acc:.2f}%, F1: {val_avg_f1:.4f}, Kappa: {val_avg_kappa:.4f}")
            print(f"最终测试集结果:")
            print(f"  测试集 - 准确率: {testset_result[0]:.2f}%, F1: {testset_result[1]:.4f}, Kappa: {testset_result[2]:.4f}")
        
        # 打印所有被试的结果汇总
        print("\n\n" + "="*120)
        print("===== 所有被试结果汇总 (PINN + CNN 混合模型) =====")
        print("="*120)
        
        if not subject_results:
             print("没有有效的训练结果。")
        else:
            # 计算所有平均值
            avg_train_acc = np.mean([r[1] for r in subject_results])
            avg_train_f1 = np.mean([r[2] for r in subject_results])
            avg_train_kappa = np.mean([r[3] for r in subject_results])
            avg_val_acc = np.mean([r[4] for r in subject_results])
            avg_val_f1 = np.mean([r[5] for r in subject_results])
            avg_val_kappa = np.mean([r[6] for r in subject_results])
            avg_test_acc = np.mean([r[1] for r in testset_results])
            avg_test_f1 = np.mean([r[2] for r in testset_results])
            avg_test_kappa = np.mean([r[3] for r in testset_results])
            
            # 详细表格标题
            print("被试\t训练集结果\t\t\t验证集结果\t\t\t测试集结果")
            print("\t准确率\tF1\tKappa\t准确率\tF1\tKappa\t准确率\tF1\tKappa")
            print("-" * 120)
            
            # 打印每个被试的结果
            for i, subj_result in enumerate(subject_results):
                subj, train_acc, train_f1, train_kappa, val_acc, val_f1, val_kappa = subj_result
                testset_acc, testset_f1, testset_kappa = testset_results[i][1:]
                print(f"S{subj}\t{train_acc:.2f}%\t{train_f1:.4f}\t{train_kappa:.4f}\t"
                      f"{val_acc:.2f}%\t{val_f1:.4f}\t{val_kappa:.4f}\t"
                      f"{testset_acc:.2f}%\t{testset_f1:.4f}\t{testset_kappa:.4f}")
            
            print("-" * 120)
            print(f"平均\t{avg_train_acc:.2f}%\t{avg_train_f1:.4f}\t{avg_train_kappa:.4f}\t"
                  f"{avg_val_acc:.2f}%\t{avg_val_f1:.4f}\t{avg_val_kappa:.4f}\t"
                  f"{avg_test_acc:.2f}%\t{avg_test_f1:.4f}\t{avg_test_kappa:.4f}")
            print("=" * 120)
            
            
            # 保存详细结果到CSV文件
            try:
                import pandas as pd
                
                # 创建详细结果DataFrame
                detailed_results = []
                for i, subj_result in enumerate(subject_results):
                    subj, train_acc, train_f1, train_kappa, val_acc, val_f1, val_kappa = subj_result
                    testset_acc, testset_f1, testset_kappa = testset_results[i][1:]
                    
                    detailed_results.append({
                        '被试': f'S{subj}',
                        '训练准确率(%)': round(train_acc, 2),
                        '训练F1': round(train_f1, 4),
                        '训练Kappa': round(train_kappa, 4),
                        '验证准确率(%)': round(val_acc, 2),
                        '验证F1': round(val_f1, 4),
                        '验证Kappa': round(val_kappa, 4),
                        '测试准确率(%)': round(testset_acc, 2),
                        '测试F1': round(testset_f1, 4),
                        '测试Kappa': round(testset_kappa, 4)
                    })
                
                # 添加平均值行
                detailed_results.append({
                    '被试': '平均',
                    '训练准确率(%)': round(avg_train_acc, 2),
                    '训练F1': round(avg_train_f1, 4),
                    '训练Kappa': round(avg_train_kappa, 4),
                    '验证准确率(%)': round(avg_val_acc, 2),
                    '验证F1': round(avg_val_f1, 4),
                    '验证Kappa': round(avg_val_kappa, 4),
                    '测试准确率(%)': round(avg_test_acc, 2),
                    '测试F1': round(avg_test_f1, 4),
                    '测试Kappa': round(avg_test_kappa, 4)
                })
                
                df = pd.DataFrame(detailed_results)
                
                # 生成文件名（包含时间戳和参数信息）
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"BCI_V3_results_{timestamp}_E{NUM_EPOCHS}_B{BATCH_SIZE}_R{REPEAT_FACTOR}.csv"
                
                df.to_csv(filename, index=False, encoding='utf-8-sig')
                print(f"\n💾 详细结果已保存到: {filename}")
                
            except ImportError:
                print("\n⚠️  pandas未安装，跳过CSV文件保存")
            except Exception as e:
                print(f"\n⚠️  保存CSV文件时出错: {e}")
            
            print("=" * 120)
        
        # 计算全局总训练时间
        global_total_time = time.time() - global_start_time
        hours, remainder = divmod(global_total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        global_time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        print(f"\n===== 训练完成! 所有被试总训练时间: {global_time_str} =====")
            
    else:
        print(f"错误: BCI2a数据目录不存在: {DATA_DIR}")
        print("请先下载BCI2a数据集并放置到正确位置，或修改DATA_DIR变量")
        exit(1)