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
from sklearn.model_selection import KFold, StratifiedShuffleSplit, train_test_split
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

# 抑制joblib并行处理的详细输出
import joblib
import logging
logging.getLogger('joblib').setLevel(logging.ERROR)

# 设置MNE日志级别，减少输出
mne.set_log_level('ERROR')

# 设置matplotlib字体为微软黑体，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# =================================================================================
# PINN源定位网络模块
# =================================================================================
class PINNSourceLocalization(nn.Module):
    """
    PINN源定位模块 - 完全按照esinet/net.py中的实现
    使用TimeDistributed Dense层结构：22→64→128→256→512→n_sources
    """
    def __init__(self, input_dim, n_sources, leadfield, device='cuda', activation='relu', dropout_rate=0.25):
        super(PINNSourceLocalization, self).__init__()
        self.input_dim = input_dim
        self.n_sources = n_sources
        self.device = device
        self.dropout_rate = dropout_rate
        
        # 引线场矩阵 (n_channels, n_sources) - 注册为buffer
        self.register_buffer('leadfield', leadfield.to(device))
        
        # 完全按照esinet/net.py的TimeDistributed Dense层结构
        # Input: (batch, time, 22)
        self.pinn_conv1 = nn.Linear(input_dim, 64).to(device)    # TimeDistributed(Dense(64))
        self.pinn_conv2 = nn.Linear(64, 128).to(device)          # TimeDistributed(Dense(128)) 
        self.pinn_conv3 = nn.Linear(128, 256).to(device)         # TimeDistributed(Dense(256))
        self.pinn_deep_features = nn.Linear(256, 512).to(device) # TimeDistributed(Dense(512))
        self.pinn_source_fc = nn.Linear(512, n_sources).to(device) # TimeDistributed(Dense(n_sources))
        
        # 激活函数和Dropout
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
            
        self.dropout = nn.Dropout(dropout_rate)
        
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
        PINN前向传播 - 完全按照esinet/net.py中的TimeDistributed结构
        Args:
            x: 输入EEG数据 (batch_size, n_channels, n_timepoints)
            target_eeg: 目标EEG信号用于leadfield投影损失 (可选)
        Returns:
            dict: 包含main_output(源激活)和physics_output(leadfield投影)
        """
        batch_size, n_channels, n_timepoints = x.shape
        
        # 转换为时间优先格式 (batch_size, n_timepoints, n_channels)
        # 完全匹配esinet Input shape: (batch, time, channels)
        inputs = x.transpose(1, 2)  # (batch_size, time, channels)
        
        # 完全按照esinet/net.py的TimeDistributed Dense层序列
        # TimeDistributed(Dense(64)) + Dropout
        conv1 = self.activation(self.pinn_conv1(inputs))
        conv1 = self.dropout(conv1)
        
        # TimeDistributed(Dense(128)) + Dropout  
        conv2 = self.activation(self.pinn_conv2(conv1))
        conv2 = self.dropout(conv2)
        
        # TimeDistributed(Dense(256)) + Dropout
        conv3 = self.activation(self.pinn_conv3(conv2))
        conv3 = self.dropout(conv3)
        
        # TimeDistributed(Dense(512)) + Dropout [deep_features]
        deep_features = self.activation(self.pinn_deep_features(conv3))
        deep_features = self.dropout(deep_features)
        
        # TimeDistributed(Dense(n_sources)) [source_fc - main_output]
        # 使用Tanh激活限制源激活在[-1,1]范围，提高训练稳定性
        source_fc = torch.tanh(self.pinn_source_fc(deep_features))  # (batch, time, n_sources)
        
        # Leadfield投影 - physics_output分支 
        # 对每个时间步应用leadfield投影: source → electrode space
        physics_output = None
        if hasattr(self, 'leadfield') and self.leadfield is not None:
            # source_fc: (batch, time, n_sources) × leadfield.T: (n_sources, n_channels)
            # → physics_output: (batch, time, n_channels)
            physics_output = torch.matmul(source_fc, self.leadfield.t())
        
        # 按照论文公式计算完整的PINN物理损失
        physics_loss = torch.tensor(0.0, device=x.device)
        if target_eeg is not None and physics_output is not None:
            # 时间平均用于损失计算
            target_mean = torch.mean(target_eeg, dim=2) if target_eeg.dim() == 3 else target_eeg
            physics_mean = torch.mean(physics_output, dim=1)
            source_mean = torch.mean(source_fc, dim=1)
            
            # 1. 数据保真度损失 (公式5) - 余弦相似度
            dot_product = torch.sum(physics_mean * target_mean, dim=1)
            norm_pred = torch.norm(physics_mean, p=2, dim=1)
            norm_true = torch.norm(target_mean, p=2, dim=1)
            norm_pred = torch.clamp(norm_pred, min=1e-8)
            norm_true = torch.clamp(norm_true, min=1e-8)
            cosine_sim = dot_product / (norm_pred * norm_true)
            L_data = torch.mean(1.0 - cosine_sim)
            
            # 2. 物理约束损失 (公式6) - 前向模型一致性
            L_phys = torch.mean((physics_mean - target_mean) ** 2)
            
            # 3. 辅助正则化损失 (公式7) - 源激活稀疏性
            L_reg = 0.01 * torch.mean(source_mean ** 2) + 0.01 * torch.mean(torch.abs(source_mean))
            
            # 总物理损失 (公式4)
            lambda_data, lambda_phys, lambda_reg = 1.0, 0.1, 0.01
            physics_loss = lambda_data * L_data + lambda_phys * L_phys + lambda_reg * L_reg
        
        return {
            'main_output': source_fc,                           # (batch, time, n_sources) - 完全匹配esinet
            'physics_output': physics_output,                   # (batch, time, n_channels) - leadfield投影
            'deep_features': torch.mean(deep_features, dim=1),  # (batch, 512) - 时间平均，兼容现有接口
            'source_activations': torch.mean(source_fc, dim=1), # (batch, n_sources) - 时间平均，兼容现有接口
            'physics_loss': physics_loss,
            'spatial_features': torch.mean(x, dim=2)            # (batch, n_channels) - 兼容现有接口
        }


# =================================================================================
# CNN时域特征提取模块
# =================================================================================
class SimplifiedTemporalCNN(nn.Module):
    """
    简化的时域CNN - 减少过拟合，提高泛化能力
    基于经典EEGNet设计原理，但专注时域特征
    """
    def __init__(self, in_channels=22, seq_length=1000, dropout_rate=0.5):
        super(SimplifiedTemporalCNN, self).__init__()
        
        # Stage 1: 时域卷积 - 简单但有效
        self.temporal_block = nn.Sequential(
            # 使用更大的kernel size捕获低频特征
            nn.Conv1d(in_channels, 32, kernel_size=64, padding=32),  # 256ms窗口 @250Hz
            nn.BatchNorm1d(32),
            nn.ELU(inplace=True),
            nn.AvgPool1d(4),  # -> (batch, 32, 250)
            nn.Dropout(dropout_rate * 0.5),
            
            # 第二层：更细粒度的时域特征
            nn.Conv1d(32, 32, kernel_size=16, padding=8),  # 64ms窗口
            nn.BatchNorm1d(32),
            nn.ELU(inplace=True),
            nn.AvgPool1d(4),  # -> (batch, 32, 62)
            nn.Dropout(dropout_rate * 0.7),
        )
        
        # Stage 2: 特征整合
        self.feature_integration = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=8, padding=4),
            nn.BatchNorm1d(16),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # 全局池化 -> (batch, 16, 1)
            nn.Dropout(dropout_rate),
        )
        
        self.output_dim = 16
        
        # 极简CNN分类器 - 二分类
        self.cnn_classifier = nn.Sequential(
            nn.Linear(self.output_dim, 8),
            nn.ELU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(8, 2)  # 二分类输出
        )
        
    def forward(self, x):
        """
        简化的时域特征提取
        """
        # 时域特征提取
        temporal_features = self.temporal_block(x)  # -> (batch, 32, 62)
        
        # 特征整合
        features = self.feature_integration(temporal_features)  # -> (batch, 16, 1)
        features = features.squeeze(-1)  # -> (batch, 16)
        
        # CNN分支预测
        cnn_logits = self.cnn_classifier(features)
        
        return {
            'features': features,
            'logits': cnn_logits
        }


class CNNTemporalExtractor(nn.Module):
    """
    兼容性封装，使用简化的时域CNN
    """
    def __init__(self, in_channels=22, seq_length=1000, dropout_rate=0.5):
        super(CNNTemporalExtractor, self).__init__()
        self.extractor = SimplifiedTemporalCNN(in_channels, seq_length, dropout_rate)
        self.output_dim = self.extractor.output_dim
        
    def forward(self, x):
        return self.extractor(x)


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
        
        # 简化的特征融合策略 - 减少参数，增强泛化
        if self.use_pinn:
            # 使用注意力机制融合时空特征
            self.cnn_projection = nn.Linear(self.cnn_extractor.output_dim, 16).to(device)  # 16→16
            self.pinn_projection = nn.Linear(512, 16).to(device)  # 512→16，大幅压缩
            
            # 简单的注意力融合
            self.attention_fusion = nn.Sequential(
                nn.Linear(32, 16),  # CNN+PINN特征
                nn.Tanh(),
                nn.Linear(16, 2),   # 生成注意力权重
                nn.Softmax(dim=1)
            ).to(device)
            
            # 最终特征维度
            classifier_input_dim = 16
        else:
            # 仅CNN分支：简单处理
            self.cnn_projection = nn.Identity()
            classifier_input_dim = self.cnn_extractor.output_dim  # 16
        
        # 超简化分类器 - 防止过拟合，二分类
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_input_dim, 2)  # 二分类输出（原始logits，配合CrossEntropyLoss）
        ).to(device)
        
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
        cnn_outputs = self.cnn_extractor(x)
        cnn_features = cnn_outputs['features']
        cnn_logits = cnn_outputs['logits']
        
        # PINN空域特征提取
        pinn_outputs = None
        physics_loss = torch.tensor(0.0, device=x.device)
        
        if self.use_pinn:
            # PINN前向传播 - 传入原始EEG作为重建目标
            # 物理损失 = ||leadfield × source_activations - original_eeg||²
            pinn_outputs = self.pinn_extractor(x, target_eeg=x)
            physics_loss = pinn_outputs['physics_loss']
            pinn_features = pinn_outputs['deep_features']
            
            # 特征投影到相同维度
            cnn_projected = self.cnn_projection(cnn_features)      # 16→16
            pinn_projected = self.pinn_projection(pinn_features)   # 512→16
            
            # 注意力机制融合
            combined_features = torch.cat([cnn_projected, pinn_projected], dim=1)  # 16+16=32
            attention_weights = self.attention_fusion(combined_features)  # -> (batch, 2)
            
            # 加权融合
            weighted_cnn = cnn_projected * attention_weights[:, 0:1]
            weighted_pinn = pinn_projected * attention_weights[:, 1:2]
            fused_features = weighted_cnn + weighted_pinn  # -> (batch, 16)
        else:
            # 仅CNN分支
            fused_features = self.cnn_projection(cnn_features)  # 16→16
        
        # 分类预测
        logits = self.classifier(fused_features)
        
        # 返回结果
        result = {
            'logits': logits,              # 主分类器输出
            'cnn_logits': cnn_logits,      # CNN分支独立输出
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
                
                # 确保标签是1-4范围内（原始数据可能包含四个类别，但我们只使用1和2）
                if np.min(labels) < 1 or np.max(labels) > 4:
                    labels = np.clip(labels, 1, 4)
                
                # 检查标签分布 
                unique_labels = np.unique(labels)
                print(f"原始数据中的标签类别: {unique_labels}（将只使用1=左手和2=右手）")
                
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
                dropout_rate=0.35, learning_rate=0.001, physics_weight=0.3, cnn_aux_weight=0.3, use_pinn=True):
    """
    训练PINN+CNN混合模型函数 - 正确的交叉验证实验设计
    """
    # 记录开始时间
    start_time = time.time()
    
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
            return [], (0,0,0), (0,0,0)

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
        
    # 加载数据
    print(f"加载被试 S{subject_id} 的数据...")
    dataset = BCI2aDataset(data_dir, [subject_id], transform=None, repeat_factor=1)
    print(f"被试 S{subject_id} 原始数据集大小: {len(dataset)}个样本")
    
    # =================== 阶段一：数据划分（训练集80% vs 测试集20%）===================
    all_indices = np.arange(len(dataset))
    all_labels = np.array([dataset[i][1] for i in all_indices])
    
    # 使用全局统一的随机种子（由主函数传入）
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=None)
    train_idx, test_idx = next(sss.split(all_indices, all_labels))
    train_indices = all_indices[train_idx]
    test_indices = all_indices[test_idx]
    
    print(f"数据划分：训练集 {len(train_indices)} 样本，测试集 {len(test_indices)} 样本")
    
    # =================== 阶段二：9折交叉验证（仅在训练集上进行）===================
    print("\n=== 开始9折交叉验证（模型选择阶段）===")
    kf = KFold(n_splits=9, shuffle=True, random_state=None)
    cv_results = []
    
    for fold, (train_fold_idx, val_fold_idx) in enumerate(kf.split(train_indices)):
        print(f"\n--- 第 {fold+1}/9 折交叉验证 ---")
        
        # 当前折的训练集（8/9的训练数据）和验证集（1/9的训练数据）
        fold_train_indices = train_indices[train_fold_idx]  # 8折用于训练
        fold_val_indices = train_indices[val_fold_idx]      # 1折用于验证
        
        print(f"折{fold+1}: 训练集 {len(fold_train_indices)}, 验证集 {len(fold_val_indices)}")
        
        # 创建数据增强的训练数据集（只对训练集增强）
        if repeat_factor > 1:
            augmented_dataset = BCI2aDataset(data_dir, [subject_id], transform=None, repeat_factor=repeat_factor)
            augmented_dataset.set_augmentation(True)
            # 正确的索引映射：原始索引 -> 扩充后索引
            augmented_train_indices = []
            for idx in fold_train_indices:
                # 对于每个原始索引，找到它在扩充数据集中对应的所有位置
                for rep in range(repeat_factor):
                    augmented_idx = idx * repeat_factor + rep
                    augmented_train_indices.append(augmented_idx)
            fold_train_dataset = Subset(augmented_dataset, augmented_train_indices)
        else:
            fold_train_dataset = Subset(dataset, fold_train_indices)
        
        fold_val_dataset = Subset(dataset, fold_val_indices)
        
        # 创建数据加载器
        train_loader = DataLoader(fold_train_dataset, batch_size=batch_size, shuffle=True, 
                                num_workers=0, pin_memory=True)
        val_loader = DataLoader(fold_val_dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=0, pin_memory=True)
        
        # 初始化模型 - 二分类
        model = PINN_CNN_BCI(
            leadfield=leadfield,
            fwd_model=fwd,
            epochs_info=epochs_info,
            in_channels=22,
            seq_length=1000,
            num_classes=2,  # 二分类：左手、右手
            dropout_rate=dropout_rate,
            use_pinn=use_pinn,
            sfreq=250
        ).to(device)
        
        # 损失函数和优化器 - 匹配简化架构
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # 降低label smoothing
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=0.005,  # 适度的权重衰减
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        # 更快的学习率调度
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.7, patience=8, verbose=False
        )
        
        # 训练循环
        best_val_acc = 0.0
        patience_counter = 0
        max_patience = 20  # 增加patience以配合更多epoch
        best_model_state = None  # 保存最佳模型状态
        
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_main_loss = 0.0
            train_cnn_aux_loss = 0.0
            train_physics_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.long().to(device)  # 二分类：左手=1，右手=2 -> 转为0,1
                batch_labels = batch_labels - 1  # 二分类：1,2 -> 0,1
                
                optimizer.zero_grad()
                outputs = model(batch_data)
                
                # 主分类损失（融合后的预测）
                main_class_loss = criterion(outputs['logits'], batch_labels)
                
                # CNN分支辅助损失
                cnn_aux_loss = criterion(outputs['cnn_logits'], batch_labels)
                
                # PINN物理损失（如果使用PINN）
                if use_pinn:
                    physics_loss = outputs['physics_loss']
                    # 多目标损失函数：主分类 + CNN辅助 + PINN物理
                    total_loss = (1.0 * main_class_loss +      # 主损失权重: 1.0
                                 cnn_aux_weight * cnn_aux_loss +         # CNN辅助权重: 可调
                                 physics_weight * physics_loss) # PINN物理权重: 可调
                else:
                    physics_loss = torch.tensor(0.0, device=device)
                    # 双分支损失：主分类 + CNN辅助
                    total_loss = (1.0 * main_class_loss +      # 主损失权重: 1.0
                                 cnn_aux_weight * cnn_aux_loss)          # CNN辅助权重: 可调
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # 统计各种损失
                train_loss += total_loss.item()
                train_main_loss += main_class_loss.item()
                train_cnn_aux_loss += cnn_aux_loss.item()
                train_physics_loss += physics_loss.item()
                
                _, predicted = torch.max(outputs['logits'].data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
            
            # 验证阶段
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_data, batch_labels in val_loader:
                    batch_data = batch_data.to(device)
                    batch_labels = batch_labels.long().to(device)  # 二分类：左手=1，右手=2 -> 转为0,1
                    batch_labels = batch_labels - 1  # 二分类：1,2 -> 0,1
                    
                    outputs = model(batch_data)
                    _, predicted = torch.max(outputs['logits'].data, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels).sum().item()
            
            # 计算准确率
            train_acc = 100.0 * train_correct / train_total
            val_acc = 100.0 * val_correct / val_total
            
            # 学习率调度
            scheduler.step(val_acc)
            
            # 早停检查
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # 保存最佳模型状态
                best_model_state = {
                    'model_state_dict': model.state_dict().copy(),
                    'epoch': epoch + 1,
                    'best_val_acc': best_val_acc
                }
            else:
                patience_counter += 1
                
            if patience_counter >= max_patience:
                print(f"早停于第 {epoch+1} 轮，最佳验证准确率: {best_val_acc:.2f}%")
                # 恢复最佳模型状态
                if best_model_state is not None:
                    model.load_state_dict(best_model_state['model_state_dict'])
                    print(f"已恢复第 {best_model_state['epoch']} 轮的最佳模型 (验证准确率: {best_model_state['best_val_acc']:.2f}%)")
                break
            
            # 每20轮打印一次进度
            if (epoch + 1) % 20 == 0:
                avg_main_loss = train_main_loss / len(train_loader)
                avg_cnn_aux_loss = train_cnn_aux_loss / len(train_loader)
                avg_physics_loss = train_physics_loss / len(train_loader)
                
                if use_pinn:
                    print(f"轮次 {epoch+1}: 训练准确率 {train_acc:.2f}%, 验证准确率 {val_acc:.2f}%")
                    print(f"  损失分解 - 主分类: {avg_main_loss:.4f}, CNN辅助: {avg_cnn_aux_loss:.4f}, PINN物理: {avg_physics_loss:.6f}")
                else:
                    print(f"轮次 {epoch+1}: 训练准确率 {train_acc:.2f}%, 验证准确率 {val_acc:.2f}%")
                    print(f"  损失分解 - 主分类: {avg_main_loss:.4f}, CNN辅助: {avg_cnn_aux_loss:.4f}")
        
        # 如果没有早停，也要恢复最佳模型
        if best_model_state is not None and patience_counter < max_patience:
            model.load_state_dict(best_model_state['model_state_dict'])
            print(f"训练完成，已恢复第 {best_model_state['epoch']} 轮的最佳模型 (验证准确率: {best_model_state['best_val_acc']:.2f}%)")
        
        # 记录交叉验证结果和实际训练轮次
        actual_epochs = epoch + 1  # 实际训练的轮次
        cv_results.append((train_acc, best_val_acc, actual_epochs))
        
        # 清理内存
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        gc.collect()
    
    # 计算交叉验证平均结果和最大早停轮次
    cv_train_accs = [r[0] for r in cv_results]
    cv_val_accs = [r[1] for r in cv_results]
    cv_epochs = [r[2] for r in cv_results]
    avg_cv_train_acc = np.mean(cv_train_accs)
    avg_cv_val_acc = np.mean(cv_val_accs)
    max_early_stop_epoch = max(cv_epochs)
    
    # 根据最大早停轮次确定最终训练轮次（向上取整到5或0结尾）
    final_train_epochs = ((max_early_stop_epoch + 4) // 5) * 5
    
    print(f"\n交叉验证平均结果:")
    print(f"  平均训练准确率: {avg_cv_train_acc:.2f}%")
    print(f"  平均验证准确率: {avg_cv_val_acc:.2f}%")
    print(f"  各折实际训练轮次: {cv_epochs}")
    print(f"  最大早停轮次: {max_early_stop_epoch}")
    print(f"  最终模型训练轮次设定: {final_train_epochs}")
    
    # =================== 阶段三：最终模型训练和测试（论文报告结果）===================
    print("\n=== 最终模型训练（论文报告阶段）===")
    
    # 使用完整训练集训练最终模型
    if repeat_factor > 1:
        augmented_full_dataset = BCI2aDataset(data_dir, [subject_id], transform=None, repeat_factor=repeat_factor)
        augmented_full_dataset.set_augmentation(True)
        # 正确的索引映射：原始索引 -> 扩充后索引
        augmented_train_indices = []
        for idx in train_indices:
            # 对于每个原始索引，找到它在扩充数据集中对应的所有位置
            for rep in range(repeat_factor):
                augmented_idx = idx * repeat_factor + rep
                augmented_train_indices.append(augmented_idx)
        final_train_dataset = Subset(augmented_full_dataset, augmented_train_indices)
    else:
        final_train_dataset = Subset(dataset, train_indices)
    
    final_test_dataset = Subset(dataset, test_indices)
    
    final_train_loader = DataLoader(final_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    final_test_loader = DataLoader(final_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # 初始化最终模型
    final_model = PINN_CNN_BCI(
        leadfield=leadfield,
        fwd_model=fwd,
        epochs_info=epochs_info,
        in_channels=22,
        seq_length=1000,
        num_classes=2,  # 二分类：左手、右手
        dropout_rate=dropout_rate,
        use_pinn=use_pinn,
        sfreq=250
    ).to(device)
    
    final_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    final_optimizer = optim.AdamW(
        final_model.parameters(), 
        lr=learning_rate, 
        weight_decay=0.005,  # 适度的权重衰减
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    # 训练最终模型（使用固定轮次，无需内部验证集）
    print(f"训练最终模型，固定训练轮次: {final_train_epochs}")
    print("使用全部训练数据，无内部验证集划分")
    
    if repeat_factor > 1:
        # 正确的索引映射：原始索引 -> 扩充后索引（使用全部训练数据）
        augmented_final_indices = []
        for idx in train_indices:
            # 对于每个原始索引，找到它在扩充数据集中对应的所有位置
            for rep in range(repeat_factor):
                augmented_idx = idx * repeat_factor + rep
                augmented_final_indices.append(augmented_idx)
        final_train_subset = Subset(augmented_full_dataset, augmented_final_indices)
    else:
        final_train_subset = Subset(dataset, train_indices)
    
    final_train_subset_loader = DataLoader(final_train_subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    for epoch in range(final_train_epochs):
        # 训练
        final_model.train()
        for batch_data, batch_labels in final_train_subset_loader:
            batch_data = batch_data.to(device)
            batch_labels = (batch_labels - 1).long().to(device)
            
            final_optimizer.zero_grad()
            outputs = final_model(batch_data)
            
            # 主分类损失（融合后的预测）
            main_class_loss = final_criterion(outputs['logits'], batch_labels)
            
            # CNN分支辅助损失
            cnn_aux_loss = final_criterion(outputs['cnn_logits'], batch_labels)
            
            # PINN物理损失（如果使用PINN）
            if use_pinn:
                physics_loss = outputs['physics_loss']
                # 多目标损失函数：主分类 + CNN辅助 + PINN物理
                total_loss = (1.0 * main_class_loss +      # 主损失权重: 1.0
                             cnn_aux_weight * cnn_aux_loss +         # CNN辅助权重: 可调
                             physics_weight * physics_loss) # PINN物理权重: 可调
            else:
                physics_loss = torch.tensor(0.0, device=device)
                # 双分支损失：主分类 + CNN辅助
                total_loss = (1.0 * main_class_loss +      # 主损失权重: 1.0
                             cnn_aux_weight * cnn_aux_loss)          # CNN辅助权重: 可调
                
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
            final_optimizer.step()
        
        # 简单的进度打印（无验证集）
        if (epoch + 1) % 20 == 0:
            print(f"最终模型训练进度: {epoch+1}/{final_train_epochs}")
    
    # 最终评估：同一个模型分别在训练集和测试集上评估
    print("\n=== 最终模型评估 ===")
    
    # 评估训练集性能
    final_model.eval()
    train_predictions = []
    train_true_labels = []
    
    with torch.no_grad():
        for batch_data, batch_labels in final_train_loader:
            batch_data = batch_data.to(device)
            batch_labels = (batch_labels - 1).long().to(device)
            
            outputs = final_model(batch_data)
            _, predicted = torch.max(outputs['logits'].data, 1)
            
            train_predictions.extend(predicted.cpu().numpy())
            train_true_labels.extend(batch_labels.cpu().numpy())
    
    # 评估测试集性能
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
    final_train_acc = 100.0 * accuracy_score(train_true_labels, train_predictions)
    final_train_f1 = f1_score(train_true_labels, train_predictions, average='weighted')
    final_train_kappa = cohen_kappa_score(train_true_labels, train_predictions)
    
    final_test_acc = 100.0 * accuracy_score(test_true_labels, test_predictions)
    final_test_f1 = f1_score(test_true_labels, test_predictions, average='weighted')
    final_test_kappa = cohen_kappa_score(test_true_labels, test_predictions)
    
    print(f"最终模型结果（同一个模型）:")
    print(f"  训练集 - 准确率: {final_train_acc:.2f}%, F1: {final_train_f1:.4f}, Kappa: {final_train_kappa:.4f}")
    print(f"  测试集 - 准确率: {final_test_acc:.2f}%, F1: {final_test_f1:.4f}, Kappa: {final_test_kappa:.4f}")
    
    # 清理内存
    del final_model, final_optimizer
    torch.cuda.empty_cache()
    gc.collect()
    
    # 返回结果：交叉验证结果、最终训练集结果、最终验证集结果、最终测试集结果
    return (avg_cv_train_acc, avg_cv_val_acc), (final_train_acc, final_train_f1, final_train_kappa), final_train_epochs, (final_test_acc, final_test_f1, final_test_kappa)


# =================================================================================
# 主函数
# =================================================================================
if __name__ == '__main__':
    # ===================== 参数控制变量（方便修改） =====================
    # 训练参数 - 优化后的设置，匹配简化架构
    BATCH_SIZE = 32          # 增大批次，提高稳定性
    NUM_EPOCHS = 200         # 更多epoch配合早停
    LEARNING_RATE = 0.001    # 提高学习率，简化模型可以承受
    DROPOUT_RATE = 0.4       # 高dropout防止过拟合
    PHYSICS_WEIGHT = 0.2     # 降低物理损失权重
    CNN_AUX_WEIGHT = 0.2     # 大幅降低CNN辅助损失权重
    
    # 数据增强参数 - 减少增强强度
    REPEAT_FACTOR = 3        # 降低数据增强倍数
    
    # 模型参数
    USE_PINN = True  # 是否使用PINN空域特征提取
    
    # 随机性控制
    USE_FIXED_SEED = False  # True: 使用固定种子确保结果可复现, False: 使用随机种子
    FIXED_SEED = 42        # 固定种子值（仅当USE_FIXED_SEED=True时使用）
    
    # 数据路径
    DATA_DIR = r"E:\pycharm\PINN\PINN\data\BCI2a"  # 请根据实际情况修改路径
    
    # ===================== 程序开始执行 =====================
    print("=== 使用 PINN(空域) + CNN(时域) 混合模型进行BCI-IV-2a二分类（左手/右手） ===")
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
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        random.seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(current_seed)
            torch.cuda.manual_seed_all(current_seed)
    else:
        current_seed = int(time.time() * 1000) % 100000  # 使用毫秒级时间戳作为种子
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
        
        for subject_id in available_subjects:
            print(f"\n=== 开始训练被试 S{subject_id} ===\n")
            try:
                cv_results, final_train_result, actual_epochs, final_test_result = train_model(
                    DATA_DIR, 
                    subject_id=subject_id, 
                    num_epochs=NUM_EPOCHS, 
                    batch_size=BATCH_SIZE, 
                    repeat_factor=REPEAT_FACTOR,
                    dropout_rate=DROPOUT_RATE,
                    learning_rate=LEARNING_RATE,
                    physics_weight=PHYSICS_WEIGHT,
                    cnn_aux_weight=CNN_AUX_WEIGHT,
                    use_pinn=USE_PINN
                )
            except Exception as e:
                print(f"被试 S{subject_id} 训练失败: {e}")
                continue

            # 解析结果
            cv_train_acc, cv_val_acc = cv_results
            final_train_acc, final_train_f1, final_train_kappa = final_train_result
            final_test_acc, final_test_f1, final_test_kappa = final_test_result
            
            # 存储结果
            subject_results.append((subject_id, cv_train_acc, cv_val_acc, 
                                  final_train_acc, final_train_f1, final_train_kappa,
                                  actual_epochs, final_test_acc, final_test_f1, final_test_kappa))
            
            # 打印当前被试的详细结果
            print(f"\n--- 被试 S{subject_id} 结果汇总 ---")
            print(f"交叉验证结果（模型选择阶段）:")
            print(f"  平均训练准确率: {cv_train_acc:.2f}%")
            print(f"  平均验证准确率: {cv_val_acc:.2f}%")
            print(f"最终模型结果（论文报告）:")
            print(f"  实际训练轮次: {actual_epochs}")
            print(f"  训练集 - 准确率: {final_train_acc:.2f}%, F1: {final_train_f1:.4f}, Kappa: {final_train_kappa:.4f}")
            print(f"  测试集 - 准确率: {final_test_acc:.2f}%, F1: {final_test_f1:.4f}, Kappa: {final_test_kappa:.4f}")
        
        # 打印所有被试的结果汇总
        print("\n\n" + "="*170)
        print("===== 所有被试结果汇总 (PINN + CNN 混合模型 - 正确的实验设计) =====")
        print("="*170)
        
        if not subject_results:
             print("没有有效的训练结果。")
        else:
            # 计算所有平均值
            avg_cv_train_acc = np.mean([r[1] for r in subject_results])
            avg_cv_val_acc = np.mean([r[2] for r in subject_results])
            avg_final_train_acc = np.mean([r[3] for r in subject_results])
            avg_final_train_f1 = np.mean([r[4] for r in subject_results])
            avg_final_train_kappa = np.mean([r[5] for r in subject_results])
            avg_actual_epochs = np.mean([r[6] for r in subject_results])
            avg_final_test_acc = np.mean([r[7] for r in subject_results])
            avg_final_test_f1 = np.mean([r[8] for r in subject_results])
            avg_final_test_kappa = np.mean([r[9] for r in subject_results])
            
            # 详细表格标题
            print("被试\t交叉验证结果\t\t最终模型结果（论文报告）")
            print("\t训练准确率\t验证准确率\t训练准确率\tF1\tKappa\t实际轮次\t测试准确率\tF1\tKappa")
            print("-" * 150)
            
            # 打印每个被试的结果
            for subj_result in subject_results:
                subj, cv_train_acc, cv_val_acc, final_train_acc, final_train_f1, final_train_kappa, actual_epochs, final_test_acc, final_test_f1, final_test_kappa = subj_result
                print(f"S{subj}\t{cv_train_acc:.2f}%\t\t{cv_val_acc:.2f}%\t\t"
                      f"{final_train_acc:.2f}%\t\t{final_train_f1:.4f}\t{final_train_kappa:.4f}\t"
                      f"{actual_epochs}\t\t{final_test_acc:.2f}%\t\t{final_test_f1:.4f}\t{final_test_kappa:.4f}")
            
            print("-" * 150)
            print(f"平均\t{avg_cv_train_acc:.2f}%\t\t{avg_cv_val_acc:.2f}%\t\t"
                  f"{avg_final_train_acc:.2f}%\t\t{avg_final_train_f1:.4f}\t{avg_final_train_kappa:.4f}\t"
                  f"{avg_actual_epochs:.1f}\t\t{avg_final_test_acc:.2f}%\t\t{avg_final_test_f1:.4f}\t{avg_final_test_kappa:.4f}")
            print("=" * 170)
            
            # 重要结论总结
            print(f"\n📊 实验结论:")
            print(f"🔸 交叉验证阶段（模型选择）: 平均训练准确率 {avg_cv_train_acc:.2f}%, 平均验证准确率 {avg_cv_val_acc:.2f}%")
            print(f"🔸 最终模型（论文报告）: 平均训练准确率 {avg_final_train_acc:.2f}%, 平均测试准确率 {avg_final_test_acc:.2f}%")
            print(f"🔸 平均实际训练轮次: {avg_actual_epochs:.1f} (基于交叉验证最大早停轮次自动确定)")
            
            if avg_final_train_acc >= avg_final_test_acc:
                print(f"✅ 符合预期: 训练集准确率 ≥ 测试集准确率 ({avg_final_train_acc:.2f}% ≥ {avg_final_test_acc:.2f}%)")
            else:
                print(f"⚠️ 需要关注: 训练集准确率 < 测试集准确率，需要进一步分析")
            
            # 保存详细结果到CSV文件
            try:
                import pandas as pd
                
                # 创建详细结果DataFrame
                detailed_results = []
                for subj_result in subject_results:
                    subj, cv_train_acc, cv_val_acc, final_train_acc, final_train_f1, final_train_kappa, actual_epochs, final_test_acc, final_test_f1, final_test_kappa = subj_result
                    
                    detailed_results.append({
                        '被试': f'S{subj}',
                        '交叉验证_训练准确率(%)': round(cv_train_acc, 2),
                        '交叉验证_验证准确率(%)': round(cv_val_acc, 2),
                        '最终模型_训练准确率(%)': round(final_train_acc, 2),
                        '最终模型_训练F1': round(final_train_f1, 4),
                        '最终模型_训练Kappa': round(final_train_kappa, 4),
                        '实际训练轮次': actual_epochs,
                        '最终模型_测试准确率(%)': round(final_test_acc, 2),
                        '最终模型_测试F1': round(final_test_f1, 4),
                        '最终模型_测试Kappa': round(final_test_kappa, 4)
                    })
                
                # 添加平均值行
                detailed_results.append({
                    '被试': '平均',
                    '交叉验证_训练准确率(%)': round(avg_cv_train_acc, 2),
                    '交叉验证_验证准确率(%)': round(avg_cv_val_acc, 2),
                    '最终模型_训练准确率(%)': round(avg_final_train_acc, 2),
                    '最终模型_训练F1': round(avg_final_train_f1, 4),
                    '最终模型_训练Kappa': round(avg_final_train_kappa, 4),
                    '实际训练轮次': round(avg_actual_epochs, 1),
                    '最终模型_测试准确率(%)': round(avg_final_test_acc, 2),
                    '最终模型_测试F1': round(avg_final_test_f1, 4),
                    '最终模型_测试Kappa': round(avg_final_test_kappa, 4)
                })
                
                df = pd.DataFrame(detailed_results)
                
                # 生成文件名（包含时间戳和参数信息）
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"BCI_V4_2class_results_{timestamp}_E{NUM_EPOCHS}_B{BATCH_SIZE}_R{REPEAT_FACTOR}.csv"
                
                df.to_csv(filename, index=False, encoding='utf-8-sig')
                print(f"\n💾 详细结果已保存到: {filename}")
                
            except ImportError:
                print("\n⚠️  pandas未安装，跳过CSV文件保存")
            except Exception as e:
                print(f"\n⚠️  保存CSV文件时出错: {e}")
            
            print("=" * 170)
        
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








































































































































