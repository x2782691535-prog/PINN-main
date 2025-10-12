# 5 PINN空域CNN时域


import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
import mne
from mne.channels import make_standard_montage
from mne.io import RawArray
from mne.forward import make_forward_solution
from mne.simulation import simulate_sparse_stc
import argparse
import gc
import torch.nn.functional as F
import time
from scipy import signal
import copy
import contextlib
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.linalg import eigh
from sklearn.feature_selection import mutual_info_classif

# 设置matplotlib字体为微软黑体，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False


# =================================================================================
# FBCSP 和 CSP 相关类已被移除，因为模型不再使用它们
# =================================================================================


# =================================================================================
# 简化模块：1D-CNN 时域特征提取器（替代复杂的时频变换+2D-CNN）
# =================================================================================
class CNN1DFeatureExtractor(nn.Module):
    """
    用于直接处理EEG时域信号的1D卷积神经网络
    大幅简化架构，减少参数量，提高训练稳定性
    """
    def __init__(self, in_channels=22, seq_length=1000, dropout_rate=0.35):
        """
        初始化1D-CNN模块

        参数:
            in_channels: 输入通道数 (EEG通道数)
            seq_length: 序列长度
            dropout_rate: Dropout比率
        """
        super(CNN1DFeatureExtractor, self).__init__()
        
        # 轻量级1D卷积架构
        self.conv_block = nn.Sequential(
            # 第一层：捕获局部时域特征
            # 输入形状: (batch, 22, 1000)
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.LeakyReLU(0.1),
            nn.AvgPool1d(kernel_size=4),  # -> (batch, 32, 250)
            nn.Dropout(dropout_rate * 0.5),

            # 第二层：捕获中等尺度特征
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.LeakyReLU(0.1),
            nn.AvgPool1d(kernel_size=4),  # -> (batch, 64, 62)
            nn.Dropout(dropout_rate * 0.7),

            # 第三层：捕获高级特征
            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.AvgPool1d(kernel_size=2),  # -> (batch, 96, 31)
            nn.Dropout(dropout_rate)
        )
        
        # 全局平均池化，进一步减少参数
        self.global_pool = nn.AdaptiveAvgPool1d(8)  # -> (batch, 96, 8)
        self.flatten = nn.Flatten()
        
        # 大幅减少的特征维度: 96 * 8 = 768 (相比原来的5760减少了87%)
        self.output_dim = 96 * 8
        
    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入EEG信号，形状为 (batch, channels, time)
        
        返回:
            提取的时域特征，形状为 (batch, output_dim)
        """
        x = self.conv_block(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        return x


# =================================================================================
# 1D-CNN 特征提取器 (原EEGFeatureExtractor)，现在明确其作用是为PINN分支提供输入
# =================================================================================
class PINNInputExtractor(nn.Module):
    def __init__(self, input_channels, seq_length=1000, dropout_rate=0.25, activation_function='tanh'):
        """
        为PINN分支提取深度特征，架构与TensorFlow版本一致
        使用TimeDistributed Dense层结构替代1D-CNN

        参数:
            input_channels: 输入通道数
            seq_length: 序列长度
            dropout_rate: Dropout比率
            activation_function: 激活函数类型
        """
        super(PINNInputExtractor, self).__init__()
        
        # 根据激活函数字符串选择对应的PyTorch激活函数
        if activation_function.lower() == 'tanh':
            self.activation = nn.Tanh()
        elif activation_function.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation_function.lower() == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.1)
        else:
            self.activation = nn.Tanh()  # 默认使用tanh
        
        # 与TensorFlow版本一致的深度特征提取结构
        # TimeDistributed(Dense(64)) -> 对每个时间步应用相同的全连接层
        self.conv1 = nn.Linear(input_channels, 64)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.conv2 = nn.Linear(64, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.conv3 = nn.Linear(128, 256)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # 深度特征层
        self.deep_features = nn.Linear(256, 512)
        self.dropout_deep = nn.Dropout(dropout_rate)
        
        # 输出维度设置为512，与TensorFlow版本一致
        self.output_dim = 512
        
    def forward(self, x):
        """
        前向传播，模拟TensorFlow的TimeDistributed行为
        
        参数:
            x: 输入数据，形状为(batch_size, channels, seq_length)
        
        返回:
            深度特征，形状为(batch_size, output_dim)
        """
        # 转换维度以适应TimeDistributed的行为
        # 从(batch_size, channels, seq_length)转换为(batch_size, seq_length, channels)
        x = x.transpose(1, 2)  # (batch_size, seq_length, channels)
        
        # 应用第一层（相当于TimeDistributed(Dense(64))）
        x = self.conv1(x)  # (batch_size, seq_length, 64)
        x = self.activation(x)
        x = self.dropout1(x)
        
        # 应用第二层（相当于TimeDistributed(Dense(128))）
        x = self.conv2(x)  # (batch_size, seq_length, 128)
        x = self.activation(x)
        x = self.dropout2(x)
        
        # 应用第三层（相当于TimeDistributed(Dense(256))）
        x = self.conv3(x)  # (batch_size, seq_length, 256)
        x = self.activation(x)
        x = self.dropout3(x)
        
        # 深度特征层（相当于TimeDistributed(Dense(512))）
        x = self.deep_features(x)  # (batch_size, seq_length, 512)
        x = self.activation(x)
        x = self.dropout_deep(x)
        
        # 全局平均池化，将时间维度聚合
        # 相当于对时间维度取平均，得到(batch_size, 512)
        x = torch.mean(x, dim=1)
        
        return x


# =================================================================================
# 修改后的主模型：结合PINN和1D-CNN
# =================================================================================
class PINN_CNN_BCI(nn.Module):
    def __init__(self, leadfield, fwd_model, epochs_info, in_channels=22, seq_length=1000, num_classes=2, 
                 dropout_rate=0.35, use_pinn=True, sfreq=250):
        """
        初始化 PINN_CNN_BCI 模型
        - PINN提供空域特征
        - 1D-CNN提供时域特征

        参数:
            leadfield: 引线场矩阵 (PyTorch tensor)
            fwd_model: MNE前向模型对象
            epochs_info: MNE epochs info对象
            in_channels: 输入通道数
            seq_length: 序列长度
            num_classes: 类别数
            dropout_rate: Dropout率
            use_pinn: 是否使用物理信息神经网络
            sfreq: 采样频率
        """
        super(PINN_CNN_BCI, self).__init__()
        
        self.in_channels = in_channels
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.use_pinn = use_pinn
        
        device = leadfield.device if isinstance(leadfield, torch.Tensor) else 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # PINN 相关组件
        self.lead_field = leadfield
        self.fwd_model = fwd_model
        self.epochs_info = epochs_info
        
        # --- 分支1: 时域特征 (1D-CNN) ---
        # 简化架构：直接处理EEG时域信号，避免复杂的时频变换
        self.cnn_1d_extractor = CNN1DFeatureExtractor(
            in_channels=in_channels,
            seq_length=seq_length,
            dropout_rate=dropout_rate
        ).to(device)
        
        # --- 分支2: 空域特征 (PINN) ---
        # 1D-CNN用于为PINN的源定位网络提供输入
        self.pinn_input_extractor = PINNInputExtractor(
            input_channels=in_channels,
            seq_length=seq_length,
            dropout_rate=dropout_rate
        ).to(device)
        
        # --- 特征融合与分类 ---
        # 计算融合前两个分支的特征维度
        # 新的维度：768 (1D-CNN) + 512 (PINN) = 1280，相比原来的6272减少了80%
        fusion_input_dim = self.cnn_1d_extractor.output_dim + self.pinn_input_extractor.output_dim
        
        # 简化的特征融合：由于输入维度大幅减少(1280)，使用更轻量的结构
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        ).to(device)
        
        # 仅在启用PINN时创建源空间映射网络
        if self.use_pinn:
            # 源定位网络，与TensorFlow版本一致：TimeDistributed(Dense(n_dipoles, activation='linear'))
            # 直接从深度特征映射到源空间，不使用中间层
            self.source_fc = nn.Linear(self.pinn_input_extractor.output_dim, fwd_model['sol']['data'].shape[1]).to(device)
            
            # 创建物理损失计算模块
            self.physics_loss_calculator = self._create_physics_loss_calculator(fwd_model, device)
            print("已创建PINN物理损失计算模块")
        
        # 创建分类器头，输入为融合后的特征
        self.classifier = nn.Linear(64, num_classes).to(device)
    
    def _create_physics_loss_calculator(self, fwd_model, device):
        """创建物理损失计算模块（与原代码一致）"""
        class TensorFlowStylePhysicsLoss(nn.Module):
            def __init__(self, fwd_model, device):
                super().__init__()
                self.fwd_model = fwd_model
                self.device = device
                self.leadfield = torch.tensor(fwd_model['sol']['data'], dtype=torch.float32).to(device)
                
                # 可训练权重参数，与TensorFlow版本一致
                self.physics_weight = nn.Parameter(torch.tensor(0.4, dtype=torch.float32))
                self.poisson_weight = nn.Parameter(torch.tensor(0.7, dtype=torch.float32))
                self.boundary_weight = nn.Parameter(torch.tensor(0.3, dtype=torch.float32))
                self.dirichlet_weight = nn.Parameter(torch.tensor(0.6, dtype=torch.float32))
                self.robin_weight = nn.Parameter(torch.tensor(0.4, dtype=torch.float32))
                
                # 从前向模型获取源点坐标
                self._setup_source_coordinates(fwd_model)
                
            def _setup_source_coordinates(self, fwd_model):
                """设置源点坐标，与TensorFlow版本一致"""
                # 尝试从前向模型获取真实的源点坐标
                if 'source_rr' in fwd_model:
                    source_coords = torch.tensor(fwd_model['source_rr'], dtype=torch.float32).to(self.device)
                else:
                    # 使用leadfield矩阵的结构推导源点位置
                    n_sources = self.leadfield.shape[1]
                    n_dipoles_per_vertex = 3
                    n_vertices = n_sources // n_dipoles_per_vertex
                    
                    # 创建基于大脑解剖结构的源点分布（球坐标转笛卡尔坐标）
                    import math
                    vertices_per_dim = int(math.sqrt(n_vertices))
                    theta = torch.linspace(0.0, math.pi, vertices_per_dim)
                    phi = torch.linspace(0.0, 2.0 * math.pi, vertices_per_dim)
                    
                    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')
                    theta_flat = theta_grid.flatten()[:n_vertices]
                    phi_flat = phi_grid.flatten()[:n_vertices]
                    
                    # 球坐标转换为笛卡尔坐标（大脑半径约8cm）
                    radius = 0.08  # 8cm in meters
                    x_coords = radius * torch.sin(theta_flat) * torch.cos(phi_flat)
                    y_coords = radius * torch.sin(theta_flat) * torch.sin(phi_flat)
                    z_coords = radius * torch.cos(theta_flat)
                    
                    # 为每个顶点创建3个方向的偶极子
                    source_coords_list = []
                    for i in range(min(n_vertices, len(x_coords))):
                        for j in range(n_dipoles_per_vertex):
                            source_coords_list.append([x_coords[i], y_coords[i], z_coords[i]])
                    
                    n_sources_actual = min(n_sources, len(source_coords_list))
                    source_coords = torch.tensor(source_coords_list[:n_sources_actual], dtype=torch.float32).to(self.device)
                
                self.source_coords = source_coords
                
            def forward(self, source_activations):
                """前向传播，计算物理损失"""
                try:
                    # 使用简化的物理约束，避免复杂的自动微分
                    smoothness_loss = self._smoothness_constraint(source_activations)
                    boundary_loss = self._boundary_condition_loss(source_activations)
                    
                    # 使用可训练权重组合损失
                    weights = torch.softmax(torch.stack([self.poisson_weight, self.boundary_weight]), dim=0)
                    physics_loss = weights[0] * smoothness_loss + weights[1] * boundary_loss
                    
                    return torch.sigmoid(self.physics_weight) * physics_loss
                except Exception as e:
                    # 如果物理损失计算失败，返回零损失并打印警告
                    print(f"警告：物理损失计算失败: {e}")
                    return torch.tensor(0.0, device=self.device, requires_grad=True)
            
            def _smoothness_constraint(self, source_activations):
                """空间平滑约束，替代复杂的泊松方程"""
                # 基于源点坐标的空间平滑性约束
                batch_size, n_sources = source_activations.shape
                
                if n_sources <= 1:
                    return torch.tensor(0.0, device=self.device, requires_grad=True)
                
                # 确保source_coords的维度正确
                if hasattr(self, 'source_coords') and self.source_coords.shape[0] >= n_sources:
                    coords = self.source_coords[:n_sources]
                    
                    # 计算相邻源点之间的距离权重
                    distances = torch.cdist(coords, coords)  # (n_sources, n_sources)
                    
                    # 使用高斯权重，近邻源点应该有相似的激活
                    sigma = 0.02  # 2cm的空间尺度
                    weights = torch.exp(-distances**2 / (2 * sigma**2))
                    weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-8)
                    
                    # 计算平滑性损失
                    smoothness_loss = 0.0
                    for b in range(batch_size):
                        activations = source_activations[b, :]  # (n_sources,)
                        
                        # 计算加权平均激活
                        weighted_avg = torch.matmul(weights, activations)  # (n_sources,)
                        
                        # 平滑性约束：每个源点的激活应该接近其邻域的加权平均
                        smoothness_loss += torch.mean((activations - weighted_avg) ** 2)
                    
                    return smoothness_loss / batch_size
                else:
                    # 如果没有坐标信息，使用简单的空间平滑约束
                    # 相邻源点的激活应该相似
                    diff_loss = torch.mean((source_activations[:, 1:] - source_activations[:, :-1]) ** 2)
                    return diff_loss
            
            def _boundary_condition_loss(self, source_activations):
                """边界条件损失，与TensorFlow版本一致"""
                n_sources = source_activations.shape[1]
                boundary_size = max(1, n_sources // 10)
                
                # Dirichlet边界条件（边界处电势为零）
                dirichlet_loss = torch.mean(source_activations[:, :boundary_size] ** 2)
                
                # Robin边界条件（混合边界条件）
                robin_loss = torch.mean(source_activations[:, -boundary_size:] ** 2)
                
                # 参考电极约束（总电势为零）
                reference_loss = torch.mean(torch.sum(source_activations, dim=1) ** 2)
                
                # 使用可训练权重组合边界条件
                boundary_weights = torch.softmax(torch.stack([self.dirichlet_weight, self.robin_weight]), dim=0)
                boundary_loss = (boundary_weights[0] * dirichlet_loss + 
                               boundary_weights[1] * robin_loss + 
                               0.1 * reference_loss)  # 参考电极约束权重固定为0.1
                
                return boundary_loss
            
            def get_physics_weight(self):
                """获取当前物理损失权重"""
                return torch.sigmoid(self.physics_weight).item()
        
        return TensorFlowStylePhysicsLoss(fwd_model, device)
    
    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入数据，形状为(batch_size, n_channels, seq_length)
            
        返回:
            包含logits和PINN相关输出的字典
        """
        # --- 时域特征提取 (简化的1D-CNN分支) ---
        features_1d_cnn = self.cnn_1d_extractor(x)
        
        # --- 空域特征提取 (PINN分支) ---
        # 深度特征提取用于物理约束
        features_1d_pinn = self.pinn_input_extractor(x)
        
        # --- 特征融合 ---
        combined_features = torch.cat((features_1d_cnn, features_1d_pinn), dim=1)
        fused_features = self.feature_fusion(combined_features)
        
        # --- 分类 ---
        logits = self.classifier(fused_features)
        
        # --- 物理信息映射 (如果启用PINN) ---
        projected_features = None
        source_activations = None
        if self.use_pinn:
            # 从PINN分支的深度特征生成源激活（与TensorFlow版本一致）
            source_activations = self.source_fc(features_1d_pinn)
            
            # 使用引线场矩阵投影到电极空间（physics_output）
            projected_features = torch.matmul(source_activations, self.lead_field.t())
            # 标准化投影特征
            projected_features = nn.functional.normalize(projected_features, p=2, dim=1)
        
        # 返回与TensorFlow版本一致的双输出结构
        if self.use_pinn:
            return {
                'main_output': source_activations,      # 主输出：源定位结果
                'physics_output': projected_features,   # 物理输出：重建的EEG
                'logits': logits,                       # 分类logits
                'deep_features': fused_features         # 深度特征（向后兼容）
            }
        else:
            return {
                'logits': logits,
                'deep_features': fused_features
            }

# 1. 数据加载与预处理 (与原代码一致)
class BCI2aDataset(Dataset):
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
        self.apply_augmentation = False  # 默认不应用数据增强
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
                # 需要将时间和试验维度对换，保持通道维度不变
                # 最终格式: [试验, 通道, 时间点]
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
    
    def apply_random_noise(self, eeg_data, noise_level=0.005):
        """添加非常温和的随机噪声进行数据增强"""
        # 对不同通道应用不同强度的噪声，更接近实际EEG噪声特征
        if eeg_data.ndim == 2:  # 2D数据 [通道数, 时间点数]
            # 对不同通道使用略微不同的噪声水平
            channel_noise_level = np.random.uniform(0.001, noise_level, (eeg_data.shape[0], 1))
            noise = np.random.normal(0, 1, eeg_data.shape) * channel_noise_level
            return eeg_data + noise
        else:
            print(f"警告: apply_random_noise无法处理{eeg_data.ndim}D数据")
            return eeg_data
    
    def apply_amplitude_scaling(self, eeg_data, scale_range=(0.95, 1.05)):
        """随机振幅缩放，模拟信号强度变化"""
        # 对不同通道应用不同的缩放因子，更接近真实数据，但范围缩小
        if eeg_data.ndim == 2:  # 2D数据 [通道数, 时间点数]
            min_scale, max_scale = scale_range
            channel_scales = np.random.uniform(min_scale, max_scale, (eeg_data.shape[0], 1))
            return eeg_data * channel_scales
        else:
            # 简单缩放
            scale = np.random.uniform(0.95, 1.05)  # 非常温和的缩放
            return eeg_data * scale
            
    def apply_time_shift(self, eeg_data, max_shift=10):
        """
        应用随机时间偏移数据增强
        
        参数:
            eeg_data: 形状为[通道, 时间点]的EEG数据
            max_shift: 最大偏移时间点数
            
        返回:
            应用时间偏移后的EEG数据
        """
        if max_shift <= 0:
            return eeg_data
            
        # 随机选择偏移量
        shift = np.random.randint(-max_shift, max_shift + 1)
        
        # 应用偏移
        if shift > 0:
            # 向右偏移
            shifted_data = np.zeros_like(eeg_data)
            shifted_data[:, shift:] = eeg_data[:, :-shift]
            return shifted_data
        elif shift < 0:
            # 向左偏移
            shifted_data = np.zeros_like(eeg_data)
            shifted_data[:, :shift] = eeg_data[:, -shift:]
            return shifted_data
        else:
            # 无偏移
            return eeg_data
            
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
            should_augment = self.augmentation_flags[idx]  # 判断是否对该样本应用增强
            label = self.labels[real_idx]
        else:
            real_idx = idx
            should_augment = False  # 非扩充模式下，默认不增强
            label = self.labels[idx]
            
        # 创建副本以避免修改原始数据
        data = self.data[real_idx].copy().astype(np.float32)
        
        # 如果是扩充样本且启用了增强，应用随机增强方法
        if should_augment and self.apply_augmentation:
            # 应用随机高斯噪声
            noise_level = 0.005
            channel_noise_level = np.random.uniform(0.001, noise_level, (data.shape[0], 1))
            noise = np.random.normal(0, 1, data.shape) * channel_noise_level
            data = data + noise
            
            # 应用随机幅度缩放
            scale_range = (0.95, 1.05)
            channel_scales = np.random.uniform(scale_range[0], scale_range[1], (data.shape[0], 1))
            data = data * channel_scales
            
            # 应用随机时间偏移
            # data = self.apply_time_shift(data)
            
            # 确保增强后的数据仍然是float32类型
            data = data.astype(np.float32)
            
        # 应用额外的转换（如果有）
        if self.transform:
            data = self.transform(data)
            
        # 最终确保数据是float32类型
        data = data.astype(np.float32)
            
        # 返回原始标签（1或2），在训练和验证循环中统一处理映射
        return data, label


# 2. 头模型构建 (与原代码一致)
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


# 4. 训练与评估 (修改以使用新模型)
def train_model(data_dir, subject_id=None, num_epochs=500, batch_size=64, repeat_factor=3, 
              dropout_rate=0.5, learning_rate=0.0025, physics_weight=0.4, use_pinn=True):
    """
    训练PINN+1D-CNN模型函数
    (移除了fbcsp_components和freq_band_type参数)
    """
    # 确保mne模块可用
    import mne
    
    # 记录开始时间
    start_time = time.time()
    
    # 创建历史记录字典来存储训练过程
    history = {
        'train_loss': [],
        'train_class_loss': [],
        'train_physics_loss': [], 
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    device = torch.device('cuda')
    print(f"使用设备: {device}")

    # 添加硬件信息检测，只在第一个被试时打印
    if subject_id == 1:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")

    # 构建头模型 (逻辑与原代码一致)
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
            print("请确保FreeSurfer subjects目录路径正确，或通过mne.set_config('SUBJECTS_DIR', 'path/to/your/subjects')设置")
            return [], (0,0,0)

        fwd = build_head_model(subjects_dir, subject='fsaverage')  # 返回Forward对象
        leadfield = torch.tensor(fwd['sol']['data'], dtype=torch.float32).to(device)
        # 创建一个示例epochs对象用于获取info
        import mne
        ch_names = [
            'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
        ]
        sfreq = 250.
        epochs_info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        montage = mne.channels.make_standard_montage('standard_1020')
        epochs_info.set_montage(montage)
        # 创建一个简单的epochs对象
        dummy_data = np.zeros((1, len(ch_names), 1000))
        events = np.array([[0, 0, 1]])
        epochs = mne.EpochsArray(dummy_data, epochs_info, events=events, tmin=0.0)
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
            # 创建一个示例epochs对象用于获取info
            import mne
            ch_names = [
                'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
            ]
            sfreq = 250.
            epochs_info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
            montage = mne.channels.make_standard_montage('standard_1020')
            epochs_info.set_montage(montage)
            # 创建一个简单的epochs对象
            dummy_data = np.zeros((1, len(ch_names), 1000))
            events = np.array([[0, 0, 1]])
            epochs = mne.EpochsArray(dummy_data, epochs_info, events=events, tmin=0.0)

    # 设置随机种子确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)
        
    # 加载被试 S{subject_id} 的数据...
    print(f"加载被试 S{subject_id} 的数据...")
    
    # 创建数据集，不进行数据扩充
    dataset = BCI2aDataset(data_dir, [subject_id], transform=None, repeat_factor=1)
    
    print(f"被试 S{subject_id} 原始数据集大小: {len(dataset)}个样本")
    # ===================== 新增：分层划分训练集和测试集 =====================
    all_indices = np.arange(len(dataset))
    all_labels = np.array([dataset[i][1] for i in all_indices])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(all_indices, all_labels))
    train_indices = all_indices[train_idx]
    test_indices = all_indices[test_idx]
    # 构建训练集和测试集
    
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    print(f"分层划分后：训练集 {len(train_dataset)}，测试集 {len(test_dataset)}")
    # ===================== 只在训练集上做9折交叉验证 =====================
    kf = KFold(n_splits=9, shuffle=True, random_state=42)
    indices = np.arange(len(train_dataset))
    all_fold_results = []
    fold_num = 1
    for train_fold_idx, val_fold_idx in kf.split(indices):
        print(f"\n=== 训练集9折交叉验证：第{fold_num}折 ===")
        # 每一折前重置history
        history = {
            'train_loss': [], 'train_class_loss': [], 'train_physics_loss': [],
            'train_acc': [], 'val_loss': [], 'val_acc': []
        }
        # 如果需要数据增强，只对训练集进行增强
        if repeat_factor > 1:
            print(f"数据增强状态: 启用 (仅应用于训练集，将创建{repeat_factor-1}倍的增强数据)")
            class AugmentedDataset(Dataset):
                def __init__(self, orig_dataset, indices, enhance=True):
                    self.dataset = orig_dataset
                    self.indices = indices
                    self.enhance = enhance
                    self.sfreq = 250
                    self.freq_bands = [
                        (4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)
                    ]
                def __len__(self):
                    return len(self.indices)
                def __getitem__(self, idx):
                    orig_idx = self.indices[idx]
                    data = self.dataset[orig_idx][0].copy().astype(np.float32)
                    label = self.dataset[orig_idx][1]
                    if self.enhance:
                        data = self._apply_augmentation(data)
                    data = data.astype(np.float32)
                    # 修正：始终返回Tensor
                    return torch.from_numpy(data).float(), torch.tensor(label, dtype=torch.long)
                def _apply_augmentation(self, data):
                    if np.random.rand() < 0.8:
                        noise_level = np.random.uniform(0.001, 0.03)
                        channel_noise_level = np.random.uniform(0.0005, noise_level, (data.shape[0], 1))
                        noise = np.random.normal(0, 1, data.shape) * channel_noise_level
                        data = data + noise
                    if np.random.rand() < 0.7:
                        scale_range = (0.8, 1.2)
                        channel_scales = np.random.uniform(scale_range[0], scale_range[1], (data.shape[0], 1))
                        data = data * channel_scales
                    if np.random.rand() < 0.5:
                        max_shift = 20
                        shift = np.random.randint(-max_shift, max_shift + 1)
                        if shift != 0:
                            data_shifted = np.zeros_like(data)
                            if shift > 0:
                                data_shifted[:, shift:] = data[:, :-shift]
                            else:
                                data_shifted[:, :shift] = data[:, -shift:]
                            data = data_shifted
                    if np.random.rand() < 0.6:
                        n_bands = np.random.randint(1, 4)
                        selected_bands = np.random.choice(len(self.freq_bands), n_bands, replace=False)
                        for band_idx in selected_bands:
                            fmin, fmax = self.freq_bands[band_idx]
                            scale = np.random.uniform(0.2, 2.0)
                            nyq = 0.5 * self.sfreq
                            low = fmin / nyq
                            high = fmax / nyq
                            b, a = signal.butter(5, [low, high], btype='band')
                            band_signal = np.zeros_like(data)
                            for ch in range(data.shape[0]):
                                band_signal[ch] = signal.filtfilt(b, a, data[ch])
                            data = data + (band_signal * (scale - 1.0))
                    if np.random.rand() < 0.3:
                        n_swaps = np.random.randint(1, 4)
                        n_channels = data.shape[0]
                        for _ in range(n_swaps):
                            if n_channels >= 2:
                                ch1 = np.random.randint(0, n_channels)
                                ch2_options = [ch1-1, ch1+1]
                                ch2_options = [ch for ch in ch2_options if 0 <= ch < n_channels]
                                if ch2_options:
                                    ch2 = np.random.choice(ch2_options)
                                    data[[ch1, ch2]] = data[[ch2, ch1]]
                    if np.random.rand() < 0.2:
                        n_channels_affected = np.random.randint(1, 4)
                        channels = np.random.choice(data.shape[0], n_channels_affected, replace=False)
                        burst_start = np.random.randint(0, data.shape[1] - 50)
                        burst_length = np.random.randint(10, 50)
                        burst_end = min(burst_start + burst_length, data.shape[1])
                        burst_noise = np.random.normal(0, np.random.uniform(0.5, 2.0), (n_channels_affected, burst_end - burst_start))
                        for i, ch in enumerate(channels):
                            data[ch, burst_start:burst_end] += burst_noise[i]
                    return data
            train_set_original = AugmentedDataset(train_dataset, train_fold_idx, enhance=False)
            all_datasets = [train_set_original]
            for _ in range(repeat_factor - 1):
                augmented_set = AugmentedDataset(train_dataset, train_fold_idx, enhance=True)
                all_datasets.append(augmented_set)
            train_set = ConcatDataset(all_datasets)
            print(f"训练集总大小(含增强): {len(train_set)}，其中原始样本: {len(train_set_original)}，增强样本: {len(train_set)-len(train_set_original)}")
        else:
            train_set = Subset(train_dataset, train_fold_idx)
            print(f"数据增强状态: 禁用 (使用原始数据)")
        val_set = Subset(train_dataset, val_fold_idx)
        print(f"被试 S{subject_id} 训练集大小: {len(train_set)}, 验证集大小: {len(val_set)}")
        # 创建DataLoader
        train_loader = DataLoader(
            train_set, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        # 预计算采样频率
        sfreq = 250
        
        # ==================== 主要修改点: 模型实例化 ====================
        model = PINN_CNN_BCI(
            leadfield, 
            fwd,
            epochs.info,
            in_channels=dataset.data.shape[1], 
            seq_length=dataset.data.shape[2], 
            num_classes=2,
            dropout_rate=dropout_rate,
            use_pinn=use_pinn,
            sfreq=sfreq
        ).to(device)
        print(f"引线场形状: {leadfield.shape}")
        print(f"模型架构: PINN (空域) + 1D-CNN (时域)")
        print(f"PINN模式: {'启用' if model.use_pinn else '禁用'}")
        if model.use_pinn:
            print(f"  └─ 源定位维度: {fwd['sol']['data'].shape[1]} 个源点")
        # =============================================================
        
        # 优化器
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
        
        # 学习率调度
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-5
        )
        
        # 计算类别权重
        all_labels = []
        for _, labels in train_loader:
            all_labels.append(labels.flatten())
        all_labels = torch.cat(all_labels)
        mapped_labels = all_labels - 1
        label_counts = torch.bincount(mapped_labels, minlength=2)
        beta = 0.99
        effective_num = 1.0 - torch.pow(beta, label_counts.float())
        weights = (1.0 - beta) / (effective_num + 1e-8)
        weights = weights / weights.sum() * 2
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        
        # 早停机制
        patience = 15
        best_val_acc = 0.0
        early_stop_counter = 0
        best_f1 = 0.0
        best_epoch = 0
        best_model_state = None

        # --- 训练循环 (此部分逻辑与原代码完全一致) ---
        dynamic_weight_interval = 5
        for epoch in range(num_epochs):
            if epoch % dynamic_weight_interval == 0:
                all_labels = []
                for _, labels in train_loader:
                    all_labels.append(labels.flatten())
                all_labels = torch.cat(all_labels)
                mapped_labels = all_labels - 1
                label_counts = torch.bincount(mapped_labels, minlength=2)
                beta = 0.99
                effective_num = 1.0 - torch.pow(beta, label_counts.float())
                weights = (1.0 - beta) / (effective_num + 1e-8)
                weights = weights / weights.sum() * 2
                criterion.weight = weights.to(device)

            epoch_start_time = time.time()
            
            model.train()
            train_loss = 0.0
            train_class_loss = 0.0
            train_physics_loss = 0.0
            batch_count = 0
            correct = 0
            total = 0
            
            for batch_idx, (eeg_data, labels) in enumerate(train_loader):
                batch_count += 1
                eeg_data = eeg_data.to(device, dtype=torch.float32)
                physical_target = torch.mean(eeg_data, dim=2)
                physical_target = F.normalize(physical_target, p=2, dim=1)
                
                labels = labels.flatten() - 1
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(eeg_data)
                
                loss_class = criterion(outputs['logits'], labels)
                
                if model.use_pinn and 'main_output' in outputs and outputs['main_output'] is not None:
                    # 使用新的双输出结构
                    physics_loss = model.physics_loss_calculator(outputs['main_output'])
                    reconstruction_loss = nn.MSELoss()(outputs['physics_output'], physical_target)
                    total_physics_loss = physics_loss + 0.1 * reconstruction_loss
                    current_physics_weight = model.physics_loss_calculator.get_physics_weight()
                    loss = current_physics_weight * total_physics_loss + (1 - current_physics_weight) * loss_class
                    loss_physics = total_physics_loss
                else:
                    loss_physics = torch.tensor(0.0, device=device)
                    loss = loss_class
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_physics_loss += loss_physics.item()
                train_class_loss += loss_class.item()
                
                _, predicted = outputs['logits'].max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                if model.use_pinn:
                    torch.cuda.empty_cache()
            
            avg_train_loss = train_loss / max(batch_count, 1)
            avg_train_phys = train_physics_loss / max(batch_count, 1)
            avg_train_class = train_class_loss / max(batch_count, 1)
            train_acc = 100.0 * correct / total

            history['train_loss'].append(avg_train_loss)
            history['train_class_loss'].append(avg_train_class)
            history['train_physics_loss'].append(avg_train_phys)
            history['train_acc'].append(train_acc)

            epoch_time = time.time() - epoch_start_time

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for batch_idx, (eeg_data, labels) in enumerate(val_loader):
                    eeg_data = eeg_data.to(device, dtype=torch.float32)
                    labels = labels.flatten() - 1
                    labels = labels.to(device)
                    outputs = model(eeg_data)
                    loss_class = criterion(outputs['logits'], labels)
                    physical_target = torch.mean(eeg_data, dim=2)
                    physical_target = F.normalize(physical_target, p=2, dim=1)
                    if model.use_pinn and 'main_output' in outputs and outputs['main_output'] is not None:
                        # 使用新的双输出结构
                        physics_loss = model.physics_loss_calculator(outputs['main_output'])
                        reconstruction_loss = nn.MSELoss()(outputs['physics_output'], physical_target)
                        total_physics_loss = physics_loss + 0.1 * reconstruction_loss
                        current_physics_weight = model.physics_loss_calculator.get_physics_weight()
                        loss = current_physics_weight * total_physics_loss + (1 - current_physics_weight) * loss_class
                    else:
                        loss = loss_class
                    val_loss += loss.item()
                    _, predicted = outputs['logits'].max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    val_preds.extend(predicted.cpu().numpy())
                    val_targets.extend(labels.cpu().numpy())
            val_acc = 100.0 * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            val_f1 = f1_score(val_targets, val_preds, average='macro')
            
            scheduler.step()
            
            is_best = val_acc > best_val_acc
            
            if is_best:
                best_val_acc = val_acc
                best_f1 = val_f1
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            if (epoch+1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1 or is_best:
                print(f"被试 S{subject_id} | Epoch {epoch+1}/{num_epochs} | "
                      f"时间: {epoch_time:.2f}s | "
                      f"训练准确率: {train_acc:.2f}% | "
                      f"验证准确率: {val_acc:.2f}% | "
                      f"F1: {val_f1:.4f}")
                
                if model.use_pinn and (epoch+1) % 50 == 0:
                    current_physics_w = model.physics_loss_calculator.get_physics_weight()
                    print(f"  └─ 当前物理损失权重: {current_physics_w:.4f}")
            
            if model.use_pinn:
                torch.cuda.empty_cache()
            
            if early_stop_counter >= patience:
                print(f"早停: 被试 S{subject_id} 验证准确率已经{patience}个epoch没有提高")
                break
        
        print(f"\n被试 S{subject_id} 训练性能分析:")
        print(f"最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch+1})")
        print(f"最佳F1分数: {best_f1:.4f}")
        
        if best_model_state:
            model.load_state_dict(best_model_state)
            print(f"加载最佳模型 (Epoch {best_epoch+1})")
        
        model.eval()
        train_preds, train_truths = [], []
        train_loader_no_shuffle = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        with torch.no_grad():
            for eeg_data, labels in train_loader_no_shuffle:
                eeg_data = eeg_data.to(device, dtype=torch.float32)
                labels = labels.flatten() - 1
                labels = labels.to(device)
                outputs = model(eeg_data)
                _, predicted = outputs['logits'].max(1)
                train_preds.extend(predicted.cpu().tolist())
                train_truths.extend(labels.cpu().tolist())
        
        train_acc_final = accuracy_score(train_truths, train_preds) * 100
        train_f1_final = f1_score(train_truths, train_preds, average='macro')
        
        all_fold_results.append((fold_num, train_acc_final, train_f1_final, best_val_acc, best_f1))

        # 绘图
        plt.figure(figsize=(12, 10))
        def smooth_curve(points, factor=0.8):
            smoothed_points = []
            for point in points:
                if smoothed_points:
                    previous = smoothed_points[-1]
                    smoothed_points.append(previous * factor + point * (1 - factor))
                else:
                    smoothed_points.append(point)
            return smoothed_points
        plt.subplot(2, 2, 1)
        plt.plot(range(len(history['train_loss'])), smooth_curve(history['train_loss']), label='训练总损失')
        plt.plot(range(len(history['val_loss'])), smooth_curve(history['val_loss']), label='验证损失')
        plt.title(f'被试S{subject_id}的损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.subplot(2, 2, 2)
        plt.plot(range(len(history['train_class_loss'])), smooth_curve(history['train_class_loss']), label='训练分类损失')
        plt.title(f'被试S{subject_id}的分类损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.subplot(2, 2, 3)
        plt.plot(range(len(history['train_physics_loss'])), smooth_curve(history['train_physics_loss']), label='训练物理损失')
        plt.title(f'被试S{subject_id}的物理损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.subplot(2, 2, 4)
        plt.plot(range(len(history['train_acc'])), smooth_curve(history['train_acc']), label='训练准确率')
        plt.plot(range(len(history['val_acc'])), smooth_curve(history['val_acc']), label='验证准确率')
        plt.title(f'被试S{subject_id}的准确率曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs('results', exist_ok=True)
        plt.savefig(f'results/S{subject_id}_fold{fold_num}_training_curves.png', dpi=150)
        plt.close()
        fold_num += 1
    # --- 9折交叉验证结束 ---
    
    # 打印交叉验证平均结果
    if all_fold_results:
        arr = np.array(all_fold_results)
        train_avg_acc = np.mean(arr[:,1])
        train_avg_f1 = np.mean(arr[:,2])
        val_avg_acc = np.mean(arr[:,3])
        val_avg_f1 = np.mean(arr[:,4])
        print("\n===== 被试 S{} 9折交叉验证平均结果 =====".format(subject_id))
        print("训练集平均准确率: {:.2f}%, F1: {:.4f}".format(train_avg_acc, train_avg_f1))
        print("验证集平均准确率: {:.2f}%, F1: {:.4f}".format(val_avg_acc, val_avg_f1))
        
    # --- 使用全部训练集重新训练，并在独立测试集上评估 ---
    print(f"\n===== 被试 S{subject_id} 最终模型在独立测试集上的评估 =====")
    # 全集训练阶段：将全部训练集划分为90%训练、10%验证
    sss_full = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    full_indices = np.arange(len(train_dataset))
    full_labels = np.array([train_dataset[i][1] for i in full_indices])
    train90_idx, val10_idx = next(sss_full.split(full_indices, full_labels))
    val10_dataset = Subset(train_dataset, val10_idx)
    
    # 对90%训练集做数据增强
    train90_full_dataset = None
    if repeat_factor > 1:
        print(f"全集训练阶段数据增强: 启用 (仅应用于90%训练集，将创建{repeat_factor-1}倍的增强数据)")
        # 此处复用上面定义的AugmentedDataset类
        train_set_original = AugmentedDataset(train_dataset, train90_idx, enhance=False)
        all_datasets = [train_set_original]
        for _ in range(repeat_factor - 1):
            augmented_set = AugmentedDataset(train_dataset, train90_idx, enhance=True)
            all_datasets.append(augmented_set)
        train90_full_dataset = ConcatDataset(all_datasets)
        print(f"全集训练集总大小(含增强): {len(train90_full_dataset)}，其中原始样本: {len(train_set_original)}，增强样本: {len(train90_full_dataset)-len(train_set_original)}")
    else:
        train90_full_dataset = Subset(train_dataset, train90_idx)
        print(f"全集训练阶段数据增强: 禁用 (使用原始数据)")
        
    # 重新初始化模型
    model = PINN_CNN_BCI(
        leadfield, 
        fwd,
        epochs.info,
        in_channels=dataset.data.shape[1], 
        seq_length=dataset.data.shape[2], 
        num_classes=2,
        dropout_rate=dropout_rate,
        use_pinn=use_pinn,
        sfreq=sfreq
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    # DataLoader
    train_loader = DataLoader(
        train90_full_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val10_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    patience = 15
    best_val_acc_final = 0.0
    early_stop_counter = 0
    best_epoch_final = 0
    best_model_state_final = None
    
    final_history = {'train_loss': [], 'val_loss': [], 'test_loss': [],
                     'train_acc': [], 'val_acc': [], 'test_acc': []}

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for eeg_data, labels in train_loader:
            eeg_data = eeg_data.to(device, dtype=torch.float32)
            labels = labels.flatten() - 1
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(eeg_data)
            loss_class = criterion(outputs['logits'], labels)
            
            # 恢复物理损失计算，保持与交叉验证一致
            if model.use_pinn and 'main_output' in outputs and outputs['main_output'] is not None:
                physical_target = torch.mean(eeg_data, dim=2)
                physical_target = F.normalize(physical_target, p=2, dim=1)
                physics_loss = model.physics_loss_calculator(outputs['main_output'])
                reconstruction_loss = nn.MSELoss()(outputs['physics_output'], physical_target)
                total_physics_loss = physics_loss + 0.1 * reconstruction_loss
                current_physics_weight = model.physics_loss_calculator.get_physics_weight()
                loss = current_physics_weight * total_physics_loss + (1 - current_physics_weight) * loss_class
            else:
                loss = loss_class
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs['logits'].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100.0 * correct / total
        final_history['train_loss'].append(train_loss / len(train_loader))
        final_history['train_acc'].append(train_acc)

        # 验证
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for eeg_data, labels in val_loader:
                eeg_data = eeg_data.to(device, dtype=torch.float32)
                labels = labels.flatten() - 1
                labels = labels.to(device)
                outputs = model(eeg_data)
                loss = criterion(outputs['logits'], labels)
                val_loss += loss.item()
                _, predicted = outputs['logits'].max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        val_acc = 100.0 * val_correct / val_total
        final_history['val_loss'].append(val_loss / len(val_loader))
        final_history['val_acc'].append(val_acc)
        
        if (epoch+1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(f"被试 S{subject_id} | 全集训练 Epoch {epoch+1}/{num_epochs} | 训练准确率: {train_acc:.2f}% | 验证准确率: {val_acc:.2f}%")
        
        if val_acc > best_val_acc_final:
            best_val_acc_final = val_acc
            best_epoch_final = epoch
            best_model_state_final = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        if early_stop_counter >= patience:
            print(f"全集训练早停: {patience}个epoch验证准确率未提升")
            break
            
    # 加载最佳模型并进行最终测试
    test_acc, test_f1, test_kappa = 0, 0, 0
    if best_model_state_final:
        print(f"加载全集训练最佳模型 (Epoch {best_epoch_final+1})")
        model.load_state_dict(best_model_state_final)
        
        model.eval()
        test_preds, test_truths = [], []
        with torch.no_grad():
            for eeg_data, labels in test_loader:
                eeg_data = eeg_data.to(device, dtype=torch.float32)
                labels = labels.flatten() - 1
                labels = labels.to(device)
                outputs = model(eeg_data)
                _, predicted = outputs['logits'].max(1)
                test_preds.extend(predicted.cpu().tolist())
                test_truths.extend(labels.cpu().tolist())
        test_acc = accuracy_score(test_truths, test_preds) * 100
        test_f1 = f1_score(test_truths, test_preds, average='macro')
        test_cm = confusion_matrix(test_truths, test_preds)
        test_kappa = cohen_kappa_score(test_truths, test_preds)
        print(f"最终测试集准确率: {test_acc:.2f}%, F1分数: {test_f1:.4f}, Kappa: {test_kappa:.4f}")
        print("最终测试集混淆矩阵:")
        print(test_cm)
        
    return all_fold_results, (test_acc, test_f1, test_kappa)


# 运行主程序 (修改以移除FBCSP相关参数)
if __name__ == '__main__':
    # 设置参数
    batch_size = 64
    num_epochs = 150  # 减少训练周期以加快训练速度
    repeat_factor = 3
    
    # 记录全局开始时间
    global_start_time = time.time()
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)
        
    # 强制垃圾回收
    gc.collect()
    torch.cuda.empty_cache()
    
    # BCI-IV-2a数据集的实际路径
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.path.abspath('.') 
        
    data_dir = os.path.join(script_dir, "data", "BCI2a")
    if not os.path.exists(data_dir):
       # 如果在脚本相对路径找不到，使用您提供的备用绝对路径
       data_dir = r"E:\pycharm\PINN\PINN\data\BCI2a"

    # 检查数据目录结构
    if os.path.exists(data_dir):
        print("=== 使用 PINN(空域) + 1D-CNN(时域) 混合模型进行训练 ===")
        
        available_subjects = []
        for i in range(1, 10):
            file_path = os.path.join(data_dir, f'S{i}.mat')
            if os.path.exists(file_path):
                available_subjects.append(i)
            
        print(f"数据扩充倍数: {repeat_factor}倍 ({'不使用增强' if repeat_factor == 1 else f'包含{repeat_factor-1}倍增强数据'})")
        print(f"训练周期: {num_epochs}")
        
        # 存储每个被试的结果
        subject_results = []
        testset_results = []
        for subject_id in available_subjects:
            print(f"\n=== 开始训练被试 S{subject_id} ===\n")
            # 注意：调用 train_model 时已移除 freq_band_type, fbcsp_components 参数
            results, testset_result = train_model(
                data_dir, 
                subject_id=subject_id, 
                num_epochs=num_epochs, 
                batch_size=batch_size, 
                repeat_factor=repeat_factor
            )
            if not results: continue

            arr = np.array(results)
            train_avg_acc = np.mean(arr[:,1])
            train_avg_f1 = np.mean(arr[:,2])
            val_avg_acc = np.mean(arr[:,3])
            val_avg_f1 = np.mean(arr[:,4])
            subject_results.append((subject_id, train_avg_acc, train_avg_f1, val_avg_acc, val_avg_f1))
            testset_results.append((subject_id, *testset_result))
        
        # 打印所有被试的结果汇总
        print("\n\n===== 所有被试结果汇总 (PINN + 1D-CNN 简化模型) =====")
        print("被试\t训练集准确率\t验证集准确率\t最终测试集准确率\t最终测试集F1\t最终测试集Kappa")
        if not subject_results:
             print("没有有效的训练结果。")
        else:
            avg_train_acc = np.mean([r[1] for r in subject_results])
            avg_val_acc = np.mean([r[3] for r in subject_results])
            avg_test_acc = np.mean([r[1] for r in testset_results])
            avg_test_f1 = np.mean([r[2] for r in testset_results])
            avg_test_kappa = np.mean([r[3] for r in testset_results])
            for i, subj_result in enumerate(subject_results):
                subj, train_acc, _, val_acc, _ = subj_result
                testset_acc, testset_f1, testset_kappa = testset_results[i][1:]
                print(f"S{subj}\t{train_acc:.2f}%\t\t{val_acc:.2f}%\t\t{testset_acc:.2f}%\t\t{testset_f1:.4f}\t\t{testset_kappa:.4f}")
            print("-------------------------------------------------------------------------------------------")
            print(f"平均\t{avg_train_acc:.2f}%\t\t{avg_val_acc:.2f}%\t\t{avg_test_acc:.2f}%\t\t{avg_test_f1:.4f}\t\t{avg_test_kappa:.4f}")
        
        # 计算全局总训练时间
        global_total_time = time.time() - global_start_time
        hours, remainder = divmod(global_total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        global_time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        print(f"\n===== 训练完成! 所有被试总训练时间: {global_time_str} =====")
            
    else:
        print(f"错误: BCI2a数据目录不存在: {data_dir}")
        print("请先下载BCI2a数据集并放置到正确位置")
        exit(1)