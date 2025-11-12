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
from collections import deque

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
    def __init__(self, input_dim, n_sources, leadfield, source_positions, device='cuda'):
        super(PINNSourceLocalization, self).__init__()
        self.input_dim = input_dim
        self.n_sources = n_sources
        self.device = device
        
        # 引线场矩阵 (n_channels, n_sources)
        self.leadfield = leadfield.to(device)
        
        # 源空间坐标 (n_sources, 3)，用于划分左右半球
        self.source_positions = source_positions.to(device)
        
        # 计算左右半球源点索引 (基于x坐标，x>0为左半球，x<0为右半球)
        self.left_hemisphere_idx = (source_positions[:, 0] > 0).nonzero(as_tuple=True)[0].to(device)
        self.right_hemisphere_idx = (source_positions[:, 0] < 0).nonzero(as_tuple=True)[0].to(device)
        
        # ==================================================================
        # 基于运动皮层人体映射图（Motor Homunculus）的精确功能区划分
        # ==================================================================
        
        # 1. 手部运动区（左右对侧控制）
        # 位置：中央沟附近，中等z坐标（0.04-0.07），左右偏移
        hand_motor_mask = (source_positions[:, 1] > -0.02) & (source_positions[:, 1] < 0.05) & \
                          (source_positions[:, 2] > 0.04) & (source_positions[:, 2] < 0.07)
        self.left_hand_motor_idx = ((source_positions[:, 0] > 0.02) & hand_motor_mask).nonzero(as_tuple=True)[0].to(device)
        self.right_hand_motor_idx = ((source_positions[:, 0] < -0.02) & hand_motor_mask).nonzero(as_tuple=True)[0].to(device)
        
        # 2. 脚部运动区（顶部旁中央小叶，双侧对称激活）
        # 位置：顶部中线，z坐标最高（>0.065），y坐标稍靠后
        foot_motor_mask = (source_positions[:, 2] > 0.065) & \
                          (source_positions[:, 1] > -0.03) & (source_positions[:, 1] < 0.03) & \
                          (torch.abs(source_positions[:, 0]) < 0.025)  # 中线±2.5cm
        self.foot_motor_idx = foot_motor_mask.nonzero(as_tuple=True)[0].to(device)
        
        # 3. 舌头运动区（下部运动皮层，双侧激活）
        # 位置：z坐标较低（0.03-0.05），y坐标中等，可能双侧
        tongue_motor_mask = (source_positions[:, 2] > 0.03) & (source_positions[:, 2] < 0.05) & \
                            (source_positions[:, 1] > -0.01) & (source_positions[:, 1] < 0.04) & \
                            (torch.abs(source_positions[:, 0]) < 0.04)  # 中线±4cm
        self.tongue_motor_idx = tongue_motor_mask.nonzero(as_tuple=True)[0].to(device)
        
        # 4. 辅助特征：整体左右半球（用于对比）
        self.left_hemisphere_strong_idx = (source_positions[:, 0] > 0.03).nonzero(as_tuple=True)[0].to(device)
        self.right_hemisphere_strong_idx = (source_positions[:, 0] < -0.03).nonzero(as_tuple=True)[0].to(device)
        
        print(f"\n=== 基于Motor Homunculus的源空间功能区划分 ===")
        print(f"左半球 {len(self.left_hemisphere_idx)} 个源点, 右半球 {len(self.right_hemisphere_idx)} 个源点")
        print(f"左手运动区: {len(self.left_hand_motor_idx)} 个源点 (对应右手想象)")
        print(f"右手运动区: {len(self.right_hand_motor_idx)} 个源点 (对应左手想象)")
        print(f"脚部运动区: {len(self.foot_motor_idx)} 个源点 (顶部中线)")
        print(f"舌头运动区: {len(self.tongue_motor_idx)} 个源点 (下部中线)")
        print(f"==========================================\n")
        
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
            dict: 包含源激活、半球特征和损失的字典
        """
        batch_size, n_channels, n_timepoints = x.shape
        
        # 时间维度平均，得到空间特征
        spatial_features = torch.mean(x, dim=2)  # (batch_size, n_channels)
        
        # 深度特征提取
        deep_features = self.feature_extractor(spatial_features)
        
        # 源定位映射
        source_activations = self.source_mapping(deep_features)
        
        # ==================================================================
        # 四分类专用源空间特征提取（基于Motor Homunculus）
        # ==================================================================
        
        # 1. 左手想象特征：右侧手部运动区激活强
        right_hand_motor_activation = torch.mean(torch.abs(source_activations[:, self.right_hand_motor_idx]), dim=1, keepdim=True)
        right_hand_motor_max = torch.max(torch.abs(source_activations[:, self.right_hand_motor_idx]), dim=1, keepdim=True)[0]
        
        # 2. 右手想象特征：左侧手部运动区激活强
        left_hand_motor_activation = torch.mean(torch.abs(source_activations[:, self.left_hand_motor_idx]), dim=1, keepdim=True)
        left_hand_motor_max = torch.max(torch.abs(source_activations[:, self.left_hand_motor_idx]), dim=1, keepdim=True)[0]
        
        # 3. 脚部想象特征：顶部中线区域激活
        foot_motor_activation = torch.mean(torch.abs(source_activations[:, self.foot_motor_idx]), dim=1, keepdim=True)
        foot_motor_max = torch.max(torch.abs(source_activations[:, self.foot_motor_idx]), dim=1, keepdim=True)[0] if len(self.foot_motor_idx) > 0 else torch.zeros_like(foot_motor_activation)
        
        # 4. 舌头想象特征：下部运动区激活
        tongue_motor_activation = torch.mean(torch.abs(source_activations[:, self.tongue_motor_idx]), dim=1, keepdim=True)
        tongue_motor_max = torch.max(torch.abs(source_activations[:, self.tongue_motor_idx]), dim=1, keepdim=True)[0] if len(self.tongue_motor_idx) > 0 else torch.zeros_like(tongue_motor_activation)
        
        # 5. 对侧控制指数（左右手判别的核心特征）
        # 左手倾向 = 右侧手部区 - 左侧手部区
        left_hand_laterality = right_hand_motor_activation - left_hand_motor_activation
        # 右手倾向 = 左侧手部区 - 右侧手部区
        right_hand_laterality = left_hand_motor_activation - right_hand_motor_activation
        
        # 6. 脚-手判别特征（脚部激活 vs 手部激活）
        hand_activation_total = left_hand_motor_activation + right_hand_motor_activation
        foot_vs_hand = foot_motor_activation / (hand_activation_total + 1e-8)
        
        # 7. 舌头-手判别特征（舌头激活 vs 手部激活）
        tongue_vs_hand = tongue_motor_activation / (hand_activation_total + 1e-8)
        
        # 8. 脚-舌判别特征
        foot_vs_tongue = foot_motor_activation / (tongue_motor_activation + 1e-8)
        
        # 9. 双侧对称性（脚和舌头的特征）
        left_hemisphere_strong = torch.mean(torch.abs(source_activations[:, self.left_hemisphere_strong_idx]), dim=1, keepdim=True)
        right_hemisphere_strong = torch.mean(torch.abs(source_activations[:, self.right_hemisphere_strong_idx]), dim=1, keepdim=True)
        bilateral_symmetry = torch.min(left_hemisphere_strong, right_hemisphere_strong) / (torch.max(left_hemisphere_strong, right_hemisphere_strong) + 1e-8)
        
        # 10. 垂直位置特征（脚在顶部，舌头在下部）
        # 使用最大激活值的比值来判断激活位置
        foot_dominance = foot_motor_max / (foot_motor_max + tongue_motor_max + hand_activation_total + 1e-8)
        tongue_dominance = tongue_motor_max / (foot_motor_max + tongue_motor_max + hand_activation_total + 1e-8)
        
        # 整合为20维四分类判别特征向量
        hemisphere_features = torch.cat([
            # 手部运动特征 (8维)
            left_hand_motor_activation,      # [1] 左手运动区平均激活
            left_hand_motor_max,              # [2] 左手运动区峰值激活
            right_hand_motor_activation,      # [3] 右手运动区平均激活
            right_hand_motor_max,             # [4] 右手运动区峰值激活
            left_hand_laterality,             # [5] 左手倾向指数（核心）
            right_hand_laterality,            # [6] 右手倾向指数（核心）
            hand_activation_total,            # [7] 手部总激活
            
            # 脚部运动特征 (4维)
            foot_motor_activation,            # [8] 脚部区平均激活
            foot_motor_max,                   # [9] 脚部区峰值激活
            foot_vs_hand,                     # [10] 脚-手判别（核心）
            foot_dominance,                   # [11] 脚部主导度
            
            # 舌头运动特征 (4维)
            tongue_motor_activation,          # [12] 舌头区平均激活
            tongue_motor_max,                 # [13] 舌头区峰值激活
            tongue_vs_hand,                   # [14] 舌-手判别（核心）
            tongue_dominance,                 # [15] 舌头主导度
            
            # 判别特征 (4维)
            foot_vs_tongue,                   # [16] 脚-舌判别
            bilateral_symmetry,               # [17] 双侧对称性
            left_hemisphere_strong,           # [18] 强左半球激活
            right_hemisphere_strong           # [19] 强右半球激活
        ], dim=1)
        
        # 计算物理约束损失
        physics_loss = self.physics_loss_calculator(source_activations, target_eeg)
        
        return {
            'source_activations': source_activations,
            'deep_features': deep_features,
            'physics_loss': physics_loss,
            'spatial_features': spatial_features,
            'hemisphere_features': hemisphere_features,
            'left_hand_motor_activation': left_hand_motor_activation,
            'right_hand_motor_activation': right_hand_motor_activation,
            'foot_motor_activation': foot_motor_activation,
            'tongue_motor_activation': tongue_motor_activation
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
# ATCNet注意力机制模块（方案H）
# =================================================================================
class SqueezeExcitation(nn.Module):
    """
    挤压激励注意力机制 (Squeeze-and-Excitation)
    """
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, time)
        Returns:
            attended: (batch, channels, time)
        """
        batch, channels, _ = x.size()
        # Squeeze: 全局平均池化
        y = self.squeeze(x).view(batch, channels)
        # Excitation: 学习通道权重
        y = self.excitation(y).view(batch, channels, 1)
        # Scale: 应用权重
        return x * y.expand_as(x)


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制 (Multi-Head Self-Attention)
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch, time, channels)
        Returns:
            attended: (batch, time, channels)
        """
        # 自注意力
        attn_output, _ = self.attention(x, x, x)
        # 残差连接 + 层归一化
        x = self.norm(x + self.dropout(attn_output))
        return x


# =================================================================================
# ATCNet时间卷积块（方案H）
# =================================================================================
class TemporalConvBlock(nn.Module):
    """
    ATCNet的时间卷积块 - 多尺度时间特征提取
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TemporalConvBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)
        
        # 残差连接
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ELU()
        
    def forward(self, x):
        """
        Args:
            x: (batch, channels, time)
        Returns:
            out: (batch, channels, time)
        """
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        if self.downsample:
            residual = self.downsample(residual)
        
        out = self.relu(out + residual)
        return out


# =================================================================================
# ATCNet核心架构（方案H）
# =================================================================================
class ATCNet(nn.Module):
    """
    Attention Temporal Convolutional Network for EEG
    基于论文: "Physics-Informed Attention Temporal Convolutional Network 
              for EEG-Based Motor Imagery Classification"
    """
    def __init__(self, in_channels=22, seq_length=1000, num_classes=4, 
                 F1=16, D=2, F2=32, dropout=0.3):
        super(ATCNet, self).__init__()
        
        # 阶段1: 时间卷积 - 提取时域特征
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1)
        )
        
        # 阶段2: 深度可分离卷积 - 提取空间特征
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (in_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )
        
        # 阶段3: 可分离卷积
        self.separable_conv = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )
        
        # 计算卷积后的时间维度
        temp_size = seq_length // 4 // 8  # 两次池化：/4, /8 = 31 (for seq_length=1000)
        
        # 阶段4: 挤压激励注意力
        self.se_block = SqueezeExcitation(F2, reduction=8)
        
        # 阶段5: 多头自注意力（在通道维度F2=32上做注意力，F2能被4整除）
        # 输入格式：(batch, time, F2)，其中F2=32是embed_dim
        self.mhsa = MultiHeadSelfAttention(embed_dim=F2, num_heads=4, dropout=dropout)
        
        # 阶段7: 时间卷积网络（多尺度）
        self.tcn_blocks = nn.ModuleList([
            TemporalConvBlock(F2, F2, kernel_size=3, dilation=1, dropout=dropout),
            TemporalConvBlock(F2, F2, kernel_size=3, dilation=2, dropout=dropout),
            TemporalConvBlock(F2, F2, kernel_size=3, dilation=4, dropout=dropout),
        ])
        
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # 输出维度
        self.output_dim = F2
        
    def forward(self, x):
        """
        Args:
            x: (batch, channels, time)
        Returns:
            features: (batch, output_dim)
        """
        # 添加通道维度: (batch, 1, channels, time)
        x = x.unsqueeze(1)
        
        # 时间卷积
        x = self.temporal_conv(x)
        
        # 空间卷积
        x = self.spatial_conv(x)
        
        # 可分离卷积
        x = self.separable_conv(x)
        
        # 移除空间维度: (batch, F2, 1, time) -> (batch, F2, time)
        x = x.squeeze(2)
        
        # SE注意力
        x = self.se_block(x)
        
        # 多头自注意力：在通道维度F2上建模时序依赖
        # 输入：(batch, F2, time) -> 转置为 (batch, time, F2)
        x_for_attn = x.permute(0, 2, 1)  # (batch, time=31, F2=32)
        
        # 多头自注意力（embed_dim=F2=32，能被num_heads=4整除）
        x_attn = self.mhsa(x_for_attn)  # (batch, time=31, F2=32)
        
        # 转置回: (batch, time, F2) -> (batch, F2, time)
        x = x_attn.permute(0, 2, 1)  # (batch, F2=32, time=31)
        
        # 多尺度时间卷积
        for tcn_block in self.tcn_blocks:
            x = tcn_block(x)
        
        # 全局平均池化
        x = self.gap(x).squeeze(-1)  # (batch, F2)
        
        return x


# =================================================================================
# ATCNet+PINN融合模型（方案H）
# =================================================================================
class PINN_ATCNet_BCI(nn.Module):
    """
    ATCNet+PINN融合模型用于BCI-IV-2a四分类任务
    - ATCNet分支：提取时空特征（时间卷积+空间卷积+注意力+TCN）
    - PINN分支：提取物理约束的源定位特征
    - 特征融合：自适应融合两种特征
    - 分类器：最终分类输出（左手/右手/脚/舌头）
    """
    def __init__(self, leadfield, fwd_model, epochs_info, in_channels=22, seq_length=1000, 
                 num_classes=4, dropout_rate=0.3, use_pinn=True, sfreq=250,
                 init_physics_weight=0.05, init_hemisphere_weight=0.4):
        super(PINN_ATCNet_BCI, self).__init__()
        
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
        
        # 方案H核心：ATCNet时空特征提取器
        self.atcnet = ATCNet(
            in_channels=in_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            F1=16,   # 时间滤波器数量
            D=2,     # 深度乘数
            F2=32,   # 输出滤波器数量
            dropout=dropout_rate
        ).to(device)
        
        # PINN空域特征提取分支
        if self.use_pinn:
            n_sources = leadfield.shape[1]
            
            # 从fwd_model提取源空间坐标
            source_positions = self._extract_source_positions(fwd_model)
            
            self.pinn_extractor = PINNSourceLocalization(
                input_dim=in_channels,
                n_sources=n_sources,
                leadfield=leadfield,
                source_positions=source_positions,
                device=device
            ).to(device)
            
            # 特征融合层（方案H）：ATCNet(32) + PINN源激活简化(20) + Motor Homunculus(19)
            # ATCNet: 32, PINN源激活: 20, Motor: 19 = 71维
            fusion_input_dim = self.atcnet.output_dim + 20 + 19
            
            # 源激活简化降维层（大幅压缩避免过拟合）
            self.source_activation_reducer = nn.Sequential(
                nn.Linear(n_sources, 64),
                nn.ELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 20),
                nn.ELU()
            ).to(device)
            
            # Motor Homunculus特征注意力
            self.feature_attention = nn.Sequential(
                nn.Linear(19, 19),
                nn.Sigmoid()
            ).to(device)
            
            # 四分类判别监督分支（轻量化）
            self.hemisphere_supervisor = nn.Sequential(
                nn.Linear(19, 16),
                nn.ELU(),
                nn.Dropout(dropout_rate * 0.6),
                nn.Linear(16, num_classes)
            ).to(device)
            
            # 自适应损失权重（降低物理损失权重）
            self.physics_weight_param = nn.Parameter(torch.tensor(init_physics_weight, dtype=torch.float32))
            self.hemisphere_weight_param = nn.Parameter(torch.tensor(init_hemisphere_weight, dtype=torch.float32))
        else:
            # 不使用PINN时：仅ATCNet
            fusion_input_dim = self.atcnet.output_dim
        
        # 轻量化特征融合网络（方案H：大幅简化，避免过拟合）
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(dropout_rate)
        ).to(device)
        
        # 分类器头
        self.classifier = nn.Linear(32, num_classes).to(device)
    
    def _extract_source_positions(self, fwd_model):
        """从前向模型中提取源空间坐标"""
        # 获取源空间坐标 (n_sources, 3)
        src = fwd_model['src']
        positions = []
        
        for src_hemi in src:
            # 提取每个半球的源点坐标（单位：米）
            rr = src_hemi['rr'][src_hemi['vertno']]
            positions.append(rr)
        
        # 合并左右半球的坐标
        all_positions = np.concatenate(positions, axis=0)
        
        # 转换为torch tensor
        positions_tensor = torch.tensor(all_positions, dtype=torch.float32)
        
        return positions_tensor
    
    def compute_contrastive_loss(self, hemisphere_features, labels, margin=1.5):
        """
        计算对比学习损失（方案E改进2）
        
        核心思想：
        - 左手样本应该：右手运动区激活高，左手运动区激活低
        - 右手样本应该：左手运动区激活高，右手运动区激活低
        - 脚部样本应该：脚部运动区激活高
        - 舌头样本应该：舌头运动区激活高
        
        Args:
            hemisphere_features: (batch_size, 19) 源空间特征
            labels: (batch_size,) 类别标签 0=左手, 1=右手, 2=脚, 3=舌头
            margin: 对比学习的间隔参数
        """
        # 提取关键特征（基于第245-319行的特征定义）
        left_hand_motor = hemisphere_features[:, 0:1]   # [1] 左手运动区平均激活
        right_hand_motor = hemisphere_features[:, 2:3]  # [3] 右手运动区平均激活
        left_laterality = hemisphere_features[:, 4:5]   # [5] 左手倾向指数
        right_laterality = hemisphere_features[:, 5:6]  # [6] 右手倾向指数
        foot_activation = hemisphere_features[:, 7:8]   # [8] 脚部区平均激活
        tongue_activation = hemisphere_features[:, 11:12] # [12] 舌头区平均激活
        
        contrastive_loss = torch.tensor(0.0, device=hemisphere_features.device)
        
        # 1. 左手vs右手对比（对侧控制原理）
        left_hand_mask = (labels == 0).float().unsqueeze(1)  # 左手样本
        right_hand_mask = (labels == 1).float().unsqueeze(1) # 右手样本
        
        if left_hand_mask.sum() > 0 and right_hand_mask.sum() > 0:
            # 左手样本应该有正的左手倾向指数
            left_hand_positive = torch.clamp(margin - left_laterality, min=0) * left_hand_mask
            # 右手样本应该有正的右手倾向指数
            right_hand_positive = torch.clamp(margin - right_laterality, min=0) * right_hand_mask
            contrastive_loss += (left_hand_positive.sum() + right_hand_positive.sum()) / (left_hand_mask.sum() + right_hand_mask.sum())
        
        # 2. 脚部vs手部对比
        foot_mask = (labels == 2).float().unsqueeze(1)
        hand_mask = ((labels == 0) | (labels == 1)).float().unsqueeze(1)
        
        if foot_mask.sum() > 0 and hand_mask.sum() > 0:
            # 脚部样本的脚部激活应该 > 手部样本的脚部激活
            foot_avg = (foot_activation * foot_mask).sum() / (foot_mask.sum() + 1e-8)
            hand_foot_avg = (foot_activation * hand_mask).sum() / (hand_mask.sum() + 1e-8)
            foot_contrast = torch.clamp(margin - (foot_avg - hand_foot_avg), min=0)
            contrastive_loss += foot_contrast
        
        # 3. 舌头vs其他对比
        tongue_mask = (labels == 3).float().unsqueeze(1)
        non_tongue_mask = (labels != 3).float().unsqueeze(1)
        
        if tongue_mask.sum() > 0 and non_tongue_mask.sum() > 0:
            # 舌头样本的舌头激活应该 > 非舌头样本的舌头激活
            tongue_avg = (tongue_activation * tongue_mask).sum() / (tongue_mask.sum() + 1e-8)
            non_tongue_avg = (tongue_activation * non_tongue_mask).sum() / (non_tongue_mask.sum() + 1e-8)
            tongue_contrast = torch.clamp(margin - (tongue_avg - non_tongue_avg), min=0)
            contrastive_loss += tongue_contrast
        
        return contrastive_loss
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入EEG数据 (batch_size, n_channels, n_timepoints)
        Returns:
            dict: 包含分类结果和中间特征的字典
        """
        batch_size = x.shape[0]
        
        # 方案H：ATCNet时空特征提取
        atcnet_features = self.atcnet(x)  # (batch, 32)
        
        # PINN空域特征提取
        pinn_outputs = None
        physics_loss = torch.tensor(0.0, device=x.device)
        
        if self.use_pinn:
            # 计算空间平均作为重建目标（弱化物理约束）
            target_eeg = torch.mean(x, dim=2)  # (batch_size, n_channels)
            target_eeg = F.normalize(target_eeg, p=2, dim=1)
            
            # PINN前向传播
            pinn_outputs = self.pinn_extractor(x, target_eeg)
            physics_loss = pinn_outputs['physics_loss']
            source_activations = pinn_outputs['source_activations']
            hemisphere_features = pinn_outputs['hemisphere_features']
            
            # 源激活降维（大幅压缩）
            source_activations_reduced = self.source_activation_reducer(source_activations)  # (batch, 20)
            
            # Motor Homunculus特征注意力加权
            attention_weights = self.feature_attention(hemisphere_features)
            hemisphere_features_attended = hemisphere_features * attention_weights  # (batch, 19)
            
            # 方案H特征融合：ATCNet(32) + 源激活简化(20) + Motor特征(19) = 71维
            combined_features = torch.cat([
                atcnet_features,                  # ATCNet时空特征 (32)
                source_activations_reduced,        # 源激活简化 (20)
                hemisphere_features_attended       # Motor Homunculus特征 (19)
            ], dim=1)
        else:
            combined_features = atcnet_features
        
        # 特征融合处理
        fused_features = self.feature_fusion(combined_features)
        
        # 分类预测
        logits = self.classifier(fused_features)
        
        # 半球监督预测（基于半球特征直接分类）
        hemisphere_logits = None
        if self.use_pinn and hasattr(self, 'hemisphere_supervisor'):
            hemisphere_logits = self.hemisphere_supervisor(hemisphere_features)
        
        # 返回结果
        result = {
            'logits': logits,
            'atcnet_features': atcnet_features,  # 方案H：ATCNet特征
            'fused_features': fused_features,
            'physics_loss': physics_loss,
            'hemisphere_logits': hemisphere_logits
        }
        
        if pinn_outputs is not None:
            result.update({
                'source_activations': pinn_outputs['source_activations'],
                'pinn_features': pinn_outputs['deep_features'],
                'spatial_features': pinn_outputs['spatial_features'],
                'hemisphere_features': pinn_outputs['hemisphere_features'],
                'left_hand_motor_activation': pinn_outputs['left_hand_motor_activation'],
                'right_hand_motor_activation': pinn_outputs['right_hand_motor_activation'],
                'foot_motor_activation': pinn_outputs['foot_motor_activation'],
                'tongue_motor_activation': pinn_outputs['tongue_motor_activation']
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
            
            # 四分类任务：保留所有4个类别（左手、右手、脚、舌头）
            print(f"四分类任务：保留所有类别数据")
            
            # 如果需要扩充数据集
            if self.repeat_factor > 1:
                self.expand_dataset()
            
            print(f"总共加载了 {len(self.labels)} 个样本，数据形状: {self.data.shape}")
        else:
            raise ValueError("没有成功加载任何数据。请检查数据集路径和格式。")
    
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
        """
        应用物理合理的数据增强策略（方案H：保守增强）
        
        恢复保守策略，仅保留不破坏物理约束的增强方法：
        - 小量高斯噪声：模拟测量噪声
        - 轻微幅度缩放：模拟个体差异（全局缩放）
        - 时间偏移：不改变空间关系
        
        去除的方法（可能破坏ATCNet的时空特征学习）：
        - 通道dropout：破坏空间结构
        - 通道扰动：破坏通道间相对关系
        - 强噪声：影响时间模式识别
        """
        
        # 1. 小量高斯噪声（保守）
        if np.random.rand() < 0.7:
            noise_level = np.random.uniform(0.003, 0.008)
            channel_noise_level = np.random.uniform(0.001, noise_level, (data.shape[0], 1))
            noise = np.random.normal(0, 1, data.shape) * channel_noise_level
            data = data + noise

        # 2. 轻微幅度缩放（全局，保持通道间相对关系）
        if np.random.rand() < 0.5:
            scale_range = (0.92, 1.08)
            global_scale = np.random.uniform(scale_range[0], scale_range[1])
            data = data * global_scale

        # 3. 时间偏移（适度）
        if np.random.rand() < 0.5:
            max_shift = 15
            shift = np.random.randint(-max_shift, max_shift + 1)
            if shift != 0:
                data_shifted = np.zeros_like(data)
                if shift > 0:
                    data_shifted[:, shift:] = data[:, :-shift]
                    data_shifted[:, :shift] = data[:, :1]
                else:
                    data_shifted[:, :shift] = data[:, -shift:]
                    data_shifted[:, shift:] = data[:, -1:]
                data = data_shifted

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
                dropout_rate=0.35, learning_rate=0.001, physics_weight=0.3, hemisphere_weight=0.3, use_pinn=True):
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
        model = PINN_ATCNet_BCI(
            leadfield=leadfield,
            fwd_model=fwd,
            epochs_info=epochs_info,
            in_channels=22,
            seq_length=1000,
            num_classes=4,  # 四分类：左手/右手/脚/舌头
            dropout_rate=dropout_rate,
            use_pinn=use_pinn,
            init_physics_weight=physics_weight,
            init_hemisphere_weight=hemisphere_weight,
            sfreq=250
        ).to(device)
        
        # 损失函数和优化器 - 平衡优化
        # 使用标签平滑的交叉熵损失
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
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
                
                # PINN物理损失、四分类判别损失和对比学习损失（方案E）
                if use_pinn:
                    physics_loss = outputs['physics_loss']
                    
                    # 四分类判别监督损失（基于源空间特征）
                    hemisphere_loss = torch.tensor(0.0, device=device)
                    if outputs['hemisphere_logits'] is not None:
                        hemisphere_loss = criterion(outputs['hemisphere_logits'], batch_labels)
                    
                    # 对比学习损失（方案E改进2）
                    contrastive_loss = torch.tensor(0.0, device=device)
                    if 'hemisphere_features' in outputs and outputs['hemisphere_features'] is not None:
                        contrastive_loss = model.compute_contrastive_loss(
                            outputs['hemisphere_features'], 
                            batch_labels, 
                            margin=1.5
                        )
                    
                    # 使用自适应权重（方案E改进3）
                    adaptive_physics_weight = torch.abs(model.physics_weight_param)
                    adaptive_hemisphere_weight = torch.abs(model.hemisphere_weight_param)
                    contrastive_weight = 0.2  # 对比损失权重
                    
                    # 总损失 = 分类 + 自适应物理 + 自适应判别 + 对比学习
                    total_loss = (class_loss + 
                                 adaptive_physics_weight * physics_loss + 
                                 adaptive_hemisphere_weight * hemisphere_loss +
                                 contrastive_weight * contrastive_loss)
                else:
                    physics_loss = torch.tensor(0.0, device=device)
                    hemisphere_loss = torch.tensor(0.0, device=device)
                    contrastive_loss = torch.tensor(0.0, device=device)
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
                        hemisphere_loss = torch.tensor(0.0, device=device)
                        if outputs['hemisphere_logits'] is not None:
                            hemisphere_loss = criterion(outputs['hemisphere_logits'], batch_labels)
                        
                        # 验证阶段也使用自适应权重
                        adaptive_physics_weight = torch.abs(model.physics_weight_param)
                        adaptive_hemisphere_weight = torch.abs(model.hemisphere_weight_param)
                        total_loss = (class_loss + 
                                     adaptive_physics_weight * physics_loss + 
                                     adaptive_hemisphere_weight * hemisphere_loss)
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
                    # 打印自适应权重（方案E）
                    curr_physics_w = torch.abs(model.physics_weight_param).item()
                    curr_hemisphere_w = torch.abs(model.hemisphere_weight_param).item()
                    print(f"轮次 {epoch+1}/{num_epochs}: 训练准确率 {train_acc:.2f}%, 验证准确率 {val_acc:.2f}%, "
                          f"PINN损失 {avg_physics_loss:.6f}, 权重[物理:{curr_physics_w:.3f}, 判别:{curr_hemisphere_w:.3f}]")
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
    final_model = PINN_ATCNet_BCI(
        leadfield=leadfield,
        fwd_model=fwd,
        epochs_info=epochs_info,
        in_channels=22,
        seq_length=1000,
        num_classes=4,  # 四分类：左手/右手/脚/舌头
        dropout_rate=dropout_rate,
        use_pinn=use_pinn,
        init_physics_weight=physics_weight,
        init_hemisphere_weight=hemisphere_weight,
        sfreq=250
    ).to(device)
    
    final_optimizer = optim.Adam(final_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 快速训练最终模型（提高轮次并保存快照用于集成）
    final_model.train()
    FINAL_EPOCHS = max(120, num_epochs)
    snapshot_states = deque(maxlen=5)
    for epoch in range(FINAL_EPOCHS):
        for batch_data, batch_labels in final_train_loader:
            batch_data = batch_data.to(device)
            batch_labels = (batch_labels - 1).long().to(device)
            
            final_optimizer.zero_grad()
            outputs = final_model(batch_data)
            class_loss = criterion(outputs['logits'], batch_labels)
            
            if use_pinn:
                physics_loss = outputs['physics_loss']
                hemisphere_loss = torch.tensor(0.0, device=device)
                if outputs['hemisphere_logits'] is not None:
                    hemisphere_loss = criterion(outputs['hemisphere_logits'], batch_labels)
                
                # 最终模型也使用自适应权重
                adaptive_physics_weight = torch.abs(final_model.physics_weight_param)
                adaptive_hemisphere_weight = torch.abs(final_model.hemisphere_weight_param)
                total_loss = (class_loss + 
                             adaptive_physics_weight * physics_loss + 
                             adaptive_hemisphere_weight * hemisphere_loss)
            else:
                physics_loss = torch.tensor(0.0, device=device)
                total_loss = class_loss
                
            total_loss.backward()
            final_optimizer.step()
        # 保存最后若干epoch的快照
        if epoch >= FINAL_EPOCHS - 5:
            snapshot_states.append(copy.deepcopy(final_model.state_dict()))
    
    # 最终测试
    final_model.eval()
    test_predictions = []
    test_true_labels = []
    
    # --- 快照集成 + 测试时增强(TTA) ---
    def tta_predict_logits(model, x, device, n_aug=8):
        model.eval()
        logits_sum = torch.zeros((x.size(0), model.num_classes), device=device)
        with torch.no_grad():
            # 原始
            out0 = model(x)['logits']
            logits_sum += out0
            # 增强
            for _ in range(n_aug):
                xa = x.clone()
                # 轻微高斯噪声
                noise_std = (0.002 + 0.004 * torch.rand(1, device=device)).item()
                xa = xa + noise_std * torch.randn_like(xa)
                # 全局轻微缩放
                scale = 0.95 + 0.10 * torch.rand((xa.size(0), 1, 1), device=device)
                xa = xa * scale
                # 小幅时间位移
                shift = int(torch.randint(-10, 11, (1,), device=device).item())
                xa = torch.roll(xa, shifts=shift, dims=2)
                logits_sum += model(xa)['logits']
        return logits_sum / (n_aug + 1)

    # 组装快照
    snapshots = []
    try:
        snapshots = list(snapshot_states)
    except Exception:
        snapshots = []

    # 若无快照，则使用当前最终模型
    if not snapshots:
        snapshots = [copy.deepcopy(final_model.state_dict())]

    with torch.no_grad():
        for batch_data, batch_labels in final_test_loader:
            batch_data = batch_data.to(device)
            batch_labels = (batch_labels - 1).long().to(device)

            # 集成多个快照 + TTA
            logits_ensemble = torch.zeros((batch_data.size(0), final_model.num_classes), device=device)
            for state in snapshots:
                final_model.load_state_dict(state)
                logits_ensemble += tta_predict_logits(final_model, batch_data, device, n_aug=8)
            logits_ensemble /= len(snapshots)
            predicted = torch.argmax(logits_ensemble, dim=1)

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
    # 训练参数 - 方案H：ATCNet+PINN融合（轻量化+有效化）
    BATCH_SIZE = 32          # 标准批次大小
    NUM_EPOCHS = 200         # 充分训练
    LEARNING_RATE = 0.001    # ATCNet标准学习率
    DROPOUT_RATE = 0.3       # 适度Dropout（ATCNet内部已有正则化）
    PHYSICS_WEIGHT = 0.1    # 大幅降低物理损失权重（辅助作用）
    HEMISPHERE_WEIGHT = 0.4  # 适度判别权重
    CONTRASTIVE_WEIGHT = 0.2 # 标准对比学习权重
    
    # 数据增强参数 - 方案H：物理合理增强（恢复保守策略）
    REPEAT_FACTOR = 3        # 适度数据扩充
    
    # 模型参数
    USE_PINN = True  # 是否使用PINN空域特征提取
    
    # 随机性控制
    USE_FIXED_SEED = False  # True: 使用固定种子(42)确保结果可复现, False: 使用随机种子
    FIXED_SEED = 42        # 固定种子值
    
    # 数据路径
    DATA_DIR = r"E:\pycharm\PINN\PINN\data\BCI2a"  # 请根据实际情况修改路径
    
    # ===================== 程序开始执行 =====================
    print("=== 使用 PINN(空域) + CNN(时域) 混合模型进行BCI-IV-2a四分类任务 ===")
    print("=== 四个类别：左手、右手、双脚、舌头运动想象 ===")
    print(f"训练参数配置:")
    print(f"  - 批次大小: {BATCH_SIZE}")
    print(f"  - 训练轮数: {NUM_EPOCHS}")
    print(f"  - 学习率: {LEARNING_RATE}")
    print(f"  - Dropout率: {DROPOUT_RATE}")
    print(f"  - 物理损失权重(初始): {PHYSICS_WEIGHT} (自适应调整)")
    print(f"  - 四分类判别权重(初始): {HEMISPHERE_WEIGHT} (自适应调整)")
    print(f"  - 对比学习损失权重: {CONTRASTIVE_WEIGHT}")
    if REPEAT_FACTOR == 1:
        print(f"  - 数据增强: 不使用增强")
    else:
        print(f"  - 数据增强: {REPEAT_FACTOR}倍（物理合理增强：小量噪声+轻微缩放+适度时移）")
    print(f"  - PINN模式: {'启用' if USE_PINN else '禁用'}")
    print(f"  - 数据路径: {DATA_DIR}")
    print(f"\n🚀 方案H: ATCNet + PINN 融合（轻量化+有效化）")
    print(f"   ✅ 【核心】ATCNet：时间卷积 + 深度可分离卷积 + SE注意力 + 多头自注意力 + 多尺度TCN")
    print(f"   ✅ 【辅助】PINN源定位：提供19维Motor Homunculus物理特征（降低权重至{PHYSICS_WEIGHT}）")
    print(f"   ✅ 【融合】轻量化：ATCNet(32) + 源激活简化(20) + Motor特征(19) = 71维")
    print(f"   ✅ 【正则】保守增强 + 适度Dropout({DROPOUT_RATE}) + 特征注意力 + 对比学习")
    print(f"   📊 特征路径：ATCNet时空特征提取 → PINN物理特征增强 → 轻量融合 → 分类")
    print(f"   📊 损失函数：分类(主) + 弱物理({PHYSICS_WEIGHT}) + 判别({HEMISPHERE_WEIGHT}) + 对比({CONTRASTIVE_WEIGHT})\n")
    
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
                hemisphere_weight=HEMISPHERE_WEIGHT,
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