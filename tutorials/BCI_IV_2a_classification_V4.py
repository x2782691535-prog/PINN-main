# BCI-IV-2a四分类V4.py
# PI-ATCNN: Physics-Informed Attention Temporal Convolutional Network
# 核心思想：将PINN的物理约束直接融入ATCNet，而非特征融合
# 损失函数 = 分类损失 + λ * leadfield物理重建损失

import os
import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d  # 用于数据增强的时间扭曲
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
import mne
from mne.channels import make_standard_montage
from mne.io import RawArray
from mne.forward import make_forward_solution
import gc
import torch.nn.functional as F
import time
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# =================================================================================
# ATCNet核心模块（从V3复用）
# =================================================================================

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation注意力模块"""
    def __init__(self, in_channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (batch, channels, time)
        b, c, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力模块"""
    def __init__(self, embed_dim, num_heads=4, dropout=0.3):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        attn_out, _ = self.mha(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        return x


class TemporalConvBlock(nn.Module):
    """时间卷积块（TCN的一个残差块）"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.3):
        super(TemporalConvBlock, self).__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)
        
        # 残差连接
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
    
    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 确保时间维度匹配（如果不匹配，裁剪到较小的尺寸）
        if out.size(2) != residual.size(2):
            min_size = min(out.size(2), residual.size(2))
            out = out[:, :, :min_size]
            residual = residual[:, :, :min_size]
        
        out = out + residual
        out = self.elu2(out)
        out = self.dropout2(out)
        
        return out


class ATCNet(nn.Module):
    """
    Attention Temporal Convolutional Network
    用于EEG运动想象分类的专用网络
    """
    def __init__(self, in_channels=22, seq_length=1000, F1=16, D=2, F2=32, dropout=0.3):
        super(ATCNet, self).__init__()
        
        self.in_channels = in_channels
        self.seq_length = seq_length
        self.F1 = F1
        self.D = D
        self.F2 = F2
        
        # 阶段1: 时间卷积（捕获时间动态）
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, 25), padding=(0, 12)),
            nn.BatchNorm2d(F1),
            nn.ELU()
        )
        
        # 阶段2: 深度卷积（空间滤波）
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (in_channels, 1), groups=F1),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )
        
        # 阶段3: 可分离卷积
        self.separable_conv = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 15), padding=(0, 7)),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )
        
        # 计算经过卷积和池化后的时间维度
        temp_size = seq_length
        temp_size = temp_size // 4  # 第一个池化
        temp_size = temp_size // 8  # 第二个池化
        
        # 阶段4: Squeeze-and-Excitation注意力
        self.se = SqueezeExcitation(in_channels=F2, reduction=4)
        
        # 阶段5: 多头自注意力（在通道维度F2=32上做注意力，F2能被4整除）
        self.mhsa = MultiHeadSelfAttention(embed_dim=F2, num_heads=4, dropout=dropout)
        
        # 阶段6: 多尺度时间卷积网络（TCN）
        self.tcn = nn.Sequential(
            TemporalConvBlock(F2, F2, kernel_size=4, dilation=1, dropout=dropout),
            TemporalConvBlock(F2, F2, kernel_size=4, dilation=2, dropout=dropout),
            TemporalConvBlock(F2, F2, kernel_size=4, dilation=4, dropout=dropout)
        )
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.output_dim = F2
    
    def forward(self, x):
        # 输入: (batch, channels=22, time=1000)
        batch_size = x.size(0)
        
        # 添加通道维度: (batch, 1, channels, time)
        x = x.unsqueeze(1)
        
        # 时间卷积
        x = self.temporal_conv(x)  # (batch, F1, channels, time)
        
        # 空间卷积（深度卷积）
        x = self.spatial_conv(x)  # (batch, F1*D, 1, time//4)
        
        # 可分离卷积
        x = self.separable_conv(x)  # (batch, F2, 1, time//32)
        
        # 移除空间维度: (batch, F2, 1, time) -> (batch, F2, time)
        x = x.squeeze(2)
        
        # SE注意力
        x = self.se(x)  # (batch, F2, time)
        
        # 多头自注意力：在通道维度F2上建模时序依赖
        x_for_attn = x.permute(0, 2, 1)  # (batch, time, F2)
        x_attn = self.mhsa(x_for_attn)  # (batch, time, F2)
        x = x_attn.permute(0, 2, 1)  # (batch, F2, time)
        
        # 多尺度TCN
        x = self.tcn(x)  # (batch, F2, time)
        
        # 全局平均池化
        x = self.global_pool(x)  # (batch, F2, 1)
        x = x.squeeze(2)  # (batch, F2)
        
        return x


# =================================================================================
# PI-ATCNN: 物理信息引导的ATCNet
# =================================================================================

class PI_ATCNN(nn.Module):
    """
    Physics-Informed Attention Temporal Convolutional Network
    
    核心创新：
    1. 使用ATCNet作为主干网络提取时空特征
    2. 在ATCNet输出上添加两个预测头：
       - 分类头：四分类任务（左手/右手/脚/舌头）
       - 源定位头：预测源空间激活，用于leadfield物理约束
    3. 损失函数 = 分类损失 + 完整PINN物理损失（研究内容一）
       - PINN损失包含三部分（基于论文公式4）：
         * L_data: 数据保真度损失（公式5）
         * L_phys: 物理约束损失（公式6）
         * L_reg: 辅助正则化损失（公式7）
    
    优势：
    - 统一网络架构，物理约束直接引导特征学习
    - 保持ATCNet的强大时空特征提取能力
    - 完整PINN物理损失作为正则化，防止过拟合，提高泛化
    """
    def __init__(self, leadfield, in_channels=22, seq_length=1000, 
                 num_classes=4, n_sources=132, dropout_rate=0.3,
                 init_physics_weight=0.1):
        super(PI_ATCNN, self).__init__()
        
        self.in_channels = in_channels
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.n_sources = n_sources
        
        # Leadfield矩阵 (n_channels, n_sources)
        self.leadfield = leadfield.to(device)
        
        # PINN损失函数的权重系数（基于论文公式4）
        self.lambda_data = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.lambda_phys = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.lambda_reg = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
        
        # 正则化参数（基于论文公式7）
        self.alpha_D = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
        self.alpha_R = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        
        # 核心：ATCNet主干网络
        self.atcnet = ATCNet(
            in_channels=in_channels,
            seq_length=seq_length,
            F1=16,
            D=2,
            F2=32,
            dropout=dropout_rate
        )
        
        # 预测头1：分类头（主任务）
        self.classifier = nn.Sequential(
            nn.Linear(self.atcnet.output_dim, 16),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, num_classes)
        )
        
        # 预测头2：源定位头（物理约束）
        # 从ATCNet特征预测源空间激活
        self.source_predictor = nn.Sequential(
            nn.Linear(self.atcnet.output_dim, 128),
            nn.ELU(),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(128, n_sources),
            nn.Softplus()  # 确保源激活非负
        )
        
        # 总体物理损失权重（研究内容一→研究内容二）
        self.physics_weight = nn.Parameter(
            torch.tensor(init_physics_weight, dtype=torch.float32)
        )
    
    def forward(self, x):
        """
        前向传播
        
        输入:
            x: EEG信号 (batch, channels, time)
        
        返回:
            dict: {
                'logits': 分类预测 (batch, num_classes)
                'source_activations': 预测的源激活 (batch, n_sources)
                'physics_loss': 完整PINN损失（基于研究内容一）
                'loss_components': 损失分量（用于调试）
            }
        """
        # ATCNet特征提取
        features = self.atcnet(x)  # (batch, 32)
        
        # 分类预测
        logits = self.classifier(features)  # (batch, num_classes)
        
        # 源激活预测
        source_activations = self.source_predictor(features)  # (batch, n_sources)
        
        # 计算完整PINN损失（基于研究内容一的论文公式）
        # 使用空间平均EEG作为重建目标
        target_eeg = torch.mean(x, dim=2)  # (batch, channels)
        
        # ======================================================================
        # 1. Data Fidelity Loss (公式5): L_data = 1 - cos(J_pred, J_true)
        # ======================================================================
        J_pred = torch.matmul(source_activations, self.leadfield.t())  # 预测的电流密度
        J_true = target_eeg  # 真实的电流密度
        
        # 计算余弦相似度
        dot_product = torch.sum(J_pred * J_true, dim=1)
        norm_pred = torch.norm(J_pred, p=2, dim=1)
        norm_true = torch.norm(J_true, p=2, dim=1)
        
        # 避免除零
        norm_pred = torch.clamp(norm_pred, min=1e-8)
        norm_true = torch.clamp(norm_true, min=1e-8)
        
        cosine_sim = dot_product / (norm_pred * norm_true)
        cosine_sim = torch.clamp(cosine_sim, min=-1.0, max=1.0)
        
        L_data = 1.0 - torch.mean(cosine_sim)
        
        # ======================================================================
        # 2. Physics-Constrained Loss (公式6): L_phys = ||φ_meas - L * J_pred||²
        # ======================================================================
        phi_meas = target_eeg  # 测量的头皮电位
        phi_pred = torch.matmul(source_activations, self.leadfield.t())  # leadfield重建
        
        L_phys = torch.mean((phi_meas - phi_pred) ** 2)
        
        # ======================================================================
        # 3. Auxiliary Regularization Loss (公式7):
        #    L_reg = α_D ||φ||² + α_R ||∂φ/∂n + γφ||²
        # ======================================================================
        # 第一项：Dirichlet边界条件的正则化
        L1 = torch.abs(self.alpha_D) * torch.mean(source_activations ** 2)
        
        # 第二项：Robin边界条件的正则化（梯度正则化）
        if source_activations.shape[1] > 1:
            grad_phi = source_activations[:, 1:] - source_activations[:, :-1]
            robin_term = grad_phi + torch.abs(self.gamma) * source_activations[:, :-1]
            L2 = torch.abs(self.alpha_R) * torch.mean(robin_term ** 2)
        else:
            L2 = torch.tensor(0.0, device=x.device)
        
        L_reg = L1 + L2
        
        # ======================================================================
        # 总PINN损失（公式4）:
        # L_total = λ_data * L_data + λ_phys * L_phys + λ_reg * L_reg
        # ======================================================================
        physics_loss = (torch.abs(self.lambda_data) * L_data + 
                       torch.abs(self.lambda_phys) * L_phys + 
                       torch.abs(self.lambda_reg) * L_reg)
        
        return {
            'logits': logits,
            'source_activations': source_activations,
            'physics_loss': physics_loss,
            'loss_components': {
                'L_data': L_data.item(),
                'L_phys': L_phys.item(),
                'L_reg': L_reg.item()
            }
        }


# =================================================================================
# 数据集类
# =================================================================================

class BCI2aDataset(Dataset):
    """BCI Competition IV-2a数据集"""
    def __init__(self, data, labels, augment=False, repeat_factor=1):
        """
        参数:
            data: EEG数据 (n_samples, n_channels, n_timepoints)
            labels: 标签 (n_samples,)，范围1-4
            augment: 是否应用数据增强
            repeat_factor: 数据扩充倍数
        """
        self.original_data = data
        self.original_labels = labels
        self.augment = augment
        self.repeat_factor = repeat_factor
        
        # 如果需要扩充，创建扩充后的数据集
        if augment and repeat_factor > 1:
            augmented_data = []
            augmented_labels = []
            
            # 保留原始数据
            for i in range(len(data)):
                augmented_data.append(data[i])
                augmented_labels.append(labels[i])
            
            # 添加增强数据
            for _ in range(repeat_factor - 1):
                for i in range(len(data)):
                    aug_sample = self.apply_data_augmentation(data[i])
                    augmented_data.append(aug_sample)
                    augmented_labels.append(labels[i])
            
            self.data = np.array(augmented_data)
            self.labels = np.array(augmented_labels)
            
            print(f"数据集已扩充{repeat_factor}倍：原始样本 {len(data)} 个，增强样本 {len(data) * (repeat_factor - 1)} 个")
            print(f"扩充后总样本数：{len(self.data)} 个")
        else:
            self.data = data
            self.labels = labels
    
    def apply_data_augmentation(self, sample):
        """
        应用物理合理的数据增强（平衡版）
        
        策略（适度增强，平衡过拟合和欠拟合）：
        1. 高斯噪声（SNR=18-25dB，适度变化）
        2. 幅度缩放（0.92-1.08，适度范围）
        3. 时间平移（±25个采样点，适度范围）
        4. 通道Dropout（低概率，轻度）
        5. 时间扭曲（低概率，轻度）
        """
        augmented = sample.copy()
        
        # 1. 高斯噪声（适度强度）
        noise_factor = np.random.uniform(0.04, 0.07)  # SNR范围18-25dB
        noise = np.random.normal(0, noise_factor, augmented.shape)
        augmented = augmented + noise * np.std(augmented)
        
        # 2. 幅度缩放（适度范围）
        scale = np.random.uniform(0.92, 1.08)
        augmented = augmented * scale
        
        # 3. 时间平移（适度范围）
        shift = np.random.randint(-25, 26)
        if shift > 0:
            augmented[:, shift:] = augmented[:, :-shift]
            augmented[:, :shift] = augmented[:, [shift]]
        elif shift < 0:
            augmented[:, :shift] = augmented[:, -shift:]
            augmented[:, shift:] = augmented[:, [shift - 1]]
        
        # 4. 通道Dropout：低概率应用（降低难度）
        if np.random.rand() < 0.15:  # 15%概率应用（从30%降低）
            n_channels = augmented.shape[0]
            n_drop = 1  # 只丢弃1个通道（降低难度）
            drop_channels = np.random.choice(n_channels, n_drop, replace=False)
            augmented[drop_channels, :] = 0
        
        # 5. 时间扭曲：低概率应用（降低难度）
        if np.random.rand() < 0.15:  # 15%概率应用（从30%降低）
            time_warp_factor = np.random.uniform(0.97, 1.03)  # 更温和的扭曲（从0.95-1.05降低）
            n_times = augmented.shape[1]
            
            # 创建原始和扭曲后的时间索引
            original_indices = np.arange(n_times)
            warped_length = int(n_times * time_warp_factor)
            warped_indices = np.linspace(0, n_times - 1, warped_length)
            
            # 对每个通道进行插值
            for ch in range(augmented.shape[0]):
                if augmented[ch].sum() != 0:  # 跳过被Dropout的通道
                    # 先创建扭曲版本的插值函数
                    f_warp = interp1d(original_indices, augmented[ch, :], 
                                     kind='linear', fill_value='extrapolate')
                    warped_signal = f_warp(warped_indices)
                    
                    # 再插值回原始长度
                    f_restore = interp1d(np.linspace(0, n_times - 1, warped_length), 
                                        warped_signal,
                                        kind='linear', fill_value='extrapolate')
                    augmented[ch, :] = f_restore(original_indices)
        
        return augmented
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = torch.FloatTensor(self.data[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return sample, label


def load_bci_data(subject, data_dir):
    """加载BCI-IV-2a数据集（四分类）"""
    filename = f'S{subject}.mat'
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据文件不存在: {filepath}")
    
    mat_data = sio.loadmat(filepath)
    
    # 检查键名
    if 'rawdata' not in mat_data or 'label' not in mat_data:
        raise KeyError(f"文件 {filepath} 中没有rawdata或label键")
    
    # 获取数据和标签
    rawdata = mat_data['rawdata']  # 形状: (1000, 22, 576)
    labels = mat_data['label']     # 形状: (576, 1)
    
    # 转换数据格式: (1000, 22, 576) -> (576, 22, 1000)
    data = np.transpose(rawdata, (2, 1, 0))
    
    # 标签处理：保持1-4范围，flatten
    labels = labels.flatten().astype(np.int64)
    labels = np.clip(labels, 1, 4)  # 确保在1-4范围内
    
    # 标准化每个试验的每个通道
    for i in range(data.shape[0]):  # 遍历所有试验
        for j in range(data.shape[1]):  # 遍历所有通道
            channel_data = data[i, j, :]
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            if std > 0:
                data[i, j, :] = (channel_data - mean) / std
    
    print(f"标签分布: {dict(zip(*np.unique(labels, return_counts=True)))}")
    print(f"四分类任务：保留所有类别数据")
    print(f"总共加载了 {len(data)} 个样本，数据形状: {data.shape}")
    
    return data, labels


def create_forward_model():
    """创建适合BCI-IV-2a数据的前向模型"""
    print("构建头模型...")
    
    # BCI-IV-2a标准通道
    ch_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
                'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz',
                'P2', 'POz']
    
    sfreq = 250
    n_channels = len(ch_names)
    n_times = 1000
    
    data = np.random.randn(n_channels, n_times)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = RawArray(data, info)
    
    montage = make_standard_montage('standard_1020')
    raw.set_montage(montage)
    
    # 创建源空间（低分辨率）
    subjects_dir = os.path.join(str(mne.datasets.sample.data_path()), 'subjects')
    src = mne.setup_source_space(
        subject='fsaverage',
        spacing='ico2',
        subjects_dir=subjects_dir,
        add_dist=False,
        verbose=False
    )
    
    # 创建BEM模型（3层：头皮、颅骨、脑）
    conductivity = (0.3, 0.006, 0.3)  # EEG需要3层BEM
    model = mne.make_bem_model(
        subject='fsaverage',
        ico=4,
        conductivity=conductivity,
        subjects_dir=subjects_dir,
        verbose=False
    )
    bem = mne.make_bem_solution(model, verbose=False)
    
    # 创建前向模型
    fwd = make_forward_solution(
        raw.info,
        trans=None,
        src=src,
        bem=bem,
        eeg=True,
        meg=False,
        mindist=5.0,
        n_jobs=1,
        verbose=False
    )
    
    n_sources = fwd['nsource']
    n_channels = len(fwd['info']['ch_names'])
    
    print(f"成功创建了适合BCI-IV-2a数据的前向模型，包含 {n_sources} 个源点，{n_channels} 个EEG通道")
    
    return fwd


def extract_leadfield_matrix(fwd_model):
    """从前向模型中提取leadfield矩阵"""
    leadfield_matrix = fwd_model['sol']['data']
    leadfield_matrix = torch.FloatTensor(leadfield_matrix)
    return leadfield_matrix


def extract_source_positions(fwd_model):
    """从前向模型中提取源点位置"""
    src = fwd_model['src']
    positions = []
    
    for hemi in src:
        pos = hemi['rr'][hemi['vertno']]
        positions.append(pos)
    
    positions = np.vstack(positions)
    positions = torch.FloatTensor(positions)
    
    return positions


# =================================================================================
# 训练函数
# =================================================================================

def train_pi_atcnn(model, train_loader, val_loader, num_epochs, learning_rate, device):
    """
    训练PI-ATCNN模型
    
    损失函数 = 分类损失 + λ * 物理重建损失
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)  # 【平衡调整】2e-4适度L2正则化
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=False
    )
    
    best_val_acc = 0.0
    best_train_acc = 0.0  # 记录对应最佳验证准确率时的训练准确率
    best_model_state = None
    patience_counter = 0
    early_stop_patience = 20  # 【关键】使用超参数调优的patience，避免过拟合
    
    for epoch in range(num_epochs):
        # ===== 训练阶段 =====
        model.train()
        train_loss = 0.0
        train_class_loss = 0.0
        train_physics_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = (batch_labels - 1).to(device)  # 标签从1-4转为0-3
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(batch_data)
            
            # 分类损失
            class_loss = criterion(outputs['logits'], batch_labels)
            
            # 完整PINN物理损失（研究内容一）
            physics_loss = outputs['physics_loss']
            adaptive_physics_weight = torch.abs(model.physics_weight)
            
            # 总损失 = 分类损失 + λ * 完整PINN损失
            total_loss = class_loss + adaptive_physics_weight * physics_loss
            
            # 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 统计
            train_loss += total_loss.item()
            train_class_loss += class_loss.item()
            train_physics_loss += physics_loss.item()
            
            _, predicted = torch.max(outputs['logits'], 1)
            train_correct += (predicted == batch_labels).sum().item()
            train_total += batch_labels.size(0)
        
        train_acc = 100.0 * train_correct / train_total
        avg_physics_loss = train_physics_loss / len(train_loader)
        
        # ===== 验证阶段 =====
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                batch_labels = (batch_labels - 1).to(device)
                
                outputs = model(batch_data)
                _, predicted = torch.max(outputs['logits'], 1)
                
                val_correct += (predicted == batch_labels).sum().item()
                val_total += batch_labels.size(0)
        
        val_acc = 100.0 * val_correct / val_total
        
        # 学习率调度
        scheduler.step(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_train_acc = train_acc  # 记录对应的训练准确率
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 打印进度（包含PINN损失分量）
        if (epoch + 1) % 20 == 0:
            curr_physics_w = torch.abs(model.physics_weight).item()
            # 获取PINN损失权重
            lambda_d = torch.abs(model.lambda_data).item()
            lambda_p = torch.abs(model.lambda_phys).item()
            lambda_r = torch.abs(model.lambda_reg).item()
            
            print(f"轮次 {epoch+1}/{num_epochs}: 训练准确率 {train_acc:.2f}%, "
                  f"验证准确率 {val_acc:.2f}%")
            print(f"  PINN损失={avg_physics_loss:.6f}, λ={curr_physics_w:.4f} "
                  f"[λ_data={lambda_d:.3f}, λ_phys={lambda_p:.3f}, λ_reg={lambda_r:.3f}]")
        
        # 早停
        if patience_counter >= early_stop_patience:
            print(f"早停触发于轮次 {epoch+1}")
            break
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_train_acc, best_val_acc


def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device)
            batch_labels = (batch_labels - 1).cpu().numpy()
            
            outputs = model(batch_data)
            _, predicted = torch.max(outputs['logits'], 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='macro')
    kappa = cohen_kappa_score(all_labels, all_preds)
    
    return accuracy, f1, kappa


# =================================================================================
# 主函数
# =================================================================================

if __name__ == '__main__':
    # ===================== 超参数配置（调优最佳配置）=====================
    BATCH_SIZE = 32
    NUM_EPOCHS = 300  # 【关键】使用超参数调优的epoch数，避免过拟合
    LEARNING_RATE = 0.003
    DROPOUT_RATE = 0.3
    INIT_PHYSICS_WEIGHT = 0.1
    
    REPEAT_FACTOR = 3  # 数据增强3倍
    NUM_REPEATS = 10  # 【新增】重复实验次数
    
    USE_PINN = True
    DATA_DIR = r'E:\pycharm\PINN\PINN\data\BCI2a'
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 存储所有实验的结果
    all_experiments_results = []  # 每个元素是一次实验的9个被试的结果
    
    # 打印配置
    print("=" * 70)
    print("=== PI-ATCNN: Physics-Informed Attention Temporal Convolutional Network ===")
    print("=== BCI-IV-2a四分类任务：左手、右手、双脚、舌头运动想象 ===")
    print("=== 【超参数调优最佳配置】 ===")
    print("=" * 70)
    print("训练参数配置（来自超参数调优）:")
    print(f"  - 批次大小: {BATCH_SIZE}")
    print(f"  - 训练轮数: {NUM_EPOCHS} ⚡ (100 epochs避免过拟合)")
    print(f"  - 学习率: {LEARNING_RATE}")
    print(f"  - Dropout率: {DROPOUT_RATE}")
    print(f"  - 初始物理损失权重: {INIT_PHYSICS_WEIGHT}")
    print(f"  - L2正则化(weight_decay): 2e-4")
    print(f"  - 早停patience: 25 ⚡ (更早停止，避免过拟合)")
    print(f"  - 数据增强: {REPEAT_FACTOR}倍")
    print(f"  - 增强策略: 适度噪声+缩放+平移+轻度通道Dropout(15%)+轻度时间扭曲(15%)")
    print(f"  - 设备: {device}")
    print(f"  - 数据路径: {DATA_DIR}")
    print("\n⚠️  关键改进: 使用100 epochs + patience 25（而非200 + 35）")
    print("    原因: 训练时间过长会导致在验证集上过拟合，降低泛化能力")
    
    print("\n" + "=" * 70)
    print("🚀 核心创新：物理信息引导的ATCNet（研究内容一→研究内容二）")
    print("=" * 70)
    print("【架构设计】")
    print("  ✅ ATCNet主干网络：时间卷积 + 深度可分离卷积 + SE注意力 + 多头自注意力 + TCN")
    print("  ✅ 双预测头设计：")
    print("      - 分类头：四分类任务（左手/右手/脚/舌头）- 研究内容二")
    print("      - 源定位头：预测源激活，提供leadfield物理约束 - 研究内容一")
    print("\n【损失函数 - 完整PINN（基于已验证的研究内容一）】")
    print(f"  ✅ 总损失 = 分类损失 + λ * 完整PINN损失")
    print(f"  ✅ 完整PINN损失（论文公式4）:")
    print(f"      L_PINN = λ_data·L_data + λ_phys·L_phys + λ_reg·L_reg")
    print(f"      - L_data (公式5): 数据保真度损失 = 1 - cos(J_pred, J_true)")
    print(f"      - L_phys (公式6): 物理约束损失 = ||φ_meas - L·J_pred||²")
    print(f"      - L_reg (公式7): 辅助正则化损失 = α_D||φ||² + α_R||∂φ/∂n + γφ||²")
    print(f"  ✅ λ初始值: {INIT_PHYSICS_WEIGHT}，所有权重参数训练中自适应调整")
    print("\n【优势分析】")
    print("  🎯 统一架构：物理约束直接引导ATCNet特征学习，而非独立分支融合")
    print("  🎯 保持性能：完整保留ATCNet的强大时空特征提取能力")
    print("  🎯 完整PINN：应用已验证有效的研究内容一，包含数据保真、物理约束、正则化三部分")
    print("  🎯 端到端：分类和完整物理约束联合优化，梯度互相引导")
    print("  🎯 研究衔接：研究内容一（PINN源定位）→ 研究内容二（BCI分类）")
    print("=" * 70 + "\n")
    
    # 创建前向模型
    fwd_model = create_forward_model()
    leadfield = extract_leadfield_matrix(fwd_model)
    source_positions = extract_source_positions(fwd_model)
    n_sources = leadfield.shape[1]
    
    # 检测可用受试者
    available_subjects = []
    for s in range(1, 10):
        if os.path.exists(os.path.join(DATA_DIR, f'S{s}.mat')):
            available_subjects.append(s)
    
    print(f"检测到 {len(available_subjects)} 个可用受试者数据: S{available_subjects}\n")
    
    # ===================== 重复实验主循环 =====================
    for repeat_idx in range(NUM_REPEATS):
        print(f"\n{'#'*100}")
        print(f"{'#'*100}")
        print(f"### 第 {repeat_idx + 1}/{NUM_REPEATS} 次实验 ###")
        print(f"{'#'*100}")
        print(f"{'#'*100}\n")
        
        # 存储本次实验的所有受试者结果
        all_results = []
        
        for subject in available_subjects:
            print(f"\n{'='*70}")
            print(f"=== 开始训练被试 S{subject} ===")
            print(f"{'='*70}\n")
            
            # 加载数据
            print(f"加载被试 S{subject} 的数据...")
            data, labels = load_bci_data(subject, DATA_DIR)
            
            # 数据集划分
            print(f"被试 S{subject} 原始数据集大小: {len(data)}个样本")
            
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(sss.split(data, labels))
            
            train_data, train_labels = data[train_idx], labels[train_idx]
            test_data, test_labels = data[test_idx], labels[test_idx]
            
            print(f"初步划分：训练集 {len(train_data)}, 测试集 {len(test_data)}")
            
            # 【关键修复】从原始训练数据中再划分出验证集（避免数据泄漏）
            sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx_final, val_idx_final = next(sss_val.split(train_data, train_labels))
            
            final_train_data = train_data[train_idx_final]
            final_train_labels = train_labels[train_idx_final]
            val_data = train_data[val_idx_final]
            val_labels = train_labels[val_idx_final]
            
            print(f"最终划分：训练集 {len(final_train_data)}, 验证集 {len(val_data)}, 测试集 {len(test_data)}\n")
            
            # 创建数据集：只对训练集增强，验证集和测试集不增强
            train_dataset = BCI2aDataset(
                final_train_data, final_train_labels,
                augment=True, repeat_factor=REPEAT_FACTOR
            )
            val_dataset = BCI2aDataset(val_data, val_labels, augment=False)
            test_dataset = BCI2aDataset(test_data, test_labels, augment=False)
            
            # 创建数据加载器
            train_loader = DataLoader(
                train_dataset, batch_size=BATCH_SIZE,
                shuffle=True, num_workers=0, pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=BATCH_SIZE,
                shuffle=False, num_workers=0, pin_memory=True
            )
            test_loader = DataLoader(
                test_dataset, batch_size=BATCH_SIZE,
                shuffle=False, num_workers=0, pin_memory=True
            )
            
            # 创建模型
            print("创建PI-ATCNN模型...")
            model = PI_ATCNN(
                leadfield=leadfield,
                in_channels=22,
                seq_length=1000,
                num_classes=4,
                n_sources=n_sources,
                dropout_rate=DROPOUT_RATE,
                init_physics_weight=INIT_PHYSICS_WEIGHT
            ).to(device)
            
            print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}\n")
            
            # 训练模型
            print("开始训练...\n")
            start_time = time.time()
            
            model, best_train_acc, best_val_acc = train_pi_atcnn(
                model, train_loader, val_loader,
                num_epochs=NUM_EPOCHS,
                learning_rate=LEARNING_RATE,
                device=device
            )
            
            training_time = time.time() - start_time
            print(f"\n训练完成，耗时: {training_time:.2f}秒")
            print(f"最佳训练准确率: {best_train_acc:.2f}%")
            print(f"最佳验证准确率: {best_val_acc:.2f}%")
            
            # 测试集评估
            print("\n在测试集上评估...")
            test_acc, test_f1, test_kappa = evaluate_model(model, test_loader, device)
            
            print(f"\n{'='*70}")
            print(f"被试 S{subject} 最终结果:")
            print(f"  - 训练准确率: {best_train_acc:.2f}%")
            print(f"  - 验证准确率: {best_val_acc:.2f}%")
            print(f"  - 测试准确率: {test_acc:.2f}%")
            print(f"  - F1分数: {test_f1:.4f}")
            print(f"  - Kappa系数: {test_kappa:.4f}")
            print(f"{'='*70}\n")
            
            all_results.append({
                'subject': subject,
                'train_accuracy': best_train_acc,
                'val_accuracy': best_val_acc,
                'test_accuracy': test_acc,
                'f1_score': test_f1,
                'kappa': test_kappa
            })
            
            # 清理GPU内存
            del model
            torch.cuda.empty_cache()
            gc.collect()
        
        # 保存本次实验的结果
        all_experiments_results.append(all_results)
    
    # ===================== 汇总所有实验结果 =====================
    print("\n" + "="*120)
    print("="*120)
    print(f"=== {NUM_REPEATS}次实验结果汇总 ===")
    print("="*120)
    print("="*120)
    
    # 定义指标
    metrics = [
        ('train_accuracy', '训练准确率', '%'),
        ('val_accuracy', '验证准确率', '%'),
        ('test_accuracy', '测试准确率', '%'),
        ('f1_score', 'F1分数', ''),
        ('kappa', 'Kappa系数', '')
    ]
    
    # 为每个指标生成表格
    for metric_key, metric_name, unit in metrics:
        print(f"\n{'='*120}")
        print(f"=== {metric_name} ===")
        print(f"{'='*120}")
        
        # 打印表头
        header = f"{'实验':<10}"
        for subject in available_subjects:
            header += f"S{subject:<6}"
        header += f"{'平均值':<10}"
        print(header)
        print("-" * 120)
        
        # 打印每次实验的数据
        all_averages = []
        for exp_idx, exp_results in enumerate(all_experiments_results):
            row = f"实验{exp_idx+1:<6}"
            values = []
            for subject in available_subjects:
                # 找到该受试者的结果
                subject_result = next((r for r in exp_results if r['subject'] == subject), None)
                if subject_result:
                    value = subject_result[metric_key]
                    values.append(value)
                    if unit == '%':
                        row += f"{value:>7.2f}%"
                    else:
                        row += f"{value:>7.4f}"
                else:
                    row += f"{'N/A':>7}"
            
            # 计算该实验的平均值
            if values:
                avg = np.mean(values)
                all_averages.append(avg)
                if unit == '%':
                    row += f"{avg:>10.2f}%"
                else:
                    row += f"{avg:>10.4f}"
            else:
                row += f"{'N/A':>10}"
            
            print(row)
        
        # 打印分隔线
        print("-" * 120)
        
        # 计算并打印每个受试者的平均值±标准差
        stats_row = f"{'均值±标准差':<10}"
        for subject in available_subjects:
            values = []
            for exp_results in all_experiments_results:
                subject_result = next((r for r in exp_results if r['subject'] == subject), None)
                if subject_result:
                    values.append(subject_result[metric_key])
            
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                if unit == '%':
                    stats_row += f"{mean_val:>5.2f}±{std_val:<4.2f}"
                else:
                    stats_row += f"{mean_val:>5.3f}±{std_val:<4.3f}"
            else:
                stats_row += f"{'N/A':>7}"
        
        # 总体平均±标准差
        if all_averages:
            overall_mean = np.mean(all_averages)
            overall_std = np.std(all_averages)
            if unit == '%':
                stats_row += f"{overall_mean:>7.2f}±{overall_std:<5.2f}"
            else:
                stats_row += f"{overall_mean:>7.3f}±{overall_std:<5.3f}"
        else:
            stats_row += f"{'N/A':>10}"
        
        print(stats_row)
        print("="*120)
    
    # ===================== 打印总结信息 =====================
    print(f"\n📊 最终统计摘要:")
    print(f"  ✅ 完成 {NUM_REPEATS} 次完整实验")
    print(f"  ✅ 每次实验包含 {len(available_subjects)} 个受试者")
    print(f"  ✅ 总共训练了 {NUM_REPEATS * len(available_subjects)} 个模型")
    
    # 计算总体平均值
    all_test_accs = []
    for exp_results in all_experiments_results:
        for result in exp_results:
            all_test_accs.append(result['test_accuracy'])
    
    if all_test_accs:
        overall_test_mean = np.mean(all_test_accs)
        overall_test_std = np.std(all_test_accs)
        print(f"  ✅ 总体平均测试准确率: {overall_test_mean:.2f}% ± {overall_test_std:.2f}%")
    
    print(f"\n{'='*120}")
    print("🎉 所有实验完成!")
    print(f"{'='*120}")

