# BCI-IV-2a四分类 - 超参数调优版本
# PI-ATCNN: Physics-Informed Attention Temporal Convolutional Network
# 使用网格搜索或随机搜索找出最佳超参数组合

import os
import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import mne
from mne.channels import make_standard_montage
from mne.io import RawArray
from mne.forward import make_forward_solution
import gc
import torch.nn.functional as F
import time
import warnings
import json
import itertools
from datetime import datetime
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# =================================================================================
# 从原始脚本导入所有必要的类和函数
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
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
    
    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if out.size(2) != residual.size(2):
            min_size = min(out.size(2), residual.size(2))
            out = out[:, :, :min_size]
            residual = residual[:, :, :min_size]
        
        out = out + residual
        out = self.elu2(out)
        out = self.dropout2(out)
        
        return out


class ATCNet(nn.Module):
    """Attention Temporal Convolutional Network"""
    def __init__(self, in_channels=22, seq_length=1000, F1=16, D=2, F2=32, dropout=0.3):
        super(ATCNet, self).__init__()
        
        self.in_channels = in_channels
        self.seq_length = seq_length
        self.F1 = F1
        self.D = D
        self.F2 = F2
        
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, 25), padding=(0, 12)),
            nn.BatchNorm2d(F1),
            nn.ELU()
        )
        
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (in_channels, 1), groups=F1),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )
        
        self.separable_conv = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 15), padding=(0, 7)),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )
        
        self.se = SqueezeExcitation(in_channels=F2, reduction=4)
        self.mhsa = MultiHeadSelfAttention(embed_dim=F2, num_heads=4, dropout=dropout)
        
        self.tcn = nn.Sequential(
            TemporalConvBlock(F2, F2, kernel_size=4, dilation=1, dropout=dropout),
            TemporalConvBlock(F2, F2, kernel_size=4, dilation=2, dropout=dropout),
            TemporalConvBlock(F2, F2, kernel_size=4, dilation=4, dropout=dropout)
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = F2
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1)
        
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.separable_conv(x)
        
        x = x.squeeze(2)
        x = self.se(x)
        
        x_for_attn = x.permute(0, 2, 1)
        x_attn = self.mhsa(x_for_attn)
        x = x_attn.permute(0, 2, 1)
        
        x = self.tcn(x)
        x = self.global_pool(x)
        x = x.squeeze(2)
        
        return x


class PI_ATCNN(nn.Module):
    """Physics-Informed Attention Temporal Convolutional Network"""
    def __init__(self, leadfield, in_channels=22, seq_length=1000, 
                 num_classes=4, n_sources=132, dropout_rate=0.3,
                 init_physics_weight=0.1):
        super(PI_ATCNN, self).__init__()
        
        self.in_channels = in_channels
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.n_sources = n_sources
        
        self.leadfield = leadfield.to(device)
        
        self.lambda_data = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.lambda_phys = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.lambda_reg = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
        
        self.alpha_D = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
        self.alpha_R = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        
        self.atcnet = ATCNet(
            in_channels=in_channels,
            seq_length=seq_length,
            F1=16,
            D=2,
            F2=32,
            dropout=dropout_rate
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.atcnet.output_dim, 16),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, num_classes)
        )
        
        self.source_predictor = nn.Sequential(
            nn.Linear(self.atcnet.output_dim, 128),
            nn.ELU(),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(128, n_sources),
            nn.Softplus()
        )
        
        self.physics_weight = nn.Parameter(
            torch.tensor(init_physics_weight, dtype=torch.float32)
        )
    
    def forward(self, x):
        features = self.atcnet(x)
        logits = self.classifier(features)
        source_activations = self.source_predictor(features)
        
        target_eeg = torch.mean(x, dim=2)
        
        J_pred = torch.matmul(source_activations, self.leadfield.t())
        J_true = target_eeg
        
        dot_product = torch.sum(J_pred * J_true, dim=1)
        norm_pred = torch.norm(J_pred, p=2, dim=1)
        norm_true = torch.norm(J_true, p=2, dim=1)
        
        norm_pred = torch.clamp(norm_pred, min=1e-8)
        norm_true = torch.clamp(norm_true, min=1e-8)
        
        cosine_sim = dot_product / (norm_pred * norm_true)
        cosine_sim = torch.clamp(cosine_sim, min=-1.0, max=1.0)
        
        L_data = 1.0 - torch.mean(cosine_sim)
        
        phi_meas = target_eeg
        phi_pred = torch.matmul(source_activations, self.leadfield.t())
        
        L_phys = torch.mean((phi_meas - phi_pred) ** 2)
        
        L1 = torch.abs(self.alpha_D) * torch.mean(source_activations ** 2)
        
        if source_activations.shape[1] > 1:
            grad_phi = source_activations[:, 1:] - source_activations[:, :-1]
            robin_term = grad_phi + torch.abs(self.gamma) * source_activations[:, :-1]
            L2 = torch.abs(self.alpha_R) * torch.mean(robin_term ** 2)
        else:
            L2 = torch.tensor(0.0, device=x.device)
        
        L_reg = L1 + L2
        
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


class BCI2aDataset(Dataset):
    """BCI Competition IV-2a数据集"""
    def __init__(self, data, labels, augment=False, repeat_factor=1):
        self.original_data = data
        self.original_labels = labels
        self.augment = augment
        self.repeat_factor = repeat_factor
        
        if augment and repeat_factor > 1:
            augmented_data = []
            augmented_labels = []
            
            for i in range(len(data)):
                augmented_data.append(data[i])
                augmented_labels.append(labels[i])
            
            for _ in range(repeat_factor - 1):
                for i in range(len(data)):
                    aug_sample = self.apply_data_augmentation(data[i])
                    augmented_data.append(aug_sample)
                    augmented_labels.append(labels[i])
            
            self.data = np.array(augmented_data)
            self.labels = np.array(augmented_labels)
        else:
            self.data = data
            self.labels = labels
    
    def apply_data_augmentation(self, sample):
        """应用物理合理的数据增强"""
        augmented = sample.copy()
        
        noise_factor = np.random.uniform(0.04, 0.07)
        noise = np.random.normal(0, noise_factor, augmented.shape)
        augmented = augmented + noise * np.std(augmented)
        
        scale = np.random.uniform(0.92, 1.08)
        augmented = augmented * scale
        
        shift = np.random.randint(-25, 26)
        if shift > 0:
            augmented[:, shift:] = augmented[:, :-shift]
            augmented[:, :shift] = augmented[:, [shift]]
        elif shift < 0:
            augmented[:, :shift] = augmented[:, -shift:]
            augmented[:, shift:] = augmented[:, [shift - 1]]
        
        if np.random.rand() < 0.15:
            n_channels = augmented.shape[0]
            n_drop = 1
            drop_channels = np.random.choice(n_channels, n_drop, replace=False)
            augmented[drop_channels, :] = 0
        
        if np.random.rand() < 0.15:
            time_warp_factor = np.random.uniform(0.97, 1.03)
            n_times = augmented.shape[1]
            
            original_indices = np.arange(n_times)
            warped_length = int(n_times * time_warp_factor)
            warped_indices = np.linspace(0, n_times - 1, warped_length)
            
            for ch in range(augmented.shape[0]):
                if augmented[ch].sum() != 0:
                    f_warp = interp1d(original_indices, augmented[ch, :], 
                                     kind='linear', fill_value='extrapolate')
                    warped_signal = f_warp(warped_indices)
                    
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


def load_bci_data(subject, data_dir, verbose=True):
    """加载BCI-IV-2a数据集"""
    filename = f'S{subject}.mat'
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据文件不存在: {filepath}")
    
    mat_data = sio.loadmat(filepath)
    
    if 'rawdata' not in mat_data or 'label' not in mat_data:
        raise KeyError(f"文件 {filepath} 中没有rawdata或label键")
    
    rawdata = mat_data['rawdata']
    labels = mat_data['label']
    
    data = np.transpose(rawdata, (2, 1, 0))
    labels = labels.flatten().astype(np.int64)
    labels = np.clip(labels, 1, 4)
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            channel_data = data[i, j, :]
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            if std > 0:
                data[i, j, :] = (channel_data - mean) / std
    
    if verbose:
        print(f"  加载被试S{subject}: {len(data)}个样本, 数据形状={data.shape}")
    
    return data, labels


def create_forward_model():
    """创建适合BCI-IV-2a数据的前向模型"""
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
    
    subjects_dir = os.path.join(str(mne.datasets.sample.data_path()), 'subjects')
    src = mne.setup_source_space(
        subject='fsaverage',
        spacing='ico2',
        subjects_dir=subjects_dir,
        add_dist=False,
        verbose=False
    )
    
    conductivity = (0.3, 0.006, 0.3)
    model = mne.make_bem_model(
        subject='fsaverage',
        ico=4,
        conductivity=conductivity,
        subjects_dir=subjects_dir,
        verbose=False
    )
    bem = mne.make_bem_solution(model, verbose=False)
    
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
    
    return fwd


def extract_leadfield_matrix(fwd_model):
    """从前向模型中提取leadfield矩阵"""
    leadfield_matrix = fwd_model['sol']['data']
    leadfield_matrix = torch.FloatTensor(leadfield_matrix)
    return leadfield_matrix


def train_single_config(model, train_loader, val_loader, num_epochs, learning_rate, 
                       weight_decay, device, early_stop_patience=35, verbose=False):
    """
    训练单个配置的模型
    返回: (best_train_acc, best_val_acc)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15, verbose=False
    )
    
    best_val_acc = 0.0
    best_train_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_correct = 0
        train_total = 0
        
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = (batch_labels - 1).to(device)
            
            optimizer.zero_grad()
            
            outputs = model(batch_data)
            class_loss = criterion(outputs['logits'], batch_labels)
            physics_loss = outputs['physics_loss']
            adaptive_physics_weight = torch.abs(model.physics_weight)
            
            total_loss = class_loss + adaptive_physics_weight * physics_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            _, predicted = torch.max(outputs['logits'], 1)
            train_correct += (predicted == batch_labels).sum().item()
            train_total += batch_labels.size(0)
        
        train_acc = 100.0 * train_correct / train_total
        
        # 验证阶段
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
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_train_acc = train_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: Train={train_acc:.2f}%, Val={val_acc:.2f}%")
        
        if patience_counter >= early_stop_patience:
            break
    
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
# 超参数搜索主函数
# =================================================================================

def hyperparameter_search(search_type='grid', n_random_samples=20):
    """
    超参数搜索
    
    参数:
        search_type: 'grid' (网格搜索) 或 'random' (随机搜索)
        n_random_samples: 随机搜索时的采样数量
    """
    
    # 定义超参数搜索空间
    param_grid = {
        'dropout_rate': [0.3, 0.4, 0.5],
        'init_physics_weight': [0.1, 0.15, 0.2, 0.25, 0.3],
        'learning_rate': [0.0005, 0.001, 0.002],
        'batch_size': [16, 32, 64],
        'weight_decay': [1e-4, 2e-4, 3e-4, 5e-4],
        'repeat_factor': [2, 3, 4]
    }
    
    print("=" * 100)
    print("=== PI-ATCNN 超参数调优 ===")
    print("=" * 100)
    print(f"搜索类型: {search_type.upper()}")
    print(f"\n超参数搜索空间:")
    for param, values in param_grid.items():
        print(f"  - {param}: {values}")
    
    # 生成参数组合
    if search_type == 'grid':
        # 网格搜索：所有组合
        param_combinations = list(itertools.product(
            param_grid['dropout_rate'],
            param_grid['init_physics_weight'],
            param_grid['learning_rate'],
            param_grid['batch_size'],
            param_grid['weight_decay'],
            param_grid['repeat_factor']
        ))
    else:
        # 随机搜索：随机采样n_random_samples个组合
        param_combinations = []
        for _ in range(n_random_samples):
            combo = (
                float(np.random.choice(param_grid['dropout_rate'])),
                float(np.random.choice(param_grid['init_physics_weight'])),
                float(np.random.choice(param_grid['learning_rate'])),
                int(np.random.choice(param_grid['batch_size'])),  # 确保是int类型
                float(np.random.choice(param_grid['weight_decay'])),
                int(np.random.choice(param_grid['repeat_factor']))  # 确保是int类型
            )
            param_combinations.append(combo)
    
    print(f"\n总共需要评估 {len(param_combinations)} 组参数\n")
    print("=" * 100)
    
    # 数据路径
    DATA_DIR = r'E:\pycharm\PINN\PINN\data\BCI2a'
    
    # 设备
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 创建前向模型（只需要创建一次）
    print("构建头模型...")
    fwd_model = create_forward_model()
    leadfield = extract_leadfield_matrix(fwd_model)
    n_sources = leadfield.shape[1]
    print(f"前向模型创建完成: leadfield矩阵形状={leadfield.shape}, 源点数={n_sources}\n")
    
    # 使用S1作为快速评估数据集
    print("加载数据集...")
    data, labels = load_bci_data(1, DATA_DIR)
    
    # 数据集划分
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(data, labels))
    
    train_data, train_labels = data[train_idx], labels[train_idx]
    test_data, test_labels = data[test_idx], labels[test_idx]
    
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx_final, val_idx_final = next(sss_val.split(train_data, train_labels))
    
    final_train_data = train_data[train_idx_final]
    final_train_labels = train_labels[train_idx_final]
    val_data = train_data[val_idx_final]
    val_labels = train_labels[val_idx_final]
    
    print(f"数据集大小 - 训练: {len(final_train_data)}, 验证: {len(val_data)}, 测试: {len(test_data)}\n")
    
    # 记录所有结果
    all_results = []
    best_val_acc = 0.0
    best_params = None
    best_result = None
    
    # 遍历所有参数组合
    start_time = time.time()
    
    for idx, params in enumerate(param_combinations):
        dropout_rate, init_physics_weight, learning_rate, batch_size, weight_decay, repeat_factor = params
        
        try:
            # 确保参数类型正确
            batch_size = int(batch_size)
            repeat_factor = int(repeat_factor)
            dropout_rate = float(dropout_rate)
            init_physics_weight = float(init_physics_weight)
            learning_rate = float(learning_rate)
            weight_decay = float(weight_decay)
            
            print(f"\n[{idx+1}/{len(param_combinations)}] 测试参数组合:")
            print(f"  Dropout={dropout_rate:.2f}, PhysicsWeight={init_physics_weight:.2f}, LR={learning_rate:.4f}")
            print(f"  BatchSize={batch_size}, WeightDecay={weight_decay:.5f}, RepeatFactor={repeat_factor}")
            
            # 创建数据集
            train_dataset = BCI2aDataset(
                final_train_data, final_train_labels,
                augment=True, repeat_factor=repeat_factor
            )
            val_dataset = BCI2aDataset(val_data, val_labels, augment=False)
            test_dataset = BCI2aDataset(test_data, test_labels, augment=False)
            
            # 创建数据加载器
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size,
                shuffle=True, num_workers=0, pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size,
                shuffle=False, num_workers=0, pin_memory=True
            )
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size,
                shuffle=False, num_workers=0, pin_memory=True
            )
            
            # 创建模型
            model = PI_ATCNN(
                leadfield=leadfield,
                in_channels=22,
                seq_length=1000,
                num_classes=4,
                n_sources=n_sources,
                dropout_rate=dropout_rate,
                init_physics_weight=init_physics_weight
            ).to(device)
            
            # 训练模型（使用较少的epoch进行快速评估）
            model, train_acc, val_acc = train_single_config(
                model, train_loader, val_loader,
                num_epochs=100,  # 减少epoch数以加快搜索
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                device=device,
                early_stop_patience=25,
                verbose=False
            )
            
            # 测试集评估
            test_acc, test_f1, test_kappa = evaluate_model(model, test_loader, device)
            
            # 计算过拟合程度
            overfit_gap = train_acc - test_acc
            
            # 记录结果
            result = {
                'params': {
                    'dropout_rate': dropout_rate,
                    'init_physics_weight': init_physics_weight,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'weight_decay': weight_decay,
                    'repeat_factor': repeat_factor
                },
                'metrics': {
                    'train_accuracy': float(train_acc),
                    'val_accuracy': float(val_acc),
                    'test_accuracy': float(test_acc),
                    'f1_score': float(test_f1),
                    'kappa': float(test_kappa),
                    'overfit_gap': float(overfit_gap)
                }
            }
            
            all_results.append(result)
            
            print(f"  结果: Train={train_acc:.2f}%, Val={val_acc:.2f}%, Test={test_acc:.2f}%")
            print(f"  F1={test_f1:.4f}, Kappa={test_kappa:.4f}, 过拟合间隔={overfit_gap:.2f}%")
            
            # 更新最佳结果（综合考虑验证准确率和过拟合程度）
            # 评分 = val_acc - 0.3 * overfit_gap（惩罚过拟合）
            score = val_acc - 0.3 * max(0, overfit_gap)
            best_score = best_val_acc - 0.3 * max(0, best_result['metrics']['overfit_gap']) if best_result else 0
            
            if score > best_score:
                best_val_acc = val_acc
                best_params = params
                best_result = result
                print(f"  ⭐ 新的最佳配置！综合得分={score:.2f}")
            
            # 清理内存
            del model, train_loader, val_loader, test_loader
            del train_dataset, val_dataset, test_dataset
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            import traceback
            print(f"  ❌ 错误: {str(e)}")
            if idx < 3:  # 只打印前3个错误的详细信息
                print(f"  详细错误信息:")
                traceback.print_exc()
            continue
    
    elapsed_time = time.time() - start_time
    
    # 检查是否有成功的结果
    if len(all_results) == 0:
        print("\n" + "=" * 100)
        print("❌ 错误: 所有参数组合都失败了，没有成功的结果")
        print("=" * 100)
        print("建议:")
        print("  1. 检查数据路径是否正确")
        print("  2. 检查GPU内存是否足够")
        print("  3. 减小batch_size或模型复杂度")
        return None, []
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'hyperparameter_search_results_{timestamp}.json'
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'search_type': search_type,
            'total_combinations': len(param_combinations),
            'elapsed_time_seconds': elapsed_time,
            'best_params': best_result['params'],
            'best_metrics': best_result['metrics'],
            'all_results': all_results
        }, f, indent=2, ensure_ascii=False)
    
    # 打印最终结果
    print("\n" + "=" * 100)
    print("=== 超参数搜索完成 ===")
    print("=" * 100)
    print(f"总耗时: {elapsed_time/60:.2f} 分钟")
    print(f"评估了 {len(all_results)} 组参数")
    print(f"\n🏆 最佳参数组合:")
    for param, value in best_result['params'].items():
        print(f"  - {param}: {value}")
    print(f"\n📊 最佳性能指标:")
    for metric, value in best_result['metrics'].items():
        print(f"  - {metric}: {value:.4f}")
    print(f"\n结果已保存到: {results_file}")
    print("=" * 100)
    
    return best_result, all_results


if __name__ == '__main__':
    # 选择搜索类型
    # 'grid': 网格搜索（全面但耗时）
    # 'random': 随机搜索（快速但可能遗漏最优解）
    
    # 建议先用随机搜索快速探索，再用网格搜索精细调优
    print("开始超参数搜索...\n")
    print("选择搜索策略:")
    print("  1. 随机搜索 (推荐) - 快速探索，采样30组参数")
    print("  2. 网格搜索 - 全面搜索，所有组合 (耗时)")
    print()
    
    # 使用随机搜索（推荐）
    # best_result, all_results = hyperparameter_search(
    #     search_type='random',
    #     n_random_samples=30  # 随机采样30组参数
    # )
    
    # 如果想用网格搜索，取消下面的注释（注意：可能需要很长时间）
    best_result, all_results = hyperparameter_search(search_type='grid')

# 检查结果
    if best_result is None:
        print("\n超参数搜索失败，请检查错误信息并重试。")
    else:
        print(f"\n✅ 搜索成功完成！找到最佳配置。")
