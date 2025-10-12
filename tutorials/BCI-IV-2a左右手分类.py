# 5简化


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

# FBCSP特征提取相关类
class CSP:
    """
    公共空间模式(Common Spatial Pattern)实现
    用于提取空间特征，增强不同类别运动想象之间的可区分性
    """
    def __init__(self, n_components=4):
        """
        初始化CSP过滤器
        
        参数:
            n_components: 每个类别要保留的空间特征数量（总特征数为n_components*2）
        """
        self.n_components = n_components
        self.filters_ = None
        self.patterns_ = None
        self.mean_ = None
        self.classes_ = None
        
    def fit(self, X, y):
        """
        使用训练数据拟合CSP过滤器
        
        参数:
            X: 训练数据，形状为(n_trials, n_channels, n_samples)
            y: 训练标签，形状为(n_trials,)
            
        返回:
            self: CSP对象
        """
        X = np.asarray(X)
        
        # 获取数据集中的类别
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        if n_classes < 2:
            raise ValueError("至少需要两个类别来训练CSP")
        
        # 计算每个类别的协方差矩阵的平均值
        covs = []
        for class_idx in self.classes_:
            # 获取当前类别的试验
            class_trials = X[y == class_idx]
            # 标准化数据
            class_trials = self._normalize(class_trials)
            # 计算协方差矩阵
            class_cov = np.zeros((class_trials.shape[1], class_trials.shape[1]))
            
            for trial in class_trials:
                # 计算单个试验的协方差矩阵
                trial_cov = np.cov(trial)
                # 累加协方差矩阵
                class_cov += trial_cov / np.trace(trial_cov)
            
            # 计算平均协方差矩阵
            class_cov /= len(class_trials)
            covs.append(class_cov)
        
        # 解决广义特征值问题以找到CSP过滤器
        # 对于二分类问题，我们可以使用一个类别的协方差矩阵除以所有协方差矩阵的和
        if n_classes == 2:
            cov_1 = covs[0]
            cov_2 = covs[1]
            
            # 解决广义特征值问题
            evals, evecs = eigh(cov_1, cov_1 + cov_2)
            
            # 对特征值和特征向量进行排序
            idx = np.argsort(evals)
            # 取最大和最小的n_components个特征向量
            idx = np.concatenate([idx[:self.n_components], idx[-self.n_components:]])
            
            # 提取过滤器
            self.filters_ = evecs[:, idx].T
            
            # 计算空间模式（滤波器的逆）
            self.patterns_ = np.linalg.pinv(self.filters_).T
            
        else:
            # 对于多类别问题，使用一对多的方法
            # 这里实现简化版本，只考虑每个类别与其余所有类别的对比
            n_channels = X.shape[1]
            self.filters_ = np.zeros((n_classes * self.n_components * 2, n_channels))
            self.patterns_ = np.zeros((n_classes * self.n_components * 2, n_channels))
            
            for i, class_idx in enumerate(self.classes_):
                # 当前类别的协方差矩阵
                cov_current = covs[i]
                # 其他类别的协方差矩阵总和
                cov_rest = sum(covs[:i] + covs[i+1:])
                
                # 解决广义特征值问题
                evals, evecs = eigh(cov_current, cov_current + cov_rest)
                
                # 对特征值和特征向量进行排序
                idx = np.argsort(evals)
                # 取最大和最小的n_components个特征向量
                idx = np.concatenate([idx[:self.n_components], idx[-self.n_components:]])
                
                # 提取过滤器
                filters = evecs[:, idx].T
                # 保存该类别的过滤器
                start_idx = i * 2 * self.n_components
                end_idx = (i + 1) * 2 * self.n_components
                self.filters_[start_idx:end_idx] = filters
                
                # 计算空间模式
                self.patterns_[start_idx:end_idx] = np.linalg.pinv(filters).T
        
        return self
    
    def transform(self, X):
        """
        应用CSP过滤器提取特征
        
        参数:
            X: 数据，形状为(n_trials, n_channels, n_samples)
            
        返回:
            特征，形状为(n_trials, n_components*2)
        """
        X = np.asarray(X)
        
        # 标准化数据
        X = self._normalize(X)
        
        n_trials, n_channels, n_samples = X.shape
        
        # 应用CSP过滤器
        X_filtered = np.dot(self.filters_, X.reshape(n_trials, n_channels, n_samples))
        
        # 计算对数方差特征
        X_features = np.log(np.mean(X_filtered ** 2, axis=2))
        
        return X_features
    
    def _normalize(self, X):
        """
        对数据进行标准化
        
        参数:
            X: 数据，形状为(n_trials, n_channels, n_samples)
            
        返回:
            标准化后的数据
        """
        X = np.asarray(X)
        
        # 检查数据形状
        if X.ndim != 3:
            raise ValueError("输入数据必须是3维的(n_trials, n_channels, n_samples)")
        
        # 对每个试验进行标准化
        for i in range(X.shape[0]):
            X_trial = X[i]
            X[i] = X_trial - np.mean(X_trial, axis=1, keepdims=True)
        
        return X


class FBCSP:
    """
    滤波器组公共空间模式(Filter Bank Common Spatial Pattern)
    将EEG信号分解为多个频带，在每个频带上应用CSP，然后选择最具区分性的特征
    """
    def __init__(self, freqs=None, n_components=4, n_features=4, sfreq=250):
        """
        初始化FBCSP
        
        参数:
            freqs: 频带列表，每个元素为(fmin, fmax)元组
            n_components: 每个类别在每个频带要保留的CSP组件数量
            n_features: 最终使用的特征数量（基于互信息选择）
            sfreq: 采样频率
        """
        # 默认频带为包含μ和β节律的常用频带
        self.freqs = freqs if freqs is not None else [
            (4, 8),    # θ带
            (8, 12),   # μ带(低)
            (12, 16),  # μ带(高)/β带(低)
            (16, 20),  # β带(中)
            (20, 24),  # β带(高)
            (24, 28),  # β带(高)
            (28, 32),  # β带(高)
        ]
        self.n_components = n_components
        self.n_features = n_features
        self.sfreq = sfreq
        self.filters = []  # 每个频带的CSP过滤器
        self.selected_features = None  # 基于互信息选择的特征索引
        
    def fit(self, X, y):
        """
        使用训练数据拟合FBCSP
        
        参数:
            X: 训练数据，形状为(n_trials, n_channels, n_samples)
            y: 训练标签，形状为(n_trials,)
            
        返回:
            self: FBCSP对象
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # 对每个频带应用滤波器并训练CSP
        for fmin, fmax in self.freqs:
            # 应用带通滤波器
            X_filtered = self._bandpass_filter(X, fmin, fmax)
            
            # 创建并训练CSP
            csp = CSP(n_components=self.n_components)
            csp.fit(X_filtered, y)
            
            self.filters.append(csp)
        
        # 提取所有频带的特征
        features = self.transform(X, select_features=False)
        
        # 使用互信息选择最具区分性的特征
        mi_scores = mutual_info_classif(features, y)
        
        # 选择得分最高的特征
        self.selected_features = np.argsort(mi_scores)[-self.n_features:]
        
        return self
    
    def transform(self, X, select_features=True):
        """
        应用FBCSP提取特征
        
        参数:
            X: 数据，形状为(n_trials, n_channels, n_samples)
            select_features: 是否只返回选定的特征
            
        返回:
            特征矩阵
        """
        X = np.asarray(X)
        n_trials = X.shape[0]
        
        # 存储所有频带的特征
        all_features = []
        
        # 对每个频带进行处理
        for i, (fmin, fmax) in enumerate(self.freqs):
            # 应用带通滤波器
            X_filtered = self._bandpass_filter(X, fmin, fmax)
            
            # 应用CSP并提取特征
            features = self.filters[i].transform(X_filtered)
            
            all_features.append(features)
        
        # 将所有频带的特征合并
        all_features = np.hstack(all_features)
        
        # 如果需要进行特征选择，只返回选定的特征
        if select_features and self.selected_features is not None:
            return all_features[:, self.selected_features]
        
        return all_features
    
    def _bandpass_filter(self, X, fmin, fmax):
        """
        应用带通滤波器
        
        参数:
            X: 数据，形状为(n_trials, n_channels, n_samples)
            fmin: 最低频率
            fmax: 最高频率
            
        返回:
            滤波后的数据
        """
        # 设计带通滤波器
        nyq = 0.5 * self.sfreq
        low = fmin / nyq
        high = fmax / nyq
        
        # 创建一个5阶的Butterworth带通滤波器
        b, a = signal.butter(5, [low, high], btype='band')
        
        # 应用滤波器到每个试验和通道
        X_filtered = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X_filtered[i, j] = signal.filtfilt(b, a, X[i, j])
        
        return X_filtered


class FBCSP_FeatureExtractor(nn.Module):
    """
    基于FBCSP的特征提取模块，可以与神经网络集成
    简化版本只使用3个核心频带
    """
    def __init__(self, n_channels=22, sfreq=250, n_components=4, n_features=None):
        """
        初始化FBCSP特征提取器
        
        参数:
            n_channels: EEG通道数
            sfreq: 采样频率
            n_components: 每个CSP的组件数量
            n_features: 选择的特征数量，默认为总特征数
        """
        super(FBCSP_FeatureExtractor, self).__init__()
        
        # 使用3个关键频带覆盖μ和β节律
        self.freqs = [
            (8, 12),   # μ带
            (12, 20),  # β带(低/中)
            (20, 30)   # β带(高)
        ]
        
        self.n_bands = len(self.freqs)
        self.n_channels = n_channels
        self.sfreq = sfreq
        self.n_components = n_components
        
        # 每个频带的CSP产生2*n_components个特征
        self.features_per_band = 2 * n_components
        self.n_fbcsp_features = self.n_bands * self.features_per_band
        
        # 如果没有指定特征数量，使用所有特征
        if n_features is None:
            self.n_features = self.n_fbcsp_features
        else:
            self.n_features = min(n_features, self.n_fbcsp_features)
        
        # 特征转换层
        self.feature_transform = nn.Sequential(
            nn.BatchNorm1d(self.n_features),
            nn.LeakyReLU(0.1)  # 替代nn.ReLU()
        )
    
    def forward(self, x):
        """
        前向传播，提取FBCSP特征
        
        参数:
            x: 输入数据，形状为(batch_size, n_channels, n_samples)
            
        返回:
            提取的特征，形状为(batch_size, n_features)
        """
        batch_size, n_channels, n_samples = x.shape
        device = x.device
        
        # 转移到CPU进行信号处理
        x_cpu = x.cpu().numpy()
        
        # 存储每个频带的CSP特征
        band_features = []
        
        # 对每个频带进行处理
        for fmin, fmax in self.freqs:
            # 应用带通滤波器
            x_filtered = self._bandpass_filter(x_cpu, fmin, fmax)
            
            # 计算空间协方差矩阵并提取特征
            covs = self._compute_covariance(x_filtered)
            
            # 对数处理协方差特征
            log_covs = np.log(np.maximum(covs, 1e-10))
            
            # 添加到特征列表
            band_features.append(log_covs)
        
        # 将所有频带的特征连接起来
        # 形状: (batch_size, n_bands * features_per_band)
        all_features = np.concatenate(band_features, axis=1)
        
        # 转换回PyTorch张量
        all_features = torch.from_numpy(all_features).float().to(device)
        
        # 应用特征变换
        output_features = self.feature_transform(all_features)
        
        return output_features
    
    def _bandpass_filter(self, X, fmin, fmax):
        """
        应用带通滤波器
        
        参数:
            X: 数据，形状为(batch_size, n_channels, n_samples)
            fmin: 最低频率
            fmax: 最高频率
            
        返回:
            滤波后的数据
        """
        # 设计带通滤波器
        nyq = 0.5 * self.sfreq
        low = fmin / nyq
        high = fmax / nyq
        
        # 创建一个5阶的Butterworth带通滤波器
        b, a = signal.butter(5, [low, high], btype='band')
        
        # 应用滤波器到每个试验和通道
        X_filtered = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X_filtered[i, j] = signal.filtfilt(b, a, X[i, j])
        
        return X_filtered
    
    def _compute_covariance(self, X):
        """
        计算空间协方差矩阵并提取特征
        
        参数:
            X: 数据，形状为(batch_size, n_channels, n_samples)
            
        返回:
            协方差特征，形状为(batch_size, n_components*2)
        """
        batch_size = X.shape[0]
        
        # 存储协方差特征
        features_per_band = 2 * self.n_components
        covs = np.zeros((batch_size, features_per_band))
        
        for i in range(batch_size):
            # 标准化
            X_norm = X[i] - np.mean(X[i], axis=1, keepdims=True)
            
            # 计算协方差矩阵
            cov = np.cov(X_norm)
            
            # 确保协方差矩阵尺寸正确
            if cov.shape[0] != self.n_channels:
                # 如果大小不匹配，则调整矩阵大小
                if cov.shape[0] > self.n_channels:
                    cov = cov[:self.n_channels, :self.n_channels]
                else:
                    # 如果矩阵太小，使用零填充
                    cov_padded = np.zeros((self.n_channels, self.n_channels))
                    cov_padded[:cov.shape[0], :cov.shape[1]] = cov
                    cov = cov_padded
            
            # 正则化协方差矩阵
            cov = cov / (np.trace(cov) + 1e-10)
            
            # 提取对角线元素作为特征 - 取前n_components个
            diag_features = np.diag(cov)[:self.n_components]
            if len(diag_features) < self.n_components:
                # 填充不足的部分
                diag_features = np.pad(diag_features, 
                                       (0, self.n_components - len(diag_features)), 
                                       'constant')
            
            # 提取非对角元素作为特征 - 取前n_components个
            off_diag = cov - np.diag(np.diag(cov))
            off_diag_flat = off_diag.flatten()
            # 按绝对值大小排序
            idx = np.argsort(np.abs(off_diag_flat))[::-1]  # 降序
            top_indices = idx[:self.n_components]
            top_off_diag = off_diag_flat[top_indices]
            
            if len(top_off_diag) < self.n_components:
                # 填充不足的部分
                top_off_diag = np.pad(top_off_diag, 
                                      (0, self.n_components - len(top_off_diag)), 
                                      'constant')
            
            # 合并对角线和非对角线特征，确保维度正确
            covs[i] = np.concatenate([diag_features, top_off_diag])
        
        return covs


# 简化的EEG特征提取器
class EEGFeatureExtractor(nn.Module):
    def __init__(self, input_channels, seq_length=1000, dropout_rate=0.25, memory_efficient=True, filters=(16, 32, 32, 32)):
        """
        简化的EEG特征提取器
        
        参数:
            input_channels: 输入通道数
            seq_length: 序列长度
            dropout_rate: Dropout比率
            memory_efficient: 是否使用内存高效模式
            filters: 各卷积层的滤波器数量，元组格式
        """
        super(EEGFeatureExtractor, self).__init__()
        
        # 检查filters长度
        if len(filters) != 4:
            print("警告: filters参数必须包含4个元素，使用默认值(16, 32, 32, 32)")
            filters = (16, 32, 32, 32)
        
        # 简化的特征提取网络，只使用4个卷积层
        self.features = nn.Sequential(
            # 第一层卷积
            nn.Conv1d(input_channels, filters[0], kernel_size=5, padding=2),
            nn.BatchNorm1d(filters[0]),
            nn.LeakyReLU(0.1),  # 替代nn.ReLU()
            nn.AvgPool1d(4),  # 降采样到250
            nn.Dropout(dropout_rate * 0.5),
            
            # 第二层卷积
            nn.Conv1d(filters[0], filters[1], kernel_size=5, padding=2),
            nn.BatchNorm1d(filters[1]),
            nn.LeakyReLU(0.1),  # 替代nn.ReLU()
            nn.AvgPool1d(2),  # 降采样到125
            nn.Dropout(dropout_rate * 0.5),
            
            # 第三层卷积
            nn.Conv1d(filters[1], filters[2], kernel_size=3, padding=1),
            nn.BatchNorm1d(filters[2]),
            nn.LeakyReLU(0.1),  # 替代nn.ReLU()
            nn.AvgPool1d(2),  # 降采样到62
            nn.Dropout(dropout_rate * 0.7),
            
            # 第四层卷积
            nn.Conv1d(filters[2], filters[3], kernel_size=3, padding=1),
            nn.BatchNorm1d(filters[3]),
            nn.LeakyReLU(0.1),  # 替代nn.ReLU()
            nn.AvgPool1d(2),  # 降采样到31
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入数据，形状为(batch_size, n_channels, n_samples)
            
        返回:
            提取的特征，形状为(batch_size, n_channels, feature_dim)
        """
        x = self.features(x)
        return x


# 结合FBCSP和PINN的混合模型（简化版）
class FBCSP_PINN_BCI(nn.Module):
    def __init__(self, leadfield, in_channels=22, seq_length=1000, num_classes=2, 
                 dropout_rate=0.35, memory_efficient=True, use_pinn=True, sfreq=250,
                 fbcsp_components=4):
        """
        初始化FBCSP_PINN_BCI模型
        
        参数:
            leadfield: 引线场模型
            in_channels: 输入通道数
            seq_length: 序列长度
            num_classes: 类别数
            dropout_rate: Dropout率
            memory_efficient: 是否使用内存优化
            use_pinn: 是否使用物理信息神经网络
            sfreq: 采样频率
            fbcsp_components: FBCSP特征提取器中的组件数
        """
        super(FBCSP_PINN_BCI, self).__init__()
        
        self.in_channels = in_channels
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.use_pinn = use_pinn
        
        # 获取设备信息
        device = leadfield.device
        
        # 提取引线场模型的维度
        self.lead_field = leadfield
        
        # 创建FBCSP提取器
        self.fbcsp_extractor = FBCSP_FeatureExtractor(
            n_channels=in_channels,
            sfreq=sfreq,
            n_components=fbcsp_components,
            n_features=6 * fbcsp_components  # 3个频带 * 2个极性 * fbcsp_components
        ).to(device)
        
        # 计算FBCSP特征的维度，用于特征融合层
        fbcsp_feature_dim = 6 * fbcsp_components  # 确保与FBCSP_FeatureExtractor中的维度计算保持一致
        
        # 创建深度特征提取器
        self.feature_extractor = EEGFeatureExtractor(
            input_channels=in_channels,
            seq_length=seq_length,
            memory_efficient=memory_efficient,
            dropout_rate=dropout_rate
        ).to(device)
        
        # 添加缺失的池化和展平层
        self.feature_pooling = nn.AdaptiveAvgPool1d(8).to(device)
        self.flatten = nn.Flatten().to(device)
        
        # 创建特征降维层
        self.dim_reduction = nn.Sequential(
            nn.Linear(32 * 8, 64),  # CNN输出特征维度为32*8
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        ).to(device)  # 确保在正确的设备上
        
        # 创建特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(fbcsp_feature_dim + 64, 64),  # FBCSP特征 + 降维特征(64)
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        ).to(device)  # 确保在正确的设备上
        
        # 创建分类器头
        self.classifier = nn.Linear(64, num_classes).to(device)  # 确保在正确的设备上
        
        # 仅在启用PINN时创建源空间映射网络
        if self.use_pinn:
            # 创建源空间映射网络 - 增强版本
            self.source_fc = nn.Sequential(
                nn.Linear(64, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate),
                nn.Linear(128, leadfield.shape[1]),
                nn.Tanh()
            ).to(device)  # 确保在正确的设备上
            
            # 创建残差映射网络 - 直接从原始EEG特征到重建EEG
            self.residual_projection = nn.Sequential(
                nn.Linear(32 * 8, 64),
                nn.LeakyReLU(0.1),
                nn.Linear(64, in_channels),
                nn.Tanh()
            ).to(device)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入数据，形状为(batch_size, n_channels, seq_length)
            
        返回:
            dict: 包含logits和projected_features的字典
        """
        # 确保所有数据在同一设备上
        device = x.device
        self.lead_field = self.lead_field.to(device)
        
        batch_size = x.shape[0]
        
        # FBCSP特征提取
        fbcsp_features = self.fbcsp_extractor(x)
        
        # 深度特征提取
        deep_features = self.feature_extractor(x)
        
        # 池化和展平
        deep_features = self.feature_pooling(deep_features)
        deep_features = self.flatten(deep_features)
        
        # 降维
        reduced_features = self.dim_reduction(deep_features)
        
        # 特征融合
        combined_features = torch.cat((fbcsp_features, reduced_features), dim=1)
        fused_features = self.feature_fusion(combined_features)
        
        # 分类
        logits = self.classifier(fused_features)
        
        # 物理信息映射（如果启用）
        projected_features = None
        if self.use_pinn:
            # 注意：这里使用深度特征作为源空间映射的输入
            source_features = self.source_fc(reduced_features)
            # 使用引线场矩阵投影到电极空间
            projected_features = torch.matmul(source_features, self.lead_field.t())
            
            # 添加残差连接，直接从深层特征到电极空间
            residual_contribution = self.residual_projection(deep_features)
            
            # 组合投影特征和残差贡献
            projected_features = projected_features + 0.3 * residual_contribution
            
            # 标准化
            projected_features = nn.functional.normalize(projected_features, p=2, dim=1)
        
        return {
            'logits': logits,
            'projected_features': projected_features,
            'fbcsp_features': fbcsp_features,
            'deep_features': reduced_features,
            'fused_features': fused_features
        }

# 1. 数据加载与预处理
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


# 2. 头模型构建
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


# 4. 训练与评估
def train_model(data_dir, subject_id=None, num_epochs=500, batch_size=64, repeat_factor=3, 
              dropout_rate=0.5, fbcsp_components=4, learning_rate=0.0025, physics_weight=0.4, use_pinn=True):
    """
    训练PINN模型函数
    单个受试者数据进行训练和测试
    
    参数:
    data_dir: 数据目录
    subject_id: 被试ID
    num_epochs: 训练轮数
    batch_size: 批次大小
    repeat_factor: 数据扩充倍数
    dropout_rate: Dropout比率
    fbcsp_components: FBCSP组件数
    learning_rate: 基础学习率
    physics_weight: 物理损失权重，若为None则使用动态权重
    use_pinn: 是否使用物理信息神经网络，默认True
    """
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

    # 构建头模型，只在第一个被试时详细打印
    if subject_id == 1:
        print("构建头模型...")
        subjects_dir = os.path.join(mne.get_config('SUBJECTS_DIR') or '', 'fsaverage', '..')
        subjects_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../freesurfer/subjects')) if not os.path.exists(subjects_dir) else subjects_dir
        fwd = build_head_model(subjects_dir, subject='fsaverage')  # 返回Forward对象
        leadfield = torch.tensor(fwd['sol']['data'], dtype=torch.float32).to(device)
    else:
        # 对其他被试，使用静默模式构建头模型
        print(f"为被试 S{subject_id} 构建头模型...")
        subjects_dir = os.path.join(mne.get_config('SUBJECTS_DIR') or '', 'fsaverage', '..')
        subjects_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../freesurfer/subjects')) if not os.path.exists(subjects_dir) else subjects_dir
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            fwd = build_head_model(subjects_dir, subject='fsaverage')
            leadfield = torch.tensor(fwd['sol']['data'], dtype=torch.float32).to(device)

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
        # 交叉验证阶段不再使用测试集评估
        # test_loader = DataLoader(
        #     test_dataset, 
        #     batch_size=batch_size, 
        #     shuffle=False,
        #     num_workers=0,
        #     pin_memory=True
        # )
        
        # 获取第一个batch进行形状检查
        sample_batch = next(iter(train_loader))
        sample_eeg, sample_label = sample_batch
        
        # 获取EEG数据的尺寸
        num_channels = sample_eeg.shape[1]  # 获取通道数
        num_timepoints = sample_eeg.shape[2]  # 获取时间点数
        
        # 预计算采样频率
        sfreq = 250  # BCI竞赛IV-2a数据集的标准采样频率
        
        # 创建改进的模型
        model = FBCSP_PINN_BCI(
            leadfield, 
            in_channels=num_channels, 
            seq_length=num_timepoints, 
            num_classes=2,  # 只使用左右手数据，类别数为2
            dropout_rate=dropout_rate,  # 使用传入的dropout_rate
            memory_efficient=repeat_factor > 1,
            use_pinn=use_pinn,  # 使用传入的use_pinn
            sfreq=sfreq,  # 传入采样频率，用于FBCSP滤波
            fbcsp_components=fbcsp_components  # 使用传入的fbcsp_components
        ).to(device)
        print(f"引线场形状: {leadfield.shape}")
        print(f"PINN模式: {'启用' if model.use_pinn else '禁用'}")
        print(f"FBCSP特征提取器已启用")
        
        # 分层优化器策略
        fbcsp_params = list(model.fbcsp_extractor.parameters())
        cnn_params = list(model.feature_extractor.parameters())
        fusion_params = list(model.feature_fusion.parameters()) if hasattr(model, 'feature_fusion') else []
        other_params = [p for p in model.parameters() if not any(p is fp for fp in fbcsp_params + cnn_params + fusion_params)]
        
        optimizer = optim.AdamW([
            {'params': fbcsp_params, 'lr': learning_rate, 'weight_decay': 5e-4},  # FBCSP特征提取器参数
            {'params': cnn_params, 'lr': learning_rate, 'weight_decay': 5e-4},   # CNN特征提取器参数
            {'params': fusion_params, 'lr': learning_rate, 'weight_decay': 5e-4},  # 特征融合层参数
            {'params': other_params, 'lr': learning_rate, 'weight_decay': 5e-4}    # 其他参数
        ])
        
        # 使用余弦退火学习率调度
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-5
        )
        
        # 计算每个类别的样本数，用于为类别分配权重
        all_labels = []
        for _, labels in train_loader:
            all_labels.append(labels.flatten())
        all_labels = torch.cat(all_labels)

        # 计算类别频率 - 将标签从1,2映射到0,1
        mapped_labels = all_labels - 1  # 将1映射到0，将2映射到1
        label_counts = torch.bincount(mapped_labels, minlength=2)  # 确保有2个类别的计数

        # 有效样本数法计算类别权重（beta=0.99）
        beta = 0.99
        effective_num = 1.0 - torch.pow(beta, label_counts.float())
        weights = (1.0 - beta) / (effective_num + 1e-8)
        # 归一化权重，使总和等于类别数
        weights = weights / weights.sum() * 2

        # 使用带权重的交叉熵损失函数
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        
        # 早停机制设置
        patience = 15
        best_val_acc = 0.0
        early_stop_counter = 0
        best_f1 = 0.0
        best_epoch = 0
        best_model_state = None

        # 训练循环
        dynamic_weight_interval = 5  # 每N个epoch动态调整一次类别权重
        for epoch in range(num_epochs):
            # 动态调整类别权重
            if epoch % dynamic_weight_interval == 0:
                all_labels = []
                for _, labels in train_loader:
                    all_labels.append(labels.flatten())
                all_labels = torch.cat(all_labels)
                mapped_labels = all_labels - 1  # 将1映射到0，将2映射到1
                label_counts = torch.bincount(mapped_labels, minlength=2)
                beta = 0.99
                effective_num = 1.0 - torch.pow(beta, label_counts.float())
                weights = (1.0 - beta) / (effective_num + 1e-8)
                weights = weights / weights.sum() * 2
                criterion.weight = weights.to(device)

            # 记录每个epoch的开始时间
            epoch_start_time = time.time()
            
            # 动态调整物理损失权重
            if physics_weight is None:
                # 使用动态权重
                progress = min(1.0, epoch / (num_epochs * 0.7))
                current_physics_weight = max(0.05, 0.2 * (1.0 - progress))  # 从0.2减小到0.05
            else:
                # 使用用户指定的静态权重
                current_physics_weight = physics_weight
            
            model.train()
            train_loss = 0.0
            train_class_loss = 0.0
            train_physics_loss = 0.0
            batch_count = 0
            correct = 0
            total = 0
            
            # 训练批次处理
            for batch_idx, (eeg_data, labels) in enumerate(train_loader):
                batch_count += 1
                
                # 准备数据，确保使用float32类型
                eeg_data = eeg_data.to(device, dtype=torch.float32)
                
                # 为物理损失计算通道平均活动
                physical_target = torch.mean(eeg_data, dim=2)
                # 归一化物理目标，使其数值稳定
                physical_target = F.normalize(physical_target, p=2, dim=1)
                
                # 处理标签 (将1映射到0，将2映射到1)
                labels = labels.flatten() - 1  # 1->0, 2->1
                labels = labels.to(device)
                
                # 清除旧的梯度
                optimizer.zero_grad()
                
                # 前向传播
                outputs = model(eeg_data)
                
                # 计算分类损失
                loss_class = criterion(outputs['logits'], labels)
                
                # 物理损失
                physical_target = torch.mean(eeg_data, dim=2)
                physical_target = F.normalize(physical_target, p=2, dim=1)
                
                if model.use_pinn and outputs['projected_features'] is not None:
                    # 原始MSE损失
                    mse_physics = nn.MSELoss()(outputs['projected_features'], physical_target)
                    
                    # 引入相关性损失，导入来自source_evaluation模块
                    from source_evaluation import correlation_loss, source_sparsity_loss
                    corr_physics = correlation_loss(outputs['projected_features'], physical_target)
                    
                    # 获取源特征并计算稀疏性损失
                    source_features = model.source_fc(outputs['deep_features'])
                    sparse_loss = source_sparsity_loss(source_features, l1_weight=0.005)
                    
                    # 组合MSE损失和相关性损失，权重各为0.5和0.5
                    loss_physics = 0.4 * mse_physics + 0.4 * corr_physics + 0.2 * sparse_loss
                    
                    # 组合物理损失和分类损失
                    loss = current_physics_weight * loss_physics + (1 - current_physics_weight) * loss_class
                else:
                    # 不使用PINN时不计算物理损失
                    loss_physics = torch.tensor(0.0, device=device)
                    loss = loss_class
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # 记录损失值
                train_loss += loss.item()
                train_physics_loss += loss_physics.item()
                train_class_loss += loss_class.item()
                
                # 计算准确率
                _, predicted = outputs['logits'].max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 清理内存
                if model.use_pinn:
                    del outputs, loss, loss_physics, physical_target
                    torch.cuda.empty_cache()
            
            # 计算每个epoch的平均损失和准确率
            avg_train_loss = train_loss / max(batch_count, 1)
            avg_train_phys = train_physics_loss / max(batch_count, 1)
            avg_train_class = train_class_loss / max(batch_count, 1)
            train_acc = 100.0 * correct / total

            # 记录到历史字典中（只append一次，且为平均值）
            history['train_loss'].append(avg_train_loss)
            history['train_class_loss'].append(avg_train_class)
            history['train_physics_loss'].append(avg_train_phys)
            history['train_acc'].append(train_acc)

            # 计算每个epoch的训练时间
            epoch_time = time.time() - epoch_start_time

            # 验证（用val_loader）
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_preds = []
            val_targets = []
            conf_matrix = torch.zeros(2, 2)
            with torch.no_grad():
                for batch_idx, (eeg_data, labels) in enumerate(val_loader):
                    eeg_data = eeg_data.to(device, dtype=torch.float32)
                    labels = labels.flatten() - 1
                    labels = labels.to(device)
                    outputs = model(eeg_data)
                    loss_class = criterion(outputs['logits'], labels)
                    physical_target = torch.mean(eeg_data, dim=2)
                    physical_target = F.normalize(physical_target, p=2, dim=1)
                    if model.use_pinn and outputs['projected_features'] is not None:
                        mse_physics = nn.MSELoss()(outputs['projected_features'], physical_target)
                        from source_evaluation import correlation_loss, source_sparsity_loss
                        corr_physics = correlation_loss(outputs['projected_features'], physical_target)
                        source_features = model.source_fc(outputs['deep_features'])
                        sparse_loss = source_sparsity_loss(source_features, l1_weight=0.005)
                        loss_physics = 0.4 * mse_physics + 0.4 * corr_physics + 0.2 * sparse_loss
                        loss = current_physics_weight * loss_physics + (1 - current_physics_weight) * loss_class
                    else:
                        loss_physics = torch.tensor(0.0, device=device)
                        loss = loss_class
                    val_loss += loss.item()
                    _, predicted = outputs['logits'].max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    val_preds.extend(predicted.cpu().numpy())
                    val_targets.extend(labels.cpu().numpy())
                    for t, p in zip(labels.view(-1), predicted.view(-1)):
                        conf_matrix[t, p] += 1
            val_acc = 100.0 * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            val_f1 = f1_score(val_targets, val_preds, average='macro')
            
            # 记录到历史字典中
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            
            # 计算每个类别的F1分数
            val_f1 = f1_score(val_targets, val_preds, average='macro')
            
            # 更新学习率调度器
            scheduler.step()
            
            # 检查是否是最佳验证准确率
            is_best = val_acc > best_val_acc
            
            # 如果验证准确率提高，则更新最佳准确率和F1分数
            if is_best:
                best_val_acc = val_acc
                best_f1 = val_f1
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())
                early_stop_counter = 0  # 重置早停计数器
            else:
                early_stop_counter += 1
            
            # 每10个epoch结束时打印信息
            if (epoch+1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1 or is_best:
                print(f"被试 S{subject_id} | Epoch {epoch+1}/{num_epochs} | "
                      f"时间: {epoch_time:.2f}s | "
                      f"训练准确率: {train_acc:.2f}% | "
                      f"验证准确率: {val_acc:.2f}% | "
                      f"F1: {val_f1:.4f}")
            
            # 清理内存
            if model.use_pinn:
                torch.cuda.empty_cache()
            
            # 早停检查
            if early_stop_counter >= patience:
                print(f"早停: 被试 S{subject_id} 验证准确率已经{patience}个epoch没有提高")
                break
        
        # 训练性能分析
        print(f"\n被试 S{subject_id} 训练性能分析:")
        print(f"最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch+1})")
        print(f"最佳F1分数: {best_f1:.4f}")
        
        # 加载最佳模型参数用于最终评估
        if best_model_state:
            model.load_state_dict(best_model_state)
            print(f"加载最佳模型 (Epoch {best_epoch+1})")
        
        # 最终模型评估
        model.eval()
        test_preds, test_truths = [], []
        train_preds, train_truths = [], []
        
        # 在测试集上评估
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        with torch.no_grad():
            for eeg_data, labels in test_loader:
                eeg_data = eeg_data.to(device, dtype=torch.float32)
                # 处理标签 (将1映射到0，将2映射到1)
                labels = labels.flatten() - 1  # 1->0, 2->1
                labels = labels.to(device)
                
                outputs = model(eeg_data)
                _, predicted = outputs['logits'].max(1)
                
                test_preds.extend(predicted.cpu().tolist())
                test_truths.extend(labels.cpu().tolist())
        
        # 在训练集上评估
        with torch.no_grad():
            for eeg_data, labels in train_loader:
                eeg_data = eeg_data.to(device, dtype=torch.float32)
                # 处理标签 (将1映射到0，将2映射到1)
                labels = labels.flatten() - 1  # 1->0, 2->1
                labels = labels.to(device)
                
                outputs = model(eeg_data)
                _, predicted = outputs['logits'].max(1)
                
                train_preds.extend(predicted.cpu().tolist())
                train_truths.extend(labels.cpu().tolist())
        
        # 计算最终指标
        test_acc = accuracy_score(test_truths, test_preds) * 100
        test_f1 = f1_score(test_truths, test_preds, average='macro')
        test_cm = confusion_matrix(test_truths, test_preds)
        test_kappa = cohen_kappa_score(test_truths, test_preds)
        
        train_acc = accuracy_score(train_truths, train_preds) * 100
        train_f1 = f1_score(train_truths, train_preds, average='macro')
        
        print(f"\n被试 S{subject_id} 最终模型评估:")
        print(f"训练集准确率: {train_acc:.2f}%, F1分数: {train_f1:.4f}")
        print(f"验证集准确率: {best_val_acc:.2f}%, F1分数: {best_f1:.4f}")
        print(f"测试集准确率: {test_acc:.2f}%, F1分数: {test_f1:.4f}")
        print(f"测试集Kappa值: {test_kappa:.4f}")
        print("测试集混淆矩阵:")
        print(test_cm)
        # 混淆矩阵可视化
        plt.figure(figsize=(6, 5))
        im = plt.imshow(test_cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'S{subject_id} Confusion Matrix')
        plt.colorbar(im)
        classes = ["左手", "右手"]
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        thresh = test_cm.max() / 2.
        for i in range(test_cm.shape[0]):
            for j in range(test_cm.shape[1]):
                plt.text(j, i, format(test_cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if test_cm[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        os.makedirs('results', exist_ok=True)
        plt.savefig(f'results/S{subject_id}_confusion_matrix.png', dpi=150)
        plt.close()
        print(f"混淆矩阵图已保存至 results/S{subject_id}_confusion_matrix.png")
        
        # 记录本折训练集准确率和F1分数（用当前模型在训练集上评估）
        model.eval()
        train_preds, train_truths = [], []
        with torch.no_grad():
            for eeg_data, labels in train_loader:
                eeg_data = eeg_data.to(device, dtype=torch.float32)
                labels = labels.flatten() - 1
                labels = labels.to(device)
                outputs = model(eeg_data)
                _, predicted = outputs['logits'].max(1)
                train_preds.extend(predicted.cpu().tolist())
                train_truths.extend(labels.cpu().tolist())
        train_acc = accuracy_score(train_truths, train_preds) * 100
        train_f1 = f1_score(train_truths, train_preds, average='macro')
        # 追加到all_fold_results
        all_fold_results.append((fold_num, train_acc, train_f1, best_val_acc, best_f1))

        # 绘制loss曲线（每一折单独画）
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
    # 9折全部完成后，统计平均准确率、F1等指标
    if all_fold_results:
        arr = np.array(all_fold_results)
        train_avg_acc = np.mean(arr[:,1])
        train_avg_f1 = np.mean(arr[:,2])
        val_avg_acc = np.mean(arr[:,3])
        val_avg_f1 = np.mean(arr[:,4])
        print("\n===== 被试 S{} 9折交叉验证平均结果 =====".format(subject_id))
        print("训练集平均准确率: {:.2f}%, F1: {:.4f}".format(train_avg_acc, train_avg_f1))
        print("验证集平均准确率: {:.2f}%, F1: {:.4f}".format(val_avg_acc, val_avg_f1))
    # ========== 9折交叉验证后用全部训练集重新训练并在测试集评估 ==========
    print(f"\n===== 被试 S{subject_id} 最终模型在独立测试集上的评估 =====")
    # 全集训练阶段：将全部训练集划分为90%训练、10%验证
    sss_full = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    full_indices = np.arange(len(train_dataset))
    full_labels = np.array([train_dataset[i][1] for i in full_indices])
    train90_idx, val10_idx = next(sss_full.split(full_indices, full_labels))
    train90_dataset = Subset(train_dataset, train90_idx)
    val10_dataset = Subset(train_dataset, val10_idx)
    # 只对90%训练集做数据增强
    if repeat_factor > 1:
        print(f"全集训练阶段数据增强: 启用 (仅应用于90%训练集，将创建{repeat_factor-1}倍的增强数据)")
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
        train_set_original = AugmentedDataset(train_dataset, train90_idx, enhance=False)
        all_datasets = [train_set_original]
        for _ in range(repeat_factor - 1):
            augmented_set = AugmentedDataset(train_dataset, train90_idx, enhance=True)
            all_datasets.append(augmented_set)
        train90_dataset = ConcatDataset(all_datasets)
        print(f"全集训练集总大小(含增强): {len(train90_dataset)}，其中原始样本: {len(train_set_original)}，增强样本: {len(train90_dataset)-len(train_set_original)}")
    else:
        print(f"全集训练阶段数据增强: 禁用 (使用原始数据)")
    # 重新初始化模型
    model = FBCSP_PINN_BCI(
        leadfield, 
        in_channels=dataset.data.shape[1], 
        seq_length=dataset.data.shape[2], 
        num_classes=2, 
        dropout_rate=dropout_rate, 
        memory_efficient=repeat_factor > 1,
        use_pinn=use_pinn, 
        sfreq=250, 
        fbcsp_components=fbcsp_components
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    # 全集训练前重置history
    history = {
        'train_loss': [], 'train_class_loss': [], 'train_physics_loss': [],
        'train_acc': [], 'val_loss': [], 'val_acc': [],
        'test_loss': [], 'test_acc': []
    }
    # 用90%训练集训练模型
    train_loader = DataLoader(
        train90_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val10_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    patience = 15
    best_train_acc = 0.0
    early_stop_counter = 0
    best_epoch = 0
    best_model_state = None
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for eeg_data, labels in train_loader:
            eeg_data = eeg_data.to(device, dtype=torch.float32)
            labels = labels.flatten() - 1
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(eeg_data)
            loss = criterion(outputs['logits'], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs['logits'].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        # 在验证集上评估
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
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
        # 在测试集上评估
        test_correct = 0
        test_total = 0
        test_loss = 0.0
        with torch.no_grad():
            for eeg_data, labels in test_loader:
                eeg_data = eeg_data.to(device, dtype=torch.float32)
                labels = labels.flatten() - 1
                labels = labels.to(device)
                outputs = model(eeg_data)
                loss = criterion(outputs['logits'], labels)
                test_loss += loss.item()
                _, predicted = outputs['logits'].max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        avg_test_loss = test_loss / len(test_loader)
        test_acc = 100.0 * test_correct / test_total
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_class_loss'].append(0.0)
        history['train_physics_loss'].append(0.0)
        history['train_acc'].append(100. * correct / total)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(100. * val_correct / val_total)
        history['test_loss'].append(avg_test_loss)
        history['test_acc'].append(test_acc)
        if (epoch+1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(f"被试 S{subject_id} | 全集训练 Epoch {epoch+1}/{num_epochs} | 训练准确率: {train_acc:.2f}% | 验证准确率: {100. * val_correct / val_total:.2f}% | 损失: {avg_train_loss:.4f}")
        # 早停逻辑：以验证集准确率为监控指标
        if 100. * val_correct / val_total > best_train_acc:
            best_train_acc = 100. * val_correct / val_total
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"全集训练早停: {patience}个epoch验证准确率未提升")
            break
    # 恢复最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"加载全集训练最佳模型 (Epoch {best_epoch+1})")
        # 用最佳模型在测试集上重新评估
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
    # 全集训练过程可视化
    
    def smooth_curve(points, factor=0.8):
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(history['train_loss'])), smooth_curve(history['train_loss']), label='训练损失')
    plt.plot(range(len(history['val_loss'])), smooth_curve(history['val_loss']), label='验证损失')
    plt.plot(range(len(history['test_loss'])), smooth_curve(history['test_loss']), label='测试损失')
    plt.title(f'被试S{subject_id}全集训练Loss曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(range(len(history['train_acc'])), smooth_curve(history['train_acc']), label='训练准确率')
    plt.plot(range(len(history['val_acc'])), smooth_curve(history['val_acc']), label='验证准确率')
    plt.plot(range(len(history['test_acc'])), smooth_curve(history['test_acc']), label='测试准确率')
    plt.title(f'被试S{subject_id}全集训练Accuracy曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/S{subject_id}_full_training_curves.png', dpi=150)
    plt.close()
    return all_fold_results, (test_acc, test_f1, test_kappa)

# 运行主程序
if __name__ == '__main__':
    # 设置参数
    batch_size = 64
    num_epochs = 150  # 减少训练周期以加快训练速度
    repeat_factor = 2  # 使用原始数据集不增强
    
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
    data_dir = r"E:\pycharm\PINN\PINN\data\BCI2a"
    
    # 检查数据目录结构
    if os.path.exists(data_dir):
        print("=== 使用简化版FBCSP_PINN混合模型进行训练 ===")
        
        # 检查是否存在所有受试者数据
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
            results, testset_result = train_model(data_dir, subject_id=subject_id, num_epochs=num_epochs, 
                                 batch_size=batch_size, repeat_factor=repeat_factor)
            arr = np.array(results)
            train_avg_acc = np.mean(arr[:,1])
            train_avg_f1 = np.mean(arr[:,2])
            val_avg_acc = np.mean(arr[:,3])
            val_avg_f1 = np.mean(arr[:,4])
            # 只统计交叉验证相关的平均
            subject_results.append((subject_id, train_avg_acc, train_avg_f1, val_avg_acc, val_avg_f1))
            # 测试集结果单独存储
            testset_results.append((subject_id, *testset_result))
        
        # 打印所有被试的结果汇总
        print("\n\n===== 所有被试结果汇总 (简化版FBCSP_PINN模型) =====")
        print("被试\t训练集准确率\t验证集准确率\t最终测试集准确率\t最终测试集F1\t最终测试集Kappa")
        avg_train_acc = np.mean([r[1] for r in subject_results])
        avg_val_acc = np.mean([r[3] for r in subject_results])
        avg_test_acc = np.mean([r[1] for r in testset_results])
        avg_test_f1 = np.mean([r[2] for r in testset_results])
        avg_test_kappa = np.mean([r[3] for r in testset_results])
        for i, subj_result in enumerate(subject_results):
            subj, train_acc, train_f1, val_acc, val_f1 = subj_result
            testset_acc, testset_f1, testset_kappa = testset_results[i][1:]
            print(f"S{subj}\t{train_acc:.2f}%\t\t{val_acc:.2f}%\t\t{testset_acc:.2f}%\t\t{testset_f1:.4f}\t\t{testset_kappa:.4f}")
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
        