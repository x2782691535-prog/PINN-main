#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试PINN模型的半球预测能力
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from BCI_PINN_only_ablation import PINN_Only_BCI, BCI2aDataset, build_head_model
import mne

def test_hemisphere_prediction():
    """测试PINN模型是否能正确预测源点在哪个半球"""
    
    print("=== 测试PINN模型的半球预测能力 ===")
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据路径
    data_dir = r"E:\pycharm\PINN\PINN\data\BCI2a"
    
    if not os.path.exists(data_dir):
        print(f"错误：数据目录不存在: {data_dir}")
        return
    
    # 构建头模型
    print("构建头模型...")
    subjects_dir = str(mne.datasets.sample.data_path() / 'subjects')
    fwd = build_head_model(subjects_dir)
    
    # 获取引线场矩阵
    leadfield = torch.tensor(fwd['sol']['data'], dtype=torch.float32).to(device)
    print(f"引线场形状: {leadfield.shape}")
    
    # 加载测试数据（只用S1的少量数据）
    print("加载测试数据...")
    dataset = BCI2aDataset(data_dir, [1], transform=None, repeat_factor=1)
    
    # 取前20个样本进行测试
    test_samples = 20
    test_data = []
    test_labels = []
    
    for i in range(min(test_samples, len(dataset))):
        data, label = dataset[i]
        test_data.append(torch.tensor(data, dtype=torch.float32))
        test_labels.append(label)
    
    test_data = torch.stack(test_data).to(device)  # (20, 22, 1000)
    test_labels = torch.tensor(test_labels).to(device) - 1  # 转换为0,1
    
    print(f"测试数据形状: {test_data.shape}")
    print(f"测试标签: {test_labels}")
    print(f"标签分布: 左手(0): {(test_labels == 0).sum()}, 右手(1): {(test_labels == 1).sum()}")
    
    # 创建模型
    print("创建PINN模型...")
    model = PINN_Only_BCI(
        leadfield=leadfield,
        fwd_model=fwd,
        epochs_info=None,
        in_channels=22,
        seq_length=1000,
        num_classes=2,
        dropout_rate=0.35
    ).to(device)
    
    # 测试模式
    model.eval()
    
    with torch.no_grad():
        # 前向传播
        outputs = model(test_data)
        
        # 获取预测结果
        logits = outputs['logits']
        predictions = torch.argmax(logits, dim=1)
        
        # 获取空间特征
        source_activations = outputs['source_activations']  # (batch, n_sources)
        hemisphere_features = outputs['hemisphere_features']  # (batch, 8)
        
        print(f"\n=== 模型输出分析 ===")
        print(f"Logits形状: {logits.shape}")
        print(f"源激活形状: {source_activations.shape}")
        print(f"半球特征形状: {hemisphere_features.shape}")
        
        # 分析源点位置
        print(f"\n=== 源点位置分析 ===")
        source_positions = model.source_positions  # (n_sources, 3)
        print(f"源点位置形状: {source_positions.shape}")
        
        # 统计左右半球源点数量
        x_coords = source_positions[:, 0]
        left_sources = (x_coords < 0).sum().item()
        right_sources = (x_coords > 0).sum().item()
        center_sources = (x_coords == 0).sum().item()
        
        print(f"左半球源点 (x < 0): {left_sources}")
        print(f"右半球源点 (x > 0): {right_sources}")
        print(f"中线源点 (x = 0): {center_sources}")
        
        # 分析每个样本的半球激活模式
        print(f"\n=== 半球激活模式分析 ===")
        
        for i in range(min(10, test_samples)):  # 只分析前10个样本
            sample_activations = source_activations[i]  # (n_sources,)
            true_label = test_labels[i].item()
            pred_label = predictions[i].item()
            
            # 计算左右半球平均激活
            left_mask = x_coords < 0
            right_mask = x_coords > 0
            
            left_activation = torch.mean(sample_activations[left_mask]).item()
            right_activation = torch.mean(sample_activations[right_mask]).item()
            
            # 预测的优势半球
            dominant_hemisphere = "左半球" if left_activation > right_activation else "右半球"
            
            # 期望的优势半球（基于运动想象的对侧控制原理）
            expected_hemisphere = "右半球" if true_label == 0 else "左半球"  # 左手->右半球，右手->左半球
            
            correct_hemisphere = "正确" if dominant_hemisphere == expected_hemisphere else "错误"
            
            print(f"样本 {i+1:2d}: 真实={['左手', '右手'][true_label]}, 预测={['左手', '右手'][pred_label]} | "
                  f"左半球激活={left_activation:.4f}, 右半球激活={right_activation:.4f} | "
                  f"优势半球={dominant_hemisphere}, 期望={expected_hemisphere} [{correct_hemisphere}]")
        
        # 计算准确率
        accuracy = (predictions == test_labels).float().mean().item()
        print(f"\n=== 分类准确率 ===")
        print(f"分类准确率: {accuracy:.2%}")
        
        # 计算半球预测准确率
        hemisphere_correct = 0
        for i in range(test_samples):
            sample_activations = source_activations[i]
            true_label = test_labels[i].item()
            
            left_activation = torch.mean(sample_activations[left_mask]).item()
            right_activation = torch.mean(sample_activations[right_mask]).item()
            
            # 预测的优势半球：左半球激活高->左半球优势，右半球激活高->右半球优势
            pred_dominant = 0 if left_activation > right_activation else 1  # 0=左半球，1=右半球
            
            # 期望的优势半球：左手(0)->右半球(1)，右手(1)->左半球(0)
            expected_dominant = 1 - true_label  # 对侧控制
            
            if pred_dominant == expected_dominant:
                hemisphere_correct += 1
        
        hemisphere_accuracy = hemisphere_correct / test_samples
        print(f"半球预测准确率: {hemisphere_accuracy:.2%}")
        
        # 分析源激活的空间分布
        print(f"\n=== 源激活空间分布分析 ===")
        mean_activations = torch.mean(torch.abs(source_activations), dim=0)  # (n_sources,)
        
        # 找到最活跃的源点
        top_k = 5
        top_indices = torch.topk(mean_activations, top_k).indices
        
        print(f"最活跃的 {top_k} 个源点:")
        for i, idx in enumerate(top_indices):
            pos = source_positions[idx]
            activation = mean_activations[idx].item()
            hemisphere = "左半球" if pos[0] < 0 else "右半球" if pos[0] > 0 else "中线"
            print(f"  {i+1}. 源点{idx}: 位置=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
                  f"激活={activation:.4f}, 半球={hemisphere}")
        
        return {
            'classification_accuracy': accuracy,
            'hemisphere_accuracy': hemisphere_accuracy,
            'source_positions': source_positions.cpu().numpy(),
            'source_activations': source_activations.cpu().numpy(),
            'predictions': predictions.cpu().numpy(),
            'true_labels': test_labels.cpu().numpy()
        }

if __name__ == '__main__':
    results = test_hemisphere_prediction()
    
    if results:
        print(f"\n=== 总结 ===")
        print(f"分类准确率: {results['classification_accuracy']:.2%}")
        print(f"半球预测准确率: {results['hemisphere_accuracy']:.2%}")
        
        if results['hemisphere_accuracy'] > 0.6:
            print("PINN模型具有一定的半球预测能力")
        else:
            print("PINN模型的半球预测能力较弱，需要进一步改进")
