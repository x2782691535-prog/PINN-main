#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查BCI-IV-2a数据集的.mat文件内容
"""

import scipy.io as sio
import numpy as np
import os

def inspect_mat_file(filepath):
    """检查单个.mat文件的内容"""
    print(f"\n{'='*60}")
    print(f"检查文件: {filepath}")
    print(f"文件大小: {os.path.getsize(filepath) / (1024*1024):.2f} MB")
    print(f"{'='*60}")
    
    try:
        # 加载.mat文件
        mat_data = sio.loadmat(filepath)
        
        print(f"\n文件中的变量:")
        for key, value in mat_data.items():
            if not key.startswith('__'):  # 跳过元数据
                print(f"  {key}: {type(value)} - {np.array(value).shape if hasattr(value, 'shape') else 'N/A'}")
        
        # 详细检查主要数据变量
        main_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        
        for key in main_keys:
            data = mat_data[key]
            print(f"\n详细检查变量: {key}")
            print(f"  类型: {type(data)}")
            
            if isinstance(data, np.ndarray):
                print(f"  形状: {data.shape}")
                print(f"  数据类型: {data.dtype}")
                print(f"  数值范围: [{np.min(data):.6f}, {np.max(data):.6f}]")
                print(f"  均值: {np.mean(data):.6f}")
                print(f"  标准差: {np.std(data):.6f}")
                
                # 如果是多维数组，显示更多信息
                if data.ndim > 1:
                    print(f"  各维度信息:")
                    for i, dim_size in enumerate(data.shape):
                        print(f"    维度 {i}: 大小 = {dim_size}")
                
                # 显示一些样本数据
                if data.size > 0:
                    if data.ndim == 1:
                        sample_size = min(10, len(data))
                        print(f"  前{sample_size}个值: {data[:sample_size]}")
                    elif data.ndim == 2:
                        print(f"  数据样本 (前3x3):")
                        sample_rows = min(3, data.shape[0])
                        sample_cols = min(3, data.shape[1])
                        print(f"    {data[:sample_rows, :sample_cols]}")
                    elif data.ndim == 3:
                        print(f"  3D数据样本 (第1个切片的前3x3):")
                        if data.shape[0] > 0:
                            sample_rows = min(3, data.shape[1])
                            sample_cols = min(3, data.shape[2])
                            print(f"    {data[0, :sample_rows, :sample_cols]}")
            
            elif hasattr(data, '__len__'):
                print(f"  长度: {len(data)}")
                if len(data) > 0:
                    print(f"  第一个元素类型: {type(data[0])}")
                    if hasattr(data[0], 'shape'):
                        print(f"  第一个元素形状: {data[0].shape}")
            
            print(f"  内存使用: {data.nbytes / (1024*1024):.2f} MB" if hasattr(data, 'nbytes') else "")
        
        # 如果存在特定的BCI-IV-2a字段，进行专门分析
        if 'rawdata' in mat_data:
            print(f"\nBCI-IV-2a数据分析:")
            rawdata = mat_data['rawdata']
            print(f"  原始数据形状: {rawdata.shape}")
            
            if len(rawdata.shape) == 3:
                n_trials, n_channels, n_timepoints = rawdata.shape
                print(f"  试次数: {n_trials}")
                print(f"  通道数: {n_channels}")
                print(f"  时间点数: {n_timepoints}")
                print(f"  采样频率推测: ~250 Hz (基于1000个时间点 = 4秒)")
                
                # 分析每个通道的统计信息
                print(f"\n  各通道统计信息 (前5个通道):")
                for ch in range(min(5, n_channels)):
                    ch_data = rawdata[:, ch, :]
                    print(f"    通道 {ch+1}: 均值={np.mean(ch_data):.4f}, 标准差={np.std(ch_data):.4f}")
        
        if 'label' in mat_data:
            labels = mat_data['label']
            print(f"\n标签分析:")
            print(f"  标签形状: {labels.shape}")
            print(f"  标签类型: {labels.dtype}")
            
            if labels.ndim == 1:
                unique_labels, counts = np.unique(labels, return_counts=True)
                print(f"  唯一标签: {unique_labels}")
                print(f"  标签分布: {dict(zip(unique_labels, counts))}")
            elif labels.ndim == 2:
                # 如果是2D数组，可能需要展平
                labels_flat = labels.flatten()
                unique_labels, counts = np.unique(labels_flat, return_counts=True)
                print(f"  唯一标签 (展平后): {unique_labels}")
                print(f"  标签分布: {dict(zip(unique_labels, counts))}")
        
        return mat_data
        
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

def compare_files(file1_data, file2_data, file1_name, file2_name):
    """比较两个文件的结构"""
    print(f"\n{'='*60}")
    print(f"比较 {file1_name} 和 {file2_name}")
    print(f"{'='*60}")
    
    if file1_data is None or file2_data is None:
        print("无法比较，因为某个文件读取失败")
        return
    
    # 比较变量名
    keys1 = set(k for k in file1_data.keys() if not k.startswith('__'))
    keys2 = set(k for k in file2_data.keys() if not k.startswith('__'))
    
    print(f"变量比较:")
    print(f"  {file1_name} 的变量: {sorted(keys1)}")
    print(f"  {file2_name} 的变量: {sorted(keys2)}")
    print(f"  共同变量: {sorted(keys1 & keys2)}")
    print(f"  {file1_name} 独有: {sorted(keys1 - keys2)}")
    print(f"  {file2_name} 独有: {sorted(keys2 - keys1)}")
    
    # 比较共同变量的形状
    common_keys = keys1 & keys2
    print(f"\n共同变量的形状比较:")
    for key in sorted(common_keys):
        data1 = file1_data[key]
        data2 = file2_data[key]
        
        shape1 = data1.shape if hasattr(data1, 'shape') else 'N/A'
        shape2 = data2.shape if hasattr(data2, 'shape') else 'N/A'
        
        match = "一致" if shape1 == shape2 else "不一致"
        print(f"  {key}: {file1_name}={shape1}, {file2_name}={shape2} [{match}]")

if __name__ == '__main__':
    # 检查两个文件
    file1_path = r"E:\pycharm\PINN\PINN-main\S01.mat"
    file2_path = r"E:\pycharm\PINN\PINN-main\S02.mat"
    
    print("BCI-IV-2a数据集文件分析")
    
    # 检查文件是否存在
    for filepath in [file1_path, file2_path]:
        if not os.path.exists(filepath):
            print(f"文件不存在: {filepath}")
            exit(1)
    
    # 分析两个文件
    data1 = inspect_mat_file(file1_path)
    data2 = inspect_mat_file(file2_path)
    
    # 比较两个文件
    compare_files(data1, data2, "S01.mat", "S02.mat")
    
    print(f"\n{'='*60}")
    print("分析完成")
    print(f"{'='*60}")
