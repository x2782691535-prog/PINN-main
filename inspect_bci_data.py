#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门检查BCI-IV-2a数据集的复杂.mat文件结构
"""

import scipy.io as sio
import numpy as np
import os

def explore_nested_structure(data, name="", level=0, max_level=3):
    """递归探索嵌套数据结构"""
    indent = "  " * level
    
    if level > max_level:
        print(f"{indent}[达到最大探索深度]")
        return
    
    print(f"{indent}{name}: {type(data)}")
    
    if isinstance(data, np.ndarray):
        print(f"{indent}  形状: {data.shape}")
        print(f"{indent}  数据类型: {data.dtype}")
        
        if data.dtype == 'object':
            print(f"{indent}  对象数组内容:")
            # 探索对象数组的内容
            if data.size > 0:
                for i in range(min(3, data.size)):  # 只看前3个元素
                    if data.ndim == 1:
                        item = data[i]
                    elif data.ndim == 2:
                        row, col = np.unravel_index(i, data.shape)
                        item = data[row, col]
                    else:
                        continue
                    
                    print(f"{indent}    [{i}]: {type(item)}")
                    if hasattr(item, 'shape'):
                        print(f"{indent}        形状: {item.shape}")
                        print(f"{indent}        数据类型: {item.dtype}")
                        
                        # 如果是数值数组，显示统计信息
                        if isinstance(item, np.ndarray) and np.issubdtype(item.dtype, np.number):
                            if item.size > 0:
                                print(f"{indent}        数值范围: [{np.min(item):.4f}, {np.max(item):.4f}]")
                                print(f"{indent}        均值: {np.mean(item):.4f}")
                                
                                # 如果是3D数组，可能是EEG数据
                                if item.ndim == 3:
                                    print(f"{indent}        可能是EEG数据 (试次, 通道, 时间点)")
                                    n_trials, n_channels, n_timepoints = item.shape
                                    print(f"{indent}        试次数: {n_trials}")
                                    print(f"{indent}        通道数: {n_channels}")
                                    print(f"{indent}        时间点数: {n_timepoints}")
                                    
                                    # 分析采样率
                                    if n_timepoints == 1000:
                                        print(f"{indent}        推测采样率: 250Hz (4秒数据)")
                                    elif n_timepoints == 875:
                                        print(f"{indent}        推测采样率: 250Hz (3.5秒数据)")
                    
                    elif hasattr(item, '__len__') and not isinstance(item, str):
                        print(f"{indent}        长度: {len(item)}")
                    
                    # 递归探索
                    if level < max_level:
                        explore_nested_structure(item, f"元素[{i}]", level + 1, max_level)
        
        elif np.issubdtype(data.dtype, np.number):
            if data.size > 0:
                print(f"{indent}  数值范围: [{np.min(data):.4f}, {np.max(data):.4f}]")
                print(f"{indent}  均值: {np.mean(data):.4f}")
                print(f"{indent}  标准差: {np.std(data):.4f}")

def inspect_bci_file(filepath):
    """检查BCI-IV-2a .mat文件"""
    print(f"\n{'='*60}")
    print(f"检查文件: {os.path.basename(filepath)}")
    print(f"文件大小: {os.path.getsize(filepath) / (1024*1024):.2f} MB")
    print(f"{'='*60}")
    
    try:
        # 加载.mat文件，不进行结构数组转换
        mat_data = sio.loadmat(filepath, struct_as_record=False, squeeze_me=True)
        
        print(f"\n顶层变量:")
        for key, value in mat_data.items():
            if not key.startswith('__'):
                print(f"  {key}: {type(value)}")
                if hasattr(value, 'shape'):
                    print(f"    形状: {value.shape}")
                if hasattr(value, 'dtype'):
                    print(f"    数据类型: {value.dtype}")
        
        # 详细探索主要数据
        main_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        
        for key in main_keys:
            print(f"\n{'='*40}")
            print(f"探索变量: {key}")
            print(f"{'='*40}")
            explore_nested_structure(mat_data[key], key, 0, max_level=4)
        
        return mat_data
        
    except Exception as e:
        print(f"读取文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_eeg_data_info(mat_data, subject_name):
    """尝试提取EEG数据信息"""
    print(f"\n尝试提取 {subject_name} 的EEG数据信息:")
    
    try:
        # BCI-IV-2a数据通常存储在'data'字段中
        if 'data' in mat_data:
            data_field = mat_data['data']
            print(f"data字段类型: {type(data_field)}")
            
            if isinstance(data_field, np.ndarray):
                print(f"data数组形状: {data_field.shape}")
                
                # 如果是对象数组，尝试访问其内容
                if data_field.dtype == 'object':
                    print("这是一个对象数组，尝试访问其内容...")
                    
                    # 尝试不同的访问方式
                    for i in range(min(data_field.size, 10)):
                        try:
                            if data_field.ndim == 1:
                                item = data_field[i]
                            elif data_field.ndim == 2:
                                row, col = np.unravel_index(i, data_field.shape)
                                item = data_field[row, col]
                            else:
                                continue
                            
                            print(f"  对象[{i}]: {type(item)}")
                            
                            # 检查是否有常见的BCI-IV-2a字段
                            if hasattr(item, '__dict__'):
                                attrs = dir(item)
                                relevant_attrs = [attr for attr in attrs if not attr.startswith('_')]
                                print(f"    属性: {relevant_attrs}")
                                
                                # 查找可能的EEG数据字段
                                for attr in ['X', 'data', 'trial', 'rawdata']:
                                    if hasattr(item, attr):
                                        eeg_data = getattr(item, attr)
                                        if isinstance(eeg_data, np.ndarray):
                                            print(f"    找到EEG数据字段 '{attr}': {eeg_data.shape}")
                                            
                                            if eeg_data.ndim == 3:
                                                n_trials, n_channels, n_timepoints = eeg_data.shape
                                                print(f"      试次数: {n_trials}")
                                                print(f"      通道数: {n_channels}")
                                                print(f"      时间点数: {n_timepoints}")
                                                
                                                # 显示数据统计
                                                print(f"      数值范围: [{np.min(eeg_data):.4f}, {np.max(eeg_data):.4f}]")
                                                print(f"      均值: {np.mean(eeg_data):.4f}")
                                                print(f"      标准差: {np.std(eeg_data):.4f}")
                                
                                # 查找标签字段
                                for attr in ['y', 'label', 'labels']:
                                    if hasattr(item, attr):
                                        labels = getattr(item, attr)
                                        if isinstance(labels, np.ndarray):
                                            print(f"    找到标签字段 '{attr}': {labels.shape}")
                                            unique_labels, counts = np.unique(labels, return_counts=True)
                                            print(f"      唯一标签: {unique_labels}")
                                            print(f"      标签分布: {dict(zip(unique_labels, counts))}")
                            
                            elif isinstance(item, np.ndarray):
                                print(f"    数组形状: {item.shape}")
                                if item.ndim >= 2:
                                    print(f"    可能包含EEG数据")
                            
                        except Exception as e:
                            print(f"  访问对象[{i}]时出错: {e}")
                            continue
            
    except Exception as e:
        print(f"提取EEG数据信息时出错: {e}")

if __name__ == '__main__':
    # 检查两个文件
    file1_path = r"E:\pycharm\PINN\PINN-main\S01.mat"
    file2_path = r"E:\pycharm\PINN\PINN-main\S02.mat"
    
    print("BCI-IV-2a数据集深度分析")
    
    # 检查文件是否存在
    for filepath in [file1_path, file2_path]:
        if not os.path.exists(filepath):
            print(f"文件不存在: {filepath}")
            exit(1)
    
    # 分析两个文件
    print("\n开始分析 S01.mat...")
    data1 = inspect_bci_file(file1_path)
    if data1:
        extract_eeg_data_info(data1, "S01")
    
    print("\n开始分析 S02.mat...")
    data2 = inspect_bci_file(file2_path)
    if data2:
        extract_eeg_data_info(data2, "S02")
    
    print(f"\n{'='*60}")
    print("深度分析完成")
    print(f"{'='*60}")























