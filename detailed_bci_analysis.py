#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细分析BCI-IV-2a数据集的结构和内容
"""

import scipy.io as sio
import numpy as np
import os

def analyze_bci_structure(filepath):
    """详细分析BCI-IV-2a文件结构"""
    print(f"\n{'='*60}")
    print(f"详细分析: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    try:
        # 加载.mat文件
        mat_data = sio.loadmat(filepath, struct_as_record=False, squeeze_me=True)
        
        if 'data' in mat_data:
            data_array = mat_data['data']
            print(f"数据数组形状: {data_array.shape}")
            print(f"数据类型: {data_array.dtype}")
            print(f"包含 {len(data_array)} 个会话/运行")
            
            # 分析每个会话
            for i, session in enumerate(data_array):
                print(f"\n--- 会话/运行 {i+1} ---")
                
                # 检查会话的属性
                attrs = dir(session)
                relevant_attrs = [attr for attr in attrs if not attr.startswith('_')]
                print(f"属性: {relevant_attrs}")
                
                # 详细检查每个属性
                for attr in relevant_attrs:
                    try:
                        value = getattr(session, attr)
                        print(f"  {attr}: {type(value)}")
                        
                        if isinstance(value, np.ndarray):
                            print(f"    形状: {value.shape}")
                            print(f"    数据类型: {value.dtype}")
                            
                            # 特殊处理不同的字段
                            if attr == 'X':
                                print(f"    这是EEG数据矩阵")
                                print(f"    数值范围: [{np.min(value):.4f}, {np.max(value):.4f}]")
                                print(f"    均值: {np.mean(value):.4f}")
                                print(f"    标准差: {np.std(value):.4f}")
                                
                                # 分析数据形状
                                if value.shape[1] == 2:
                                    print(f"    可能是时间序列数据 (时间点 x 2)")
                                    print(f"    总时间点数: {value.shape[0]}")
                                    
                                    # 检查前几行数据
                                    print(f"    前5行数据:")
                                    print(f"      {value[:5]}")
                            
                            elif attr == 'trial':
                                print(f"    试次标记数组")
                                unique_trials = np.unique(value)
                                print(f"    试次范围: {unique_trials[:5]}...{unique_trials[-5:]}")
                                print(f"    总试次数: {len(unique_trials)}")
                                
                            elif attr == 'y':
                                print(f"    标签数组")
                                unique_labels, counts = np.unique(value, return_counts=True)
                                print(f"    标签类别: {unique_labels}")
                                print(f"    标签分布: {dict(zip(unique_labels, counts))}")
                                
                            elif attr == 'channels':
                                print(f"    通道信息")
                                if value.dtype.names:  # 如果是结构化数组
                                    print(f"    结构化数组字段: {value.dtype.names}")
                                else:
                                    print(f"    通道数据: {value}")
                                    
                            elif attr == 'classes':
                                print(f"    类别信息: {value}")
                                
                            elif attr == 'fs':
                                print(f"    采样频率: {value} Hz")
                                
                            elif attr == 'subject':
                                print(f"    被试编号: {value}")
                                
                            elif attr == 'session':
                                print(f"    会话编号: {value}")
                        
                        elif isinstance(value, str):
                            print(f"    字符串值: '{value}'")
                        elif isinstance(value, (int, float)):
                            print(f"    数值: {value}")
                        else:
                            print(f"    其他类型: {value}")
                            
                    except Exception as e:
                        print(f"  {attr}: 访问时出错 - {e}")
                
                # 只详细分析前3个会话，避免输出过长
                if i >= 2:
                    print(f"\n... (剩余 {len(data_array) - i - 1} 个会话结构类似)")
                    break
        
        return mat_data
        
    except Exception as e:
        print(f"分析文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_data_structure(data1, data2, name1, name2):
    """比较两个数据集的结构"""
    print(f"\n{'='*60}")
    print(f"比较 {name1} 和 {name2} 的数据结构")
    print(f"{'='*60}")
    
    if data1 is None or data2 is None:
        print("无法比较，因为某个文件读取失败")
        return
    
    # 比较会话数量
    sessions1 = len(data1['data'])
    sessions2 = len(data2['data'])
    print(f"会话数量: {name1}={sessions1}, {name2}={sessions2}")
    
    # 比较每个会话的结构
    for i in range(min(sessions1, sessions2, 3)):  # 只比较前3个会话
        print(f"\n--- 会话 {i+1} 比较 ---")
        
        session1 = data1['data'][i]
        session2 = data2['data'][i]
        
        # 比较属性
        attrs1 = set(dir(session1)) - set(dir(object))
        attrs2 = set(dir(session2)) - set(dir(object))
        
        common_attrs = attrs1 & attrs2
        
        for attr in sorted(common_attrs):
            if attr.startswith('_'):
                continue
                
            try:
                val1 = getattr(session1, attr)
                val2 = getattr(session2, attr)
                
                if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                    shape_match = "一致" if val1.shape == val2.shape else "不一致"
                    dtype_match = "一致" if val1.dtype == val2.dtype else "不一致"
                    print(f"  {attr}: 形状 {shape_match} ({val1.shape} vs {val2.shape}), 类型 {dtype_match}")
                    
                    # 对于标签，比较分布
                    if attr == 'y':
                        unique1, counts1 = np.unique(val1, return_counts=True)
                        unique2, counts2 = np.unique(val2, return_counts=True)
                        dist1 = dict(zip(unique1, counts1))
                        dist2 = dict(zip(unique2, counts2))
                        print(f"    标签分布: {name1}={dist1}, {name2}={dist2}")
                
                elif val1 == val2:
                    print(f"  {attr}: 相同 ({val1})")
                else:
                    print(f"  {attr}: 不同 ({val1} vs {val2})")
                    
            except Exception as e:
                print(f"  {attr}: 比较时出错 - {e}")

def extract_trial_structure(data, name):
    """分析试次结构"""
    print(f"\n{'='*60}")
    print(f"{name} 的试次结构分析")
    print(f"{'='*60}")
    
    if data is None:
        print("数据为空，无法分析")
        return
    
    data_array = data['data']
    
    for i, session in enumerate(data_array):
        print(f"\n--- 会话 {i+1} 试次分析 ---")
        
        try:
            X = getattr(session, 'X')
            trial = getattr(session, 'trial')
            y = getattr(session, 'y')
            
            print(f"X (EEG数据) 形状: {X.shape}")
            print(f"trial (试次标记) 形状: {trial.shape}")
            print(f"y (标签) 形状: {y.shape}")
            
            # 分析试次结构
            unique_trials = np.unique(trial)
            print(f"试次编号范围: {unique_trials[0]} 到 {unique_trials[-1]} (共{len(unique_trials)}个试次)")
            
            # 分析每个试次的长度
            trial_lengths = []
            for t in unique_trials[:10]:  # 只分析前10个试次
                trial_mask = trial == t
                trial_length = np.sum(trial_mask)
                trial_lengths.append(trial_length)
            
            print(f"前10个试次的长度: {trial_lengths}")
            if len(set(trial_lengths)) == 1:
                print(f"所有试次长度一致: {trial_lengths[0]} 个时间点")
                
                # 计算时间长度
                if hasattr(session, 'fs'):
                    fs = getattr(session, 'fs')
                    duration = trial_lengths[0] / fs
                    print(f"每个试次时长: {duration:.2f} 秒 (采样率: {fs} Hz)")
            
            # 分析标签分布
            unique_labels, counts = np.unique(y, return_counts=True)
            print(f"标签分布: {dict(zip(unique_labels, counts))}")
            
        except Exception as e:
            print(f"分析会话 {i+1} 时出错: {e}")
        
        # 只详细分析前2个会话
        if i >= 1:
            print(f"... (剩余会话结构类似)")
            break

if __name__ == '__main__':
    # 文件路径
    file1_path = r"E:\pycharm\PINN\PINN-main\S01.mat"
    file2_path = r"E:\pycharm\PINN\PINN-main\S02.mat"
    
    print("BCI-IV-2a数据集详细结构分析")
    
    # 检查文件
    for filepath in [file1_path, file2_path]:
        if not os.path.exists(filepath):
            print(f"文件不存在: {filepath}")
            exit(1)
    
    # 详细分析
    print("\n开始详细分析...")
    data1 = analyze_bci_structure(file1_path)
    data2 = analyze_bci_structure(file2_path)
    
    # 比较结构
    compare_data_structure(data1, data2, "S01", "S02")
    
    # 分析试次结构
    extract_trial_structure(data1, "S01")
    extract_trial_structure(data2, "S02")
    
    print(f"\n{'='*60}")
    print("详细分析完成")
    print(f"{'='*60}")























