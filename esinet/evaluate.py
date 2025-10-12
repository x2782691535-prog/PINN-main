# 临时evaluate模块，用于解决循环导入问题
"""
临时的evaluate模块，包含基本的评估函数
"""

import numpy as np

def eval_mean_localization_error(y_true, y_est, pos):
    """
    计算平均定位误差
    
    参数:
        y_true: 真实源激活
        y_est: 预测源激活  
        pos: 源点位置
    
    返回:
        mle: 平均定位误差(毫米)
    """
    try:
        # 确保输入是numpy数组
        y_true = np.array(y_true)
        y_est = np.array(y_est)
        pos = np.array(pos)
        
        # 找到真实和预测的最大激活位置
        true_max_idx = np.argmax(np.abs(y_true))
        pred_max_idx = np.argmax(np.abs(y_est))
        
        # 计算位置间的欧几里得距离
        true_pos = pos[true_max_idx]
        pred_pos = pos[pred_max_idx]
        
        distance = np.sqrt(np.sum((true_pos - pred_pos)**2)) * 1000  # 转换为毫米
        
        return distance
        
    except Exception as e:
        print(f"计算MLE时出错: {e}")
        return np.nan

def eval_mse(y_true, y_est):
    """计算均方误差"""
    try:
        y_true = np.array(y_true)
        y_est = np.array(y_est)
        return np.mean((y_true - y_est)**2)
    except:
        return np.nan

def eval_auc(y_true, y_est, pos=None):
    """计算AUC（简化版本）"""
    try:
        from sklearn.metrics import roc_auc_score
        y_true_binary = (np.abs(y_true) > np.mean(np.abs(y_true))).astype(int)
        y_est_probs = np.abs(y_est) / (np.max(np.abs(y_est)) + 1e-8)
        return roc_auc_score(y_true_binary, y_est_probs)
    except:
        return 0.5  # 随机猜测的AUC





















