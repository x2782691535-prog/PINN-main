import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

def calc_reconstruction_error(original_eeg, reconstructed_eeg):
    """
    计算原始EEG与重建EEG之间的误差
    
    参数:
        original_eeg: 原始EEG数据，形状为(batch_size, n_channels)
        reconstructed_eeg: 重建的EEG数据，形状为(batch_size, n_channels)
        
    返回:
        dict: 包含MSE误差和相关系数的字典
    """
    # 确保输入是PyTorch张量
    if not isinstance(original_eeg, torch.Tensor):
        original_eeg = torch.tensor(original_eeg)
    if not isinstance(reconstructed_eeg, torch.Tensor):
        reconstructed_eeg = torch.tensor(reconstructed_eeg)
    
    # 计算MSE误差
    mse = F.mse_loss(reconstructed_eeg, original_eeg).item()
    
    # 计算相关系数
    orig_mean = torch.mean(original_eeg, dim=1, keepdim=True)
    recon_mean = torch.mean(reconstructed_eeg, dim=1, keepdim=True)
    orig_std = torch.std(original_eeg, dim=1, keepdim=True)
    recon_std = torch.std(reconstructed_eeg, dim=1, keepdim=True)
    
    correlation = torch.mean(
        ((original_eeg - orig_mean) * (reconstructed_eeg - recon_mean)) / 
        (orig_std * recon_std + 1e-8)
    ).item()
    
    return {
        'mse': mse,
        'correlation': correlation
    }

def correlation_loss(x, y):
    """
    计算两组信号之间的相关性损失
    
    参数:
        x: 第一组信号，形状为(batch_size, n_features)
        y: 第二组信号，形状为(batch_size, n_features)
        
    返回:
        相关性损失 (1 - 平均相关系数)
    """
    # 计算每个样本的均值
    mean_x = torch.mean(x, dim=1, keepdim=True)
    mean_y = torch.mean(y, dim=1, keepdim=True)
    
    # 计算去均值数据
    xm = x - mean_x
    ym = y - mean_y
    
    # 计算标准差
    std_x = torch.std(x, dim=1, keepdim=True) + 1e-8
    std_y = torch.std(y, dim=1, keepdim=True) + 1e-8
    
    # 计算相关系数
    r = torch.mean(
        torch.sum(xm * ym, dim=1) / 
        (torch.sum(torch.pow(xm, 2), dim=1).sqrt() * 
         torch.sum(torch.pow(ym, 2), dim=1).sqrt() + 1e-8)
    )
    
    # 返回相关性损失 (1 - 相关系数)
    return 1.0 - r

def source_sparsity_loss(source_activations, l1_weight=0.005):
    """
    计算源空间激活的稀疏性损失
    
    参数:
        source_activations: 源空间激活，形状为(batch_size, n_sources)
        l1_weight: L1正则化权重
        
    返回:
        稀疏性损失
    """
    # L1正则化项
    l1_loss = torch.mean(torch.abs(source_activations))
    
    # 计算激活的聚集性 (鼓励激活集中在少数源点)
    # 使用L2与L1范数的比率作为稀疏性度量
    l2_norm = torch.norm(source_activations, p=2, dim=1)
    l1_norm = torch.norm(source_activations, p=1, dim=1)
    concentration_loss = torch.mean(l1_norm / (l2_norm + 1e-8))
    
    # 组合损失
    return l1_weight * l1_loss + (1 - l1_weight) * concentration_loss

def poisson_equation_loss(source_activations, lead_field, alpha=0.2):
    """
    计算泊松方程约束损失，确保源激活符合泊松方程的物理约束
    ∇²V = -ρ/ε (电位的拉普拉斯等于负的电荷密度除以介电常数)
    
    参数:
        source_activations: 源空间激活，形状为(batch_size, n_sources)
        lead_field: 引线场矩阵，形状为(n_channels, n_sources)
        alpha: 泊松方程约束权重，默认为0.2
        
    返回:
        泊松方程约束损失
    """
    batch_size = source_activations.shape[0]
    n_sources = source_activations.shape[1]
    
    # 1. 基本物理一致性约束: 正向映射与反向映射应保持一致
    # 计算从源空间到电极空间的投影，然后再投影回源空间
    electrode_potentials = torch.matmul(source_activations, lead_field.t())
    back_projected = torch.matmul(electrode_potentials, lead_field)
    consistency_loss = F.mse_loss(back_projected, source_activations)
    
    # 2. 泊松方程数学特性约束: 空间平滑性与局部相关性
    # 使用差分近似代替自动微分，确保数值稳定性
    # 计算"拉普拉斯算子"的离散近似 - 每个源点与相邻源点的差异
    # 对一维索引采用循环填充处理
    laplacian_approx = (
        2 * source_activations 
        - torch.roll(source_activations, shifts=1, dims=1) 
        - torch.roll(source_activations, shifts=-1, dims=1)
    )
    
    # 源密度分布应与拉普拉斯近似成比例
    # 在泊松方程中: ∇²V = -ρ/ε
    density_prop = -source_activations / 0.3  # 电导率约为0.3 S/m
    poisson_approx_loss = F.mse_loss(laplacian_approx, density_prop)
    
    # 3. 电荷守恒约束: 源密度积分应接近零
    conservation_loss = torch.abs(torch.sum(source_activations, dim=1)).mean()
    
    # 4. 边界条件约束: 激活应该主要集中在中心区域
    edge_mask = torch.zeros_like(source_activations)
    edge_size = n_sources // 6
    edge_mask[:, :edge_size] = 1.0
    edge_mask[:, -edge_size:] = 1.0
    boundary_loss = torch.mean(torch.abs(source_activations * edge_mask))
    
    # 5. 稀疏性约束: 激活应该集中而非分散
    l1_loss = torch.mean(torch.abs(source_activations))
    
    # 组合所有物理约束
    physics_loss = (
        0.4 * consistency_loss +
        0.3 * poisson_approx_loss +
        0.1 * conservation_loss +
        0.1 * boundary_loss +
        0.1 * l1_loss
    )
    
    return physics_loss

def calc_localization_error(source_activations, true_source_mask=None, source_positions=None):
    """
    计算源定位误差指标
    
    参数:
        source_activations: 源空间激活，形状为(batch_size, n_sources) 或 (n_sources,) 如果是单一样本
        true_source_mask: 真实的源激活掩码，形状为(batch_size, n_sources) 或 (n_sources,)
                            如果提供，将用于计算更准确的LE和AUPRC。
        source_positions: 源点的3D坐标，形状为(n_sources, 3)。如果提供，用于计算LE(mm)。
        
    返回:
        dict: 包含LE(mm)和AUPRC的字典 (如果提供了true_source_mask和source_positions)
              否则，返回基于原始方法的指标。
    """
    # 如果输入是PyTorch张量，转换为NumPy数组
    if isinstance(source_activations, torch.Tensor):
        source_activations = source_activations.detach().cpu().numpy()
    if isinstance(true_source_mask, torch.Tensor):
        true_source_mask = true_source_mask.detach().cpu().numpy()
    if isinstance(source_positions, torch.Tensor):
        source_positions = source_positions.detach().cpu().numpy()

    # 确保source_activations是二维的 (batch_size, n_sources)
    if source_activations.ndim == 1:
        source_activations = source_activations.reshape(1, -1)
    if true_source_mask is not None and true_source_mask.ndim == 1:
        true_source_mask = true_source_mask.reshape(1, -1)

    abs_activations = np.abs(source_activations)
    batch_size, n_sources = abs_activations.shape

    le_mm_values = []
    auprc_values = []

    if true_source_mask is not None and source_positions is not None:
        if true_source_mask.shape[0] != batch_size or true_source_mask.shape[1] != n_sources:
            raise ValueError("true_source_mask 维度与 source_activations 不匹配")
        if source_positions.shape[0] != n_sources or source_positions.shape[1] != 3:
            raise ValueError("source_positions 维度不正确")

        for i in range(batch_size):
            pred_acts_sample = abs_activations[i]
            true_mask_sample = true_source_mask[i]

            # LE(mm) 计算
            # 预测的源中心：最大激活点
            pred_center_idx = np.argmax(pred_acts_sample)
            pred_center_pos = source_positions[pred_center_idx]
            
            # 真实的源中心：真实激活区域的几何中心
            true_active_indices = np.where(true_mask_sample)[0]
            if len(true_active_indices) > 0:
                true_center_pos = np.mean(source_positions[true_active_indices], axis=0)
                distance = np.sqrt(np.sum((pred_center_pos - true_center_pos)**2))
                le_mm_values.append(distance)
            else: # 如果真实掩码中没有激活的源（不应发生，但作为保护）
                le_mm_values.append(np.nan) # 或者一个大的惩罚值

            # AUPRC 计算
            # pred_probs: 预测的激活值（已取绝对值并归一化）
            pred_probs_sample = pred_acts_sample / (np.max(pred_acts_sample) + 1e-8)
            # true_labels_binary: 真实的二元标签
            true_labels_binary_sample = true_mask_sample.astype(float)
            
            if np.sum(true_labels_binary_sample) == 0: # 如果没有正样本
                auprc_values.append(0.0) # 或者 np.nan，取决于如何处理这种情况
                continue
            if np.sum(true_labels_binary_sample) == n_sources: # 如果所有样本都是正样本
                auprc_values.append(1.0) # 或者 np.nan
                continue

            # 使用sklearn计算AUPRC以获得更精确的结果
            try:
                from sklearn.metrics import average_precision_score
                auprc = average_precision_score(true_labels_binary_sample, pred_probs_sample)
                auprc_values.append(auprc)
            except ImportError:
                # Fallback to manual calculation if sklearn is not available (简化版)
                sorted_indices = np.argsort(pred_probs_sample)[::-1]
                true_positives = np.cumsum(true_labels_binary_sample[sorted_indices])
                precisions = true_positives / np.arange(1, n_sources + 1)
                recalls = true_positives / (np.sum(true_labels_binary_sample) + 1e-8)
                auprc = 0
                for k in range(1, len(recalls)):
                    auprc += 0.5 * (precisions[k] + precisions[k-1]) * (recalls[k] - recalls[k-1])
                auprc_values.append(auprc)
        
        final_le_mm = np.nanmean(le_mm_values) if le_mm_values else np.nan
        final_auprc = np.nanmean(auprc_values) if auprc_values else np.nan

        return {
            'le_mm': final_le_mm,
            'auprc': final_auprc
        }
    else:
        # 如果没有提供ground truth，执行原始的基于统计的指标计算 (作为回退)
        # 这部分代码与之前的版本类似，但现在仅作为回退
        peak_values = np.max(abs_activations, axis=1)
        mean_values = np.mean(abs_activations, axis=1)
        peak_mean_ratio = np.mean(peak_values / (mean_values + 1e-8))
        
        threshold_stat = 0.3 * np.max(abs_activations, axis=1, keepdims=True)
        active_sources_stat = np.mean((abs_activations > threshold_stat).sum(axis=1) / n_sources)
        
        normalized_activations_stat = abs_activations / (np.sum(abs_activations, axis=1, keepdims=True) + 1e-8)
        entropy_stat = -np.mean(np.sum(normalized_activations_stat * np.log(normalized_activations_stat + 1e-8), axis=1))
        
        # 返回原始指标，标记它们是基于统计的
        return {
            'peak_mean_ratio_stat': peak_mean_ratio,
            'active_source_ratio_stat': active_sources_stat,
            'activation_entropy_stat': entropy_stat,
            'le_mm': np.nan, # 明确表示无法计算
            'auprc': np.nan   # 明确表示无法计算
        }

def evaluate_source_localization(model, test_loader, source_positions, num_samples=None, use_simulated_data=False):
    """
    评估模型在测试集上的源定位性能
    
    参数:
        model: 训练好的PINN模型
        test_loader: 测试数据加载器
        source_positions: 源点3D坐标 (n_sources, 3)，用于LE(mm)计算
        num_samples: 要评估的样本数量 (None则评估整个测试集)
        use_simulated_data: 指示test_loader是否包含模拟数据的ground truth
        
    返回:
        dict: 包含源定位评估结果的字典
    """
    device = next(model.parameters()).device
    model.eval()
    
    all_localization_metrics = []
    collected_original_eeg = []
    collected_reconstructed_eeg = []
    
    with torch.no_grad():
        for i, data_batch in enumerate(test_loader):
            if num_samples is not None and i >= num_samples:
                break
            
            if use_simulated_data:
                eeg_data, _, true_source_mask_batch = data_batch
                # true_source_mask_batch: (batch, n_sources)
            else:
                eeg_data, _ = data_batch # 忽略真实数据的标签 (现在为空)
                true_source_mask_batch = None
                
            eeg_data = eeg_data.to(device)
            outputs = model(eeg_data)

            # 收集原始EEG
            collected_original_eeg.append(eeg_data.cpu().numpy())
            
            # 收集重建EEG (如果可用)
            if 'reconstructed_eeg' in outputs and outputs['reconstructed_eeg'] is not None:
                collected_reconstructed_eeg.append(outputs['reconstructed_eeg'].cpu().numpy())
            
            if hasattr(model, 'source_fc') and 'source_activations' in outputs and outputs['source_activations'] is not None:
                pred_source_activations = outputs['source_activations'] # (batch, n_sources)
                
                # 即使是真实数据，也传入source_positions，但true_source_mask为None
                loc_metrics_batch = calc_localization_error(pred_source_activations, 
                                                            true_source_mask=true_source_mask_batch,
                                                            source_positions=source_positions)
                all_localization_metrics.append(loc_metrics_batch)

    # 在循环后合并收集的数据
    if collected_original_eeg:
        all_original_eeg_np = np.concatenate(collected_original_eeg, axis=0)
    else:
        all_original_eeg_np = np.array([])

    if collected_reconstructed_eeg:
        all_reconstructed_eeg_np = np.concatenate(collected_reconstructed_eeg, axis=0)
    else:
        all_reconstructed_eeg_np = np.array([])
    
    # 计算平均结果
    if not all_localization_metrics:
        return {'le_mm': np.nan, 'auprc': np.nan, 'error': 'No localization metrics calculated.'}

    # 根据是否有ground truth来平均不同指标
    if use_simulated_data:
        avg_le_mm = np.nanmean([m.get('le_mm', np.nan) for m in all_localization_metrics])
        avg_auprc = np.nanmean([m.get('auprc', np.nan) for m in all_localization_metrics])
        avg_loc_metrics = {
            'le_mm': avg_le_mm,
            'auprc': avg_auprc
        }
    else:
        # 对于真实数据，我们只报告统计指标（如果calc_localization_error返回了它们）
        # 或者，如果calc_localization_error在没有ground truth时返回nan，这里也会是nan
        # avg_le_mm = np.nanmean([m.get('le_mm', np.nan) for m in all_localization_metrics]) # 应该是nan
        # avg_auprc = np.nanmean([m.get('auprc', np.nan) for m in all_localization_metrics]) # 应该是nan
        avg_peak_mean_ratio = np.nanmean([m.get('peak_mean_ratio_stat', np.nan) for m in all_localization_metrics])
        avg_active_source_ratio = np.nanmean([m.get('active_source_ratio_stat', np.nan) for m in all_localization_metrics])
        avg_activation_entropy = np.nanmean([m.get('activation_entropy_stat', np.nan) for m in all_localization_metrics])
        avg_loc_metrics = {
            'le_mm': np.nan,
            'auprc': np.nan,
            'peak_mean_ratio_stat': avg_peak_mean_ratio,
            'active_source_ratio_stat': avg_active_source_ratio,
            'activation_entropy_stat': avg_activation_entropy
        }

    # 保存评估结果 (简化，只保存主要指标)
    results_dir = os.path.join('results', 'source_localization')
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, 'evaluation_report_detailed.txt')

    with open(report_path, 'w') as f:
        f.write("===== 源定位评估 =====\n")
        if use_simulated_data:
            f.write(f"模式: 模拟数据 (Ground Truth评估)\n")
            f.write(f"LE(mm): {avg_loc_metrics.get('le_mm', 'N/A'):.4f}\n")
            f.write(f"AUPRC: {avg_loc_metrics.get('auprc', 'N/A'):.4f}\n")
        else:
            f.write("模式: 真实数据 (无Ground Truth，统计指标)\n")
            f.write(f"LE(mm) (无法计算): {avg_loc_metrics.get('le_mm', 'N/A')}\n") # 应为nan
            f.write(f"AUPRC (无法计算): {avg_loc_metrics.get('auprc', 'N/A')}\n")   # 应为nan
            f.write(f"统计 - 峰均比: {avg_loc_metrics.get('peak_mean_ratio_stat', 'N/A'):.4f}\n")
            f.write(f"统计 - 活跃源比例: {avg_loc_metrics.get('active_source_ratio_stat', 'N/A'):.4f}\n")
            f.write(f"统计 - 激活熵: {avg_loc_metrics.get('activation_entropy_stat', 'N/A'):.4f}\n")
        f.write("\n详细批次指标:\n")
        for i, m in enumerate(all_localization_metrics):
            f.write(f"  Batch {i}: {m}\n")
            
    print(f"详细源定位评估报告已保存至: {report_path}")
    
    # Reconstruction error calculation
    recon_errors = {'mse': np.nan, 'correlation': np.nan}
    if all_original_eeg_np.size > 0 and all_reconstructed_eeg_np.size > 0:
        if all_original_eeg_np.shape[0] == all_reconstructed_eeg_np.shape[0] and \
           all_original_eeg_np.ndim == all_reconstructed_eeg_np.ndim and \
           (all_original_eeg_np.ndim == 2 and all_original_eeg_np.shape[1] == all_reconstructed_eeg_np.shape[1] or \
            all_original_eeg_np.ndim > 2 and all_original_eeg_np.shape[1:] == all_reconstructed_eeg_np.shape[1:]): # More robust shape check
            
            num_recon_samples = min(all_original_eeg_np.shape[0], 100) 
            # Ensure inputs to calc_reconstruction_error are 2D (batch_size, n_features) or handle ND if necessary
            # Assuming calc_reconstruction_error expects (batch, n_channels*n_times) or similar
            # For now, assuming it can handle the direct output shape if consistent
            
            # If original_eeg and reconstructed_eeg are (batch, channels, time), 
            # calc_reconstruction_error might need them reshaped or to average over time.
            # For simplicity, let's ensure they are 2D for calc_reconstruction_error
            # This might need adjustment based on how calc_reconstruction_error expects input
            
            # Assuming calc_reconstruction_error expects (batch_size, n_features)
            # If data is (batch, channels, time), we might need to flatten the last two dims
            # or average over one of them if calc_reconstruction_error is not designed for 3D+
            
            # Sticking to current calc_reconstruction_error which seems to expect (batch, n_channels) or similar for MSE
            # If original_eeg_np is (batch, channels, time), we need to decide how to pass it.
            # For now, let's assume if it's 3D, we take the mean over the time dimension (axis 2)
            # to make it (batch, channels) like the function calc_reconstruction_error might expect for MSE.
            
            original_for_recon = all_original_eeg_np[:num_recon_samples]
            reconstructed_for_recon = all_reconstructed_eeg_np[:num_recon_samples]

            if original_for_recon.ndim == 3 and reconstructed_for_recon.ndim == 3: # (batch, channels, time)
                 # Assuming calc_reconstruction_error is designed for 2D (batch, features)
                 # Reshape: (batch, channels * time) or take mean over time (batch, channels)
                 # Option 1: Mean over time
                 # original_for_recon = np.mean(original_for_recon, axis=2)
                 # reconstructed_for_recon = np.mean(reconstructed_for_recon, axis=2)
                 # Option 2: Reshape (less common for direct MSE unless features are concatenated timepoints)
                 # original_for_recon = original_for_recon.reshape(original_for_recon.shape[0], -1)
                 # reconstructed_for_recon = reconstructed_for_recon.reshape(reconstructed_for_recon.shape[0], -1)
                 # For now, let calc_reconstruction_error handle it; it uses torch.mean(dim=1) for correlation
                 # which implies input is (batch, features).
                 # If original_eeg is (batch, channels, time) and calc_reconstruction_error takes (batch, features)
                 # it will fail. calc_reconstruction_error uses F.mse_loss directly on inputs.
                 # F.mse_loss can handle ND tensors. So, we might not need to reshape here.
                 pass


            recon_errors = calc_reconstruction_error(
                original_for_recon,
                reconstructed_for_recon
            )
        else:
            print(f"警告: 原始EEG和重建EEG的形状不匹配或为空，无法计算重建误差. "
                  f"原始形状: {all_original_eeg_np.shape}, 重建形状: {all_reconstructed_eeg_np.shape}")
    elif all_original_eeg_np.size > 0 and all_reconstructed_eeg_np.size == 0:
        print("警告: 收集到原始EEG数据，但没有可用的重建EEG数据用于计算误差。")
    
    # Correctly calculate final metrics by averaging from the list of batch metrics
    if not all_localization_metrics: # Should be handled earlier, but as a safeguard
        final_le_mm = np.nan
        final_auprc = np.nan
        final_composite_score = np.nan
        final_peak_mean_ratio = np.nan
        final_active_source_ratio = np.nan
        final_activation_entropy = np.nan
    else:
        final_le_mm = np.nanmean([m.get('le_mm', np.nan) for m in all_localization_metrics])
        final_auprc = np.nanmean([m.get('auprc', np.nan) for m in all_localization_metrics])
        # Note: composite_localization_score might be calculated once on aggregated stats in calc_localization_error
        # or per batch. If per batch, averaging here is correct.
        # If calc_localization_error returns it once for the whole dataset (e.g. when true_source_mask is None and stats are global)
        # then all_localization_metrics would ideally be a single dict, not a list.
        # Given current structure where all_localization_metrics is a list of dicts (one per batch), averaging is the way.
        final_composite_score = np.nanmean([m.get('composite_localization_score', np.nan) for m in all_localization_metrics])
        final_peak_mean_ratio = np.nanmean([m.get('peak_mean_ratio_stat', np.nan) for m in all_localization_metrics])
        final_active_source_ratio = np.nanmean([m.get('active_source_ratio_stat', np.nan) for m in all_localization_metrics])
        final_activation_entropy = np.nanmean([m.get('activation_entropy_stat', np.nan) for m in all_localization_metrics])

    final_metrics_to_return = {
        'le_mm': final_le_mm,
        'auprc': final_auprc,
        'composite_localization_score': final_composite_score,
        'peak_mean_ratio_stat': final_peak_mean_ratio,
        'active_source_ratio_stat': final_active_source_ratio,
        'activation_entropy_stat': final_activation_entropy,
        'mse': recon_errors.get('mse', np.nan),
        'correlation': recon_errors.get('correlation', np.nan)
    }
    
    return final_metrics_to_return

# 保存最终指标的辅助函数 (如果需要跨被试聚合)
# ... existing code ... 