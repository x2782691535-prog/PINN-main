#!/usr/bin/env python3
"""
传统源定位方法脚本 - MNE和eLORETA
用于对真实数据集进行传统源定位方法的对比实验

数据集：E:\pycharm\PINN\PINN-main\源定位真实数据集
该数据集包含7个被试的高密度EEG数据和颅内电刺激ground truth

支持的方法：
- MNE (Minimum Norm Estimate)
- eLORETA (exact Low Resolution Electromagnetic Tomography)

评估指标：
- MLE (Mean Localization Error)
- AUC (Area Under Curve)
"""

import numpy as np
import os
import sys
import mne
import time
import json
import pickle
from pathlib import Path
import pandas as pd
from scipy.spatial.distance import cdist
import contextlib

# 添加esinet路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'esinet'))

# 导入esinet模块
try:
    from esinet import util
    from esinet.util.util import source_to_sourceEstimate
    from esinet.evaluate import eval_mean_localization_error, eval_mse, eval_auc
    print("esinet模块导入成功")
except ImportError as e:
    print(f"esinet导入失败: {e}")
    sys.exit(1)

@contextlib.contextmanager
def suppress_stdout():
    """抑制stdout输出"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# 数据集路径
DATASET_PATH = r"E:\pycharm\PINN\PINN-main\源定位真实数据集"

class TraditionalSourceLocalizer:
    """使用传统方法（MNE和eLORETA）进行源定位的类"""
    
    def __init__(self, dataset_path=DATASET_PATH):
        self.dataset_path = Path(dataset_path)
        self.subjects = [f"sub-{i:02d}" for i in range(1, 8)]  # sub-01 到 sub-07
        self.eeg_data = {}
        self.ground_truth_positions = {}
        self.forward_models = {}
        self.electrode_positions = {}
        self.results = {}
        
        # 创建结果保存目录
        self.results_dir = Path("traditional_source_localization_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def load_forward_model(self, subject):
        """加载前向模型"""
        fwd_path = self.dataset_path / "derivatives" / "sourcemodelling" / subject / "fwd" / f"{subject}_fwd.fif"
        if fwd_path.exists():
            try:
                fwd = mne.read_forward_solution(str(fwd_path))
                # 确保前向模型使用固定方向
                fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True, verbose=False)
                print(f"成功加载 {subject} 前向模型: {fwd['nsource']} 个源点")
                return fwd
            except Exception as e:
                print(f"加载 {subject} 前向模型失败: {e}")
                return None
        else:
            print(f"前向模型文件不存在: {fwd_path}")
            return None
    
    def load_all_eeg_runs(self, subject):
        """加载所有run的EEG数据"""
        eeg_dir = self.dataset_path / "derivatives" / "epochs" / subject / "eeg"
        
        # 查找所有run的epochs文件
        epochs_files = sorted(list(eeg_dir.glob(f"{subject}_task-seegstim_run-*_epochs.npy")))
        
        if not epochs_files:
            print(f"未找到 {subject} 的EEG数据文件")
            return None
        
        # 加载通道信息（所有run使用相同的通道）
        channels_file = epochs_files[0].parent / epochs_files[0].name.replace('_epochs.npy', '_channels.tsv')
        try:
            channels_df = pd.read_csv(channels_file, sep='\t')
        except Exception as e:
            print(f"加载通道信息失败: {e}")
            return None
        
        # 加载所有run的数据
        all_runs_data = []
        for epochs_file in epochs_files:
            try:
                epochs_data = np.load(epochs_file)
                run_number = epochs_file.stem.split('run-')[1].split('_')[0]
                all_runs_data.append({
                    'run': int(run_number),
                    'data': epochs_data,
                    'channels': channels_df
                })
                print(f"  成功加载 run-{run_number}: {epochs_data.shape}")
            except Exception as e:
                print(f"  加载 {epochs_file} 失败: {e}")
                continue
        
        if not all_runs_data:
            print(f"未能加载 {subject} 的任何run数据")
            return None
        
        print(f"成功加载 {subject} 的 {len(all_runs_data)} 个run")
        return all_runs_data
    
    def load_ground_truth_positions(self, subject):
        """加载ground truth电极位置"""
        ieeg_dir = self.dataset_path / "derivatives" / "epochs" / subject / "ieeg"
        
        # 加载surface空间的电极位置
        electrodes_file = ieeg_dir / f"{subject}_task-seegstim_space-surface_electrodes.tsv"
        
        if electrodes_file.exists():
            try:
                electrodes_df = pd.read_csv(electrodes_file, sep='\t')
                positions = electrodes_df[['x', 'y', 'z']].values
                names = electrodes_df['name'].values
                print(f"成功加载 {subject} ground truth位置: {len(positions)} 个电极")
                return positions, names
            except Exception as e:
                print(f"加载 {subject} ground truth位置失败: {e}")
                return None, None
        else:
            print(f"Ground truth文件不存在: {electrodes_file}")
            return None, None
    
    def create_epochs_object(self, eeg_data, channels_df, sfreq=1000):
        """创建MNE Epochs对象"""
        try:
            # 获取通道名称
            ch_names = channels_df['name'].tolist()
            n_channels = len(ch_names)
            n_trials, n_samples = eeg_data.shape[0], eeg_data.shape[-1]
            
            # 确保数据维度正确
            if eeg_data.ndim == 2:
                # 如果是2D，假设是 (trials, samples)，需要添加通道维度
                eeg_data = eeg_data[:, np.newaxis, :]
                n_channels = 1
                ch_names = ch_names[:1]
            elif eeg_data.ndim == 3:
                # 3D数据 (trials, channels, samples)
                n_channels = min(eeg_data.shape[1], len(ch_names))
                ch_names = ch_names[:n_channels]
                eeg_data = eeg_data[:, :n_channels, :]
            
            # 创建info对象
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
            
            # 创建事件数组
            events = np.zeros((n_trials, 3), dtype=int)
            events[:, 0] = np.arange(n_trials) * n_samples  # 时间戳
            events[:, 2] = 1  # 事件类型
            
            # 创建Epochs对象
            epochs = mne.EpochsArray(
                eeg_data, info, events=events, 
                event_id={'stimulation': 1}, 
                tmin=0.0, verbose=False
            )
            
            # 应用基线校正
            with suppress_stdout():
                epochs.apply_baseline((None, 0.1))  # 使用前100ms作为基线
            
            print(f"成功创建Epochs对象: {len(epochs)} trials, {len(ch_names)} channels")
            return epochs
            
        except Exception as e:
            print(f"创建Epochs对象失败: {e}")
            return None
    
    def apply_mne_method(self, epochs, fwd, method='MNE'):
        """应用MNE方法进行源定位"""
        try:
            print(f"开始应用 {method} 方法...")
            
            # 参考MNE.py和eLORETA.py的参数设置
            if method == 'MNE':
                snr = 3.0  # MNE通常使用较高的SNR
                lambda2 = 1. / snr ** 2
            else:  # eLORETA
                snr = 2.0  # eLORETA使用较低的SNR
                lambda2 = 1. / snr ** 2
            
            # 设置平均参考
            epochs_ref = epochs.copy()
            with suppress_stdout():
                epochs_ref.set_eeg_reference('average', projection=True)
                epochs_ref.apply_proj()
            
            # 计算噪声协方差矩阵
            # 参考MNE.py的方式，使用基线期计算协方差
            noise_cov = mne.compute_covariance(
                epochs_ref, method=['shrunk', 'empirical'], 
                rank=None, verbose=False
            )
            
            # 创建逆算子
            inverse_operator = mne.minimum_norm.make_inverse_operator(
                epochs_ref.info, fwd, noise_cov, 
                loose='auto', depth=None, fixed=True, verbose=False
            )
            
            # 应用逆解
            stc = mne.minimum_norm.apply_inverse(
                epochs_ref.average(), inverse_operator, lambda2,
                method=method, return_residual=False, verbose=False
            )
            
            print(f"{method} 方法应用成功")
            return stc, inverse_operator
            
        except Exception as e:
            print(f"{method} 方法应用失败: {e}")
            return None, None
    
    def get_source_positions(self, fwd):
        """从前向模型中提取源空间位置"""
        try:
            # 获取源空间信息
            src = fwd['src']
            positions = []
            
            for hemi in src:
                # 获取每个半球的顶点位置
                rr = hemi['rr']  # 顶点位置
                inuse = hemi['inuse']  # 使用的顶点
                # 只保留使用的顶点
                positions.append(rr[inuse.astype(bool)])
            
            # 合并两个半球的位置
            all_positions = np.vstack(positions)
            return all_positions
            
        except Exception as e:
            print(f"提取源空间位置失败: {e}")
            return None

    def evaluate_localization(self, stc, ground_truth_positions, fwd):
        """评估源定位结果 - 计算MLE和AUC"""
        try:
            # 获取源空间位置
            src_positions = self.get_source_positions(fwd)
            if src_positions is None:
                return None
            
            # 获取峰值时间点的源活动
            peak_vertex, peak_time_idx = stc.get_peak(
                hemi=None, tmin=None, tmax=None, 
                mode='abs', vert_as_index=True, time_as_index=True
            )
            
            # 使用峰值时间点的源活动进行评估
            y_est = stc.data[:, peak_time_idx]
            
            # 创建ground truth向量（简化版本，假设最近的电极位置为真值）
            y_true = np.zeros_like(y_est)
            if len(ground_truth_positions) > 0:
                # 找到最接近的源点作为ground truth
                distances_to_sources = cdist(ground_truth_positions, src_positions)
                closest_source_idx = np.argmin(distances_to_sources[0])  # 使用第一个ground truth位置
                y_true[closest_source_idx] = 1.0
            
            # 计算MLE (Mean Localization Error)
            mle = eval_mean_localization_error(y_true, y_est, src_positions)
            
            # 计算AUC
            try:
                auc_result = eval_auc(y_true, y_est, src_positions)
                if isinstance(auc_result, tuple) and len(auc_result) == 2:
                    auc_close, auc_far = auc_result
                    auc_mean = (auc_close + auc_far) / 2
                else:
                    auc_close = auc_far = auc_mean = float(auc_result)
            except Exception as e:
                print(f"AUC计算失败: {e}")
                auc_close = auc_far = auc_mean = 0.0
            
            # 计算评估指标
            metrics = {
                'mle': mle,
                'auc_close': auc_close,
                'auc_far': auc_far,
                'auc_mean': auc_mean,
                'peak_time': stc.times[peak_time_idx],
                'peak_vertex': peak_vertex,
                'max_activation': np.max(np.abs(y_est))
            }
            
            return metrics
            
        except Exception as e:
            print(f"评估失败: {e}")
            return None
    
    
    def process_subject(self, subject):
        """处理单个被试的所有run"""
        print(f"\n{'='*50}")
        print(f"开始处理被试: {subject}")
        print(f"{'='*50}")
        
        # 1. 加载前向模型
        fwd = self.load_forward_model(subject)
        if fwd is None:
            return None
        
        # 2. 加载所有run的EEG数据
        all_runs_data = self.load_all_eeg_runs(subject)
        if all_runs_data is None:
            return None
        
        # 3. 加载ground truth位置
        gt_positions, gt_names = self.load_ground_truth_positions(subject)
        if gt_positions is None:
            return None
        
        # 4. 对每个run进行处理，收集结果
        run_results = {'MNE': [], 'eLORETA': []}
        
        for run_data in all_runs_data:
            run_num = run_data['run']
            eeg_data = run_data['data']
            channels_df = run_data['channels']
            
            print(f"\n--- 处理 Run {run_num} ---")
            
            # 创建Epochs对象
            epochs = self.create_epochs_object(eeg_data, channels_df)
            if epochs is None:
                print(f"  Run {run_num}: 创建Epochs对象失败，跳过")
                continue
            
            # 应用MNE方法
            stc_mne, inv_op_mne = self.apply_mne_method(epochs, fwd, method='MNE')
            if stc_mne is not None:
                metrics_mne = self.evaluate_localization(stc_mne, gt_positions, fwd)
                if metrics_mne:
                    run_results['MNE'].append(metrics_mne)
                    print(f"  MNE - MLE: {metrics_mne['mle']:.2f} mm, AUC: {metrics_mne['auc_mean']:.4f}")
            
            # 应用eLORETA方法
            stc_eloreta, inv_op_eloreta = self.apply_mne_method(epochs, fwd, method='eLORETA')
            if stc_eloreta is not None:
                metrics_eloreta = self.evaluate_localization(stc_eloreta, gt_positions, fwd)
                if metrics_eloreta:
                    run_results['eLORETA'].append(metrics_eloreta)
                    print(f"  eLORETA - MLE: {metrics_eloreta['mle']:.2f} mm, AUC: {metrics_eloreta['auc_mean']:.4f}")
        
        # 5. 计算每个方法的统计结果
        subject_results = {}
        
        for method in ['MNE', 'eLORETA']:
            if run_results[method]:
                mle_values = [m['mle'] for m in run_results[method]]
                auc_values = [m['auc_mean'] for m in run_results[method]]
                
                subject_results[method] = {
                    'mle_mean': np.mean(mle_values),
                    'mle_std': np.std(mle_values),
                    'mle_min': np.min(mle_values),
                    'mle_max': np.max(mle_values),
                    'auc_mean': np.mean(auc_values),
                    'auc_std': np.std(auc_values),
                    'auc_min': np.min(auc_values),
                    'auc_max': np.max(auc_values),
                    'n_runs': len(run_results[method]),
                    'all_runs': run_results[method]
                }
                
                print(f"\n{method} 统计结果 (n={len(run_results[method])} runs):")
                print(f"  MLE: {subject_results[method]['mle_mean']:.2f} ± {subject_results[method]['mle_std']:.2f} mm (最小: {subject_results[method]['mle_min']:.2f})")
                print(f"  AUC: {subject_results[method]['auc_mean']:.4f} ± {subject_results[method]['auc_std']:.4f} (最大: {subject_results[method]['auc_max']:.4f})")
        
        # 保存被试结果
        self.save_subject_results(subject, subject_results, gt_positions, gt_names)
        
        return subject_results
    
    def save_subject_results(self, subject, results, gt_positions, gt_names):
        """保存被试结果（包含所有run的统计信息）"""
        try:
            # 准备保存的数据
            save_data = {
                'subject': subject,
                'ground_truth_positions': gt_positions.tolist(),
                'ground_truth_names': gt_names.tolist(),
                'methods': {}
            }
            
            for method, result in results.items():
                save_data['methods'][method] = {
                    'n_runs': result['n_runs'],
                    'mle_mean': result['mle_mean'],
                    'mle_std': result['mle_std'],
                    'mle_min': result['mle_min'],
                    'mle_max': result['mle_max'],
                    'auc_mean': result['auc_mean'],
                    'auc_std': result['auc_std'],
                    'auc_min': result['auc_min'],
                    'auc_max': result['auc_max'],
                    'all_runs_metrics': [
                        {
                            'mle': run['mle'],
                            'auc_close': run['auc_close'],
                            'auc_far': run['auc_far'],
                            'auc_mean': run['auc_mean'],
                            'peak_time': run['peak_time'],
                            'max_activation': run['max_activation']
                        } for run in result['all_runs']
                    ]
                }
            
            # 保存JSON文件
            json_path = self.results_dir / f"{subject}_results.json"
            with open(json_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            print(f"被试结果已保存: {json_path}")
            
        except Exception as e:
            print(f"保存被试结果失败: {e}")
    
    def run_all_subjects(self):
        """运行所有被试的源定位分析"""
        print("开始传统源定位方法对比实验")
        print(f"数据集路径: {self.dataset_path}")
        print(f"被试列表: {self.subjects}")
        print(f"结果保存路径: {self.results_dir}")
        
        all_results = {}
        summary_stats = {
            'MNE': {'mle': [], 'auc': []}, 
            'eLORETA': {'mle': [], 'auc': []}
        }
        
        for subject in self.subjects:
            try:
                subject_results = self.process_subject(subject)
                if subject_results:
                    all_results[subject] = subject_results
                    
                    # 收集统计数据（使用每个被试的平均值）
                    for method in ['MNE', 'eLORETA']:
                        if method in subject_results:
                            summary_stats[method]['mle'].append(subject_results[method]['mle_mean'])
                            summary_stats[method]['auc'].append(subject_results[method]['auc_mean'])
                
            except Exception as e:
                print(f"处理被试 {subject} 时出错: {e}")
                continue
        
        # 生成总结报告（包含每个被试的详细结果）
        self.generate_summary_report(summary_stats, all_results)
        
        return all_results
    
    def generate_summary_report(self, summary_stats, all_results):
        """生成总结报告和汇总表格（包含每个被试的详细结果）"""
        try:
            report = {
                'experiment_info': {
                    'dataset_path': str(self.dataset_path),
                    'subjects': self.subjects,
                    'methods': ['MNE', 'eLORETA'],
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'summary_statistics': {},
                'subject_details': {}
            }
            
            print(f"\n{'='*80}")
            print("传统源定位方法对比实验总结")
            print(f"{'='*80}")
            
            # 创建汇总表格数据
            table_data = []
            
            for method, metrics in summary_stats.items():
                mle_values = metrics['mle']
                auc_values = metrics['auc']
                
                if mle_values and auc_values:
                    mean_mle = np.mean(mle_values)
                    std_mle = np.std(mle_values)
                    mean_auc = np.mean(auc_values)
                    std_auc = np.std(auc_values)
                    min_mle = np.min(mle_values)
                    max_mle = np.max(mle_values)
                    min_auc = np.min(auc_values)
                    max_auc = np.max(auc_values)
                    
                    report['summary_statistics'][method] = {
                        'mean_mle_mm': float(mean_mle),
                        'std_mle_mm': float(std_mle),
                        'min_mle_mm': float(min_mle),
                        'max_mle_mm': float(max_mle),
                        'mean_auc': float(mean_auc),
                        'std_auc': float(std_auc),
                        'min_auc': float(min_auc),
                        'max_auc': float(max_auc),
                        'n_subjects': len(mle_values)
                    }
                    
                    table_data.append({
                        'Method': method,
                        'N': len(mle_values),
                        'MLE_Mean': mean_mle,
                        'MLE_Std': std_mle,
                        'MLE_Min': min_mle,
                        'MLE_Max': max_mle,
                        'AUC_Mean': mean_auc,
                        'AUC_Std': std_auc,
                        'AUC_Min': min_auc,
                        'AUC_Max': max_auc
                    })
                else:
                    table_data.append({
                        'Method': method,
                        'N': 0,
                        'MLE_Mean': 0,
                        'MLE_Std': 0,
                        'MLE_Min': 0,
                        'MLE_Max': 0,
                        'AUC_Mean': 0,
                        'AUC_Std': 0,
                        'AUC_Min': 0,
                        'AUC_Max': 0
                    })
            
            # 打印整体汇总表格
            self.print_summary_table(table_data)
            
            # 保存整体汇总表格为CSV
            self.save_summary_table(table_data)
            
            # 生成每个被试的详细结果表格
            subject_detail_data = self.generate_subject_detail_table(all_results)
            
            # 将被试详细结果添加到报告中
            report['subject_details'] = subject_detail_data
            
            # 打印每个被试的详细结果
            self.print_subject_detail_table(subject_detail_data)
            
            # 保存每个被试的详细结果为CSV
            self.save_subject_detail_table(subject_detail_data)
            
            # 保存总结报告
            report_path = self.results_dir / "summary_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"\n总结报告已保存: {report_path}")
            
        except Exception as e:
            print(f"生成总结报告失败: {e}")
    
    def print_summary_table(self, table_data):
        """打印汇总表格"""
        print("\n" + "="*120)
        print("传统源定位方法性能汇总表")
        print("="*120)
        
        # 表头
        header = f"{'方法':<10} {'被试数':<6} {'MLE均值':<10} {'MLE标准差':<10} {'MLE最小值':<10} {'MLE最大值':<10} {'AUC均值':<10} {'AUC标准差':<10} {'AUC最小值':<10} {'AUC最大值':<10}"
        print(header)
        print("-" * 120)
        
        # 数据行
        for row in table_data:
            line = f"{row['Method']:<10} {row['N']:<6} {row['MLE_Mean']:<10.2f} {row['MLE_Std']:<10.2f} {row['MLE_Min']:<10.2f} {row['MLE_Max']:<10.2f} {row['AUC_Mean']:<10.4f} {row['AUC_Std']:<10.4f} {row['AUC_Min']:<10.4f} {row['AUC_Max']:<10.4f}"
            print(line)
        
        print("="*120)
        print("注：MLE单位为毫米(mm)，AUC无单位")
        print("="*120)
    
    def save_summary_table(self, table_data):
        """保存汇总表格为CSV文件"""
        try:
            import pandas as pd
            
            # 创建DataFrame
            df = pd.DataFrame(table_data)
            
            # 重命名列名为中文
            df.columns = ['方法', '被试数', 'MLE均值(mm)', 'MLE标准差(mm)', 'MLE最小值(mm)', 'MLE最大值(mm)', 
                         'AUC均值', 'AUC标准差', 'AUC最小值', 'AUC最大值']
            
            # 保存为CSV
            csv_path = self.results_dir / "summary_table.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"汇总表格已保存: {csv_path}")
            
            # 保存为Excel（如果可能）
            try:
                excel_path = self.results_dir / "summary_table.xlsx"
                df.to_excel(excel_path, index=False)
                print(f"汇总表格已保存: {excel_path}")
            except ImportError:
                print("未安装openpyxl，跳过Excel文件保存")
            
        except Exception as e:
            print(f"保存汇总表格失败: {e}")
    
    def generate_subject_detail_table(self, all_results):
        """生成每个被试的详细结果表格数据（显示MLE均值、最小值，AUC均值、最大值）"""
        detail_data = []
        
        for subject, subject_results in all_results.items():
            row = {'Subject': subject}
            
            # 添加MNE方法的结果
            if 'MNE' in subject_results:
                row['MNE_Runs'] = subject_results['MNE']['n_runs']
                row['MNE_MLE_Mean'] = subject_results['MNE']['mle_mean']
                row['MNE_MLE_Min'] = subject_results['MNE']['mle_min']
                row['MNE_AUC_Mean'] = subject_results['MNE']['auc_mean']
                row['MNE_AUC_Max'] = subject_results['MNE']['auc_max']
            else:
                row['MNE_Runs'] = 0
                row['MNE_MLE_Mean'] = None
                row['MNE_MLE_Min'] = None
                row['MNE_AUC_Mean'] = None
                row['MNE_AUC_Max'] = None
            
            # 添加eLORETA方法的结果
            if 'eLORETA' in subject_results:
                row['eLORETA_Runs'] = subject_results['eLORETA']['n_runs']
                row['eLORETA_MLE_Mean'] = subject_results['eLORETA']['mle_mean']
                row['eLORETA_MLE_Min'] = subject_results['eLORETA']['mle_min']
                row['eLORETA_AUC_Mean'] = subject_results['eLORETA']['auc_mean']
                row['eLORETA_AUC_Max'] = subject_results['eLORETA']['auc_max']
            else:
                row['eLORETA_Runs'] = 0
                row['eLORETA_MLE_Mean'] = None
                row['eLORETA_MLE_Min'] = None
                row['eLORETA_AUC_Mean'] = None
                row['eLORETA_AUC_Max'] = None
            
            detail_data.append(row)
        
        return detail_data
    
    def print_subject_detail_table(self, detail_data):
        """打印每个被试的详细结果表格（显示MLE均值、最小值，AUC均值、最大值）"""
        print("\n" + "="*160)
        print("每个被试的详细结果（跨所有run统计）")
        print("="*160)
        
        # 表头
        header = f"{'被试':<12} {'MNE_Runs':<10} {'MNE_MLE均值':<14} {'MNE_MLE最小':<14} {'MNE_AUC均值':<14} {'MNE_AUC最大':<14} {'eLOR_Runs':<10} {'eLOR_MLE均值':<14} {'eLOR_MLE最小':<14} {'eLOR_AUC均值':<14} {'eLOR_AUC最大':<14}"
        print(header)
        print("-" * 160)
        
        # 数据行
        for row in detail_data:
            mne_runs = row['MNE_Runs']
            mne_mle_mean = f"{row['MNE_MLE_Mean']:.2f}" if row['MNE_MLE_Mean'] is not None else "N/A"
            mne_mle_min = f"{row['MNE_MLE_Min']:.2f}" if row['MNE_MLE_Min'] is not None else "N/A"
            mne_auc_mean = f"{row['MNE_AUC_Mean']:.4f}" if row['MNE_AUC_Mean'] is not None else "N/A"
            mne_auc_max = f"{row['MNE_AUC_Max']:.4f}" if row['MNE_AUC_Max'] is not None else "N/A"
            
            eloreta_runs = row['eLORETA_Runs']
            eloreta_mle_mean = f"{row['eLORETA_MLE_Mean']:.2f}" if row['eLORETA_MLE_Mean'] is not None else "N/A"
            eloreta_mle_min = f"{row['eLORETA_MLE_Min']:.2f}" if row['eLORETA_MLE_Min'] is not None else "N/A"
            eloreta_auc_mean = f"{row['eLORETA_AUC_Mean']:.4f}" if row['eLORETA_AUC_Mean'] is not None else "N/A"
            eloreta_auc_max = f"{row['eLORETA_AUC_Max']:.4f}" if row['eLORETA_AUC_Max'] is not None else "N/A"
            
            line = f"{row['Subject']:<12} {mne_runs:<10} {mne_mle_mean:<14} {mne_mle_min:<14} {mne_auc_mean:<14} {mne_auc_max:<14} {eloreta_runs:<10} {eloreta_mle_mean:<14} {eloreta_mle_min:<14} {eloreta_auc_mean:<14} {eloreta_auc_max:<14}"
            print(line)
        
        print("="*160)
        print("注：MLE单位为毫米(mm)，显示各被试跨所有run的统计结果")
        print("="*160)
    
    def save_subject_detail_table(self, detail_data):
        """保存每个被试的详细结果为CSV文件（包含MLE均值、最小值，AUC均值、最大值）"""
        try:
            import pandas as pd
            
            # 创建DataFrame
            df = pd.DataFrame(detail_data)
            
            # 重命名列名为中文
            df.columns = ['被试', 'MNE_Run数', 'MNE_MLE均值(mm)', 'MNE_MLE最小值(mm)', 'MNE_AUC均值', 'MNE_AUC最大值',
                         'eLORETA_Run数', 'eLORETA_MLE均值(mm)', 'eLORETA_MLE最小值(mm)', 'eLORETA_AUC均值', 'eLORETA_AUC最大值']
            
            # 保存为CSV
            csv_path = self.results_dir / "subject_details.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"每个被试的详细结果已保存: {csv_path}")
            
            # 保存为Excel（如果可能）
            try:
                excel_path = self.results_dir / "subject_details.xlsx"
                df.to_excel(excel_path, index=False)
                print(f"每个被试的详细结果已保存: {excel_path}")
            except ImportError:
                print("未安装openpyxl，跳过Excel文件保存")
            
        except Exception as e:
            print(f"保存每个被试详细结果失败: {e}")


def main():
    """主函数"""
    print("传统源定位方法对比实验")
    print("支持方法: MNE, eLORETA")
    
    # 创建源定位器
    localizer = TraditionalSourceLocalizer()
    
    # 运行所有被试的分析
    results = localizer.run_all_subjects()
    
    print(f"\n实验完成！结果保存在: {localizer.results_dir}")
    
    return results


if __name__ == "__main__":
    results = main()
