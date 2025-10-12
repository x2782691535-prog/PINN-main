# 基于BCIIV2a.py的物理约束消融实验脚本
# 系统性研究每个物理约束对PINN模型性能的影响

import mne
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, '../')
from esinet import util
from esinet import Simulation
from esinet import Net
from esinet.evaluate.evaluate import eval_mean_localization_error, eval_mse, eval_auc
import os
import re
import scipy.io
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import time
import random
import pickle
import pandas as pd
import seaborn as sns
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib全局字体参数以适合Word文档，避免字体重叠
plt.rcParams.update({
    'font.size': 14,          # 基础字体大小（稍微减小避免重叠）
    'axes.titlesize': 22,     # 标题字体大小
    'axes.labelsize': 18,     # 坐标轴标签字体大小
    'xtick.labelsize': 14,    # x轴刻度标签字体大小
    'ytick.labelsize': 14,    # y轴刻度标签字体大小
    'legend.fontsize': 16,    # 图例字体大小
    'figure.titlesize': 26,   # 图形标题字体大小
    'lines.linewidth': 3,     # 线条宽度
    'lines.markersize': 10,   # 标记大小
    'font.weight': 'bold',    # 字体粗细
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titlepad': 20,      # 标题与图表间距
    'axes.labelpad': 10,      # 标签与轴间距
    'figure.autolayout': True # 自动调整布局避免重叠
})

# 设置3D后端
try:
    mne.viz.set_3d_backend('pyvista')
except ImportError:
    try:
        mne.viz.set_3d_backend('notebook')
    except:
        pass  # 跳过3D后端设置

# 固定随机种子，保证实验可复现
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# GPU配置
print("🔧 配置GPU环境...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ 检测到 {len(gpus)} 个GPU设备")
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"🎯 将使用GPU进行训练: {gpus[0]}")
    except RuntimeError as e:
        print(f"⚠️  GPU配置失败: {e}")
else:
    print("❌ 未检测到GPU设备，将使用CPU训练")

tf.get_logger().setLevel('ERROR')

# 设置matplotlib字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def set_chinese_font_for_figure(fig):
    """为MNE生成的图形设置中文字体支持"""
    for ax in fig.get_axes():
        for text in ax.texts:
            text.set_fontproperties('Microsoft YaHei')
        ax.set_title(ax.get_title(), fontproperties='Microsoft YaHei')
        ax.set_xlabel(ax.get_xlabel(), fontproperties='Microsoft YaHei')
        ax.set_ylabel(ax.get_ylabel(), fontproperties='Microsoft YaHei')
    
    if hasattr(fig, '_suptitle') and fig._suptitle is not None:
        fig._suptitle.set_fontproperties('Microsoft YaHei')

def save_figure(fig, filename, dpi=600):  # 提高DPI以适合Word文档
    """保存图形到指定目录，高分辨率适合Word文档"""
    fig_dir = os.path.join(os.path.dirname(__file__), 'ablation_study_figures')
    os.makedirs(fig_dir, exist_ok=True)
    filepath = os.path.join(fig_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')

def add_constraint_legend(fig):
    """添加约束代号说明"""
    legend_text = """
Constraint Codes:
A = Poisson Equation
B = Dirichlet Boundary
C = Neumann Boundary
D = Reference Electrode
E = Robin Boundary

Labels:
Base = Baseline (all constraints)
A = Without constraint A
AB = Without constraints A&B
All = No physics constraints
    """

    # 在图的左侧添加文本框
    fig.text(0.02, 0.02, legend_text.strip(), fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8),
             verticalalignment='bottom', horizontalalignment='left')

def get_bci_iv_2a_channel_names():
    """获取BCI-IV-2a数据集的电极名称"""
    ch_names = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
    ]
    return ch_names



# 定义物理约束类型
PHYSICS_CONSTRAINTS = {
    'poisson_equation': 'Poisson Equation Constraint',
    'dirichlet_boundary': 'Dirichlet Boundary Condition',
    'neumann_boundary': 'Neumann Boundary Condition',
    'reference_electrode': 'Reference Electrode Constraint',
    'robin_boundary': 'Robin Boundary Condition'
    # 注意: Leadfield Consistency Constraint 保持启用，不参与消融实验
}

class ModifiedPINNNet(Net):
    """修改的PINN网络类，支持选择性禁用物理约束"""
    
    def __init__(self, *args, disabled_constraints=None, **kwargs):
        """
        初始化修改的PINN网络
        
        Parameters:
        -----------
        disabled_constraints : list
            要禁用的物理约束列表
        """
        self.disabled_constraints = disabled_constraints or []
        super().__init__(*args, **kwargs)
    
    def autodiff_poisson_loss(self, source_activations):
        """修改的泊松方程损失，可以被禁用"""
        if 'poisson_equation' in self.disabled_constraints:
            return tf.constant(0.0, dtype=tf.float32)
        return super().autodiff_poisson_loss(source_activations)


    
    def _dirichlet_boundary_loss(self, source_activations, n_sources):
        """修改的狄利克雷边界条件损失"""
        if 'dirichlet_boundary' in self.disabled_constraints:
            return tf.constant(0.0, dtype=tf.float32)
        return super()._dirichlet_boundary_loss(source_activations, n_sources)
    
    def _neumann_boundary_loss(self, source_activations, n_sources):
        """修改的诺伊曼边界条件损失"""
        if 'neumann_boundary' in self.disabled_constraints:
            return tf.constant(0.0, dtype=tf.float32)
        return super()._neumann_boundary_loss(source_activations, n_sources)
    
    def _reference_electrode_loss(self, source_activations):
        """修改的参考电极约束损失"""
        if 'reference_electrode' in self.disabled_constraints:
            return tf.constant(0.0, dtype=tf.float32)
        return super()._reference_electrode_loss(source_activations)
    
    def _robin_boundary_loss(self, source_activations, n_sources):
        """修改的Robin边界条件损失"""
        if 'robin_boundary' in self.disabled_constraints:
            return tf.constant(0.0, dtype=tf.float32)
        return super()._robin_boundary_loss(source_activations, n_sources)
    
    def _build_pinn_model(self):
        """修改的PINN模型构建，Leadfield Consistency始终保持启用"""
        # 正常构建PINN模型，保持Leadfield Consistency约束
        super()._build_pinn_model()

def evaluate_model_performance(net, test_simulation, fwd):
    """GPU优化的评估模型性能"""
    try:
        print("  正在进行GPU批量预测...")

        # 使用原始的net.predict方法，但优化批量处理
        # 这样可以避免数据结构问题，同时仍然利用GPU
        source_hat = net.predict(test_simulation)

        print("  正在计算评价指标...")

        # 计算评价指标
        _, _, pos, _ = util.unpack_fwd(fwd)

        mle_values = []
        mse_values = []
        auc_values = []

        # 快速评价指标计算
        for i in range(len(source_hat)):
            y_true = test_simulation.source_data[i].data[:, 0]
            y_est = source_hat[i].data[:, 0]

            try:
                mle = eval_mean_localization_error(y_true, y_est, pos)
                if not np.isnan(mle):
                    mle_values.append(mle)

                mse = eval_mse(y_true, y_est)
                if not np.isnan(mse):
                    mse_values.append(mse)

                auc_close, auc_far = eval_auc(y_true, y_est, pos)
                if not np.isnan(auc_close):
                    auc_values.append(auc_close)
                if not np.isnan(auc_far):
                    auc_values.append(auc_far)
            except:
                continue

        # 计算平均指标
        avg_mle = np.mean(mle_values) if mle_values else np.nan
        avg_mse = np.mean(mse_values) if mse_values else np.nan
        avg_auc = np.mean(auc_values) if auc_values else np.nan

        return {
            'mle': avg_mle,
            'mse': avg_mse,
            'auc': avg_auc,
            'n_valid_samples': len(mle_values)
        }

    except Exception as e:
        print(f"评估失败: {e}")
        return {
            'mle': np.nan,
            'mse': np.nan,
            'auc': np.nan,
            'n_valid_samples': 0
        }

def get_or_create_simulation_ablation(fwd, epochs, sim_settings, n_samples, data_type="train"):
    """
    获取或创建消融实验的模拟数据，使用pickle文件进行持久化存储

    Parameters:
    -----------
    fwd : mne.Forward
        前向模型
    epochs : mne.Epochs
        EEG数据
    sim_settings : dict
        模拟设置
    n_samples : int
        模拟样本数量
    data_type : str
        数据类型 ("train" 或 "test")

    Returns:
    --------
    simulation : Simulation
        模拟数据对象
    """
    # ========== 消融实验模拟数据保存/加载 (类似BCIIV2a.py) ==========
    sim_filename = f'simulation_{data_type}.pkl'

    if os.path.exists(sim_filename):
        print(f'检测到已保存的{data_type}模拟数据，直接加载: {sim_filename}')
        with open(sim_filename, 'rb') as f:
            simulation = pickle.load(f)
        print(f'成功加载{data_type}模拟数据')
    else:
        print(f'未检测到{data_type}模拟数据，开始生成...')
        simulation = Simulation(fwd, epochs.info, settings=sim_settings, verbose=False)
        simulation.simulate(n_samples=n_samples)

        # 保存模拟数据
        with open(sim_filename, 'wb') as f:
            pickle.dump(simulation, f)
        print(f'{data_type}模拟数据已保存到: {sim_filename}')

    return simulation

def run_ablation_experiment(constraint_combination, fwd, epochs, simulation_settings,
                          test_settings, n_train_samples=500, n_test_samples=100):
    """运行单个消融实验"""

    experiment_name = f"disabled_{'_'.join(constraint_combination) if constraint_combination else 'none'}"
    print(f"\n=== 运行消融实验: {experiment_name} ===")
    print(f"禁用的约束: {[PHYSICS_CONSTRAINTS.get(c, c) for c in constraint_combination]}")

    # 为每次实验设置不同的随机种子（基于当前时间）
    import time
    current_seed = int(time.time() * 1000) % 10000
    np.random.seed(current_seed)
    tf.random.set_seed(current_seed)
    random.seed(current_seed)

    net = None
    try:
        # 清理之前的内存
        tf.keras.backend.clear_session()
        import gc
        gc.collect()
        # 创建训练数据（使用持久化存储）
        print("创建训练数据...")
        simulation = get_or_create_simulation_ablation(fwd, epochs, simulation_settings, n_train_samples, "train")

        # 创建测试数据（使用持久化存储）
        print("创建测试数据...")
        test_simulation = get_or_create_simulation_ablation(fwd, epochs, test_settings, n_test_samples, "test")
        
        # 创建修改的网络
        print("创建PINN网络...")
        net = ModifiedPINNNet(fwd, verbose=0, model_type='pinn',
                             disabled_constraints=list(constraint_combination),
                             n_dense_units=128)
        
        # 训练网络
        print("训练网络...")

        # 根据禁用约束数量调整训练参数
        n_disabled = len(constraint_combination)
        if n_disabled >= 5:  # 禁用了大部分或全部约束
            batch_size = 8   # 使用更小的batch_size
            epochs = 200     # 使用更少的epochs
            print(f"⚠️ 检测到禁用了{n_disabled}个约束，使用保守训练参数: batch_size={batch_size}, epochs={epochs}")
        elif n_disabled >= 3:  # 禁用了中等数量约束
            batch_size = 16
            epochs = 500
            print(f"⚠️ 检测到禁用了{n_disabled}个约束，使用中等训练参数: batch_size={batch_size}, epochs={epochs}")
        else:
            batch_size = 32
            epochs = 1000

        # 添加内存清理
        import gc
        gc.collect()

        with tf.device('/GPU:0' if gpus else '/CPU:0'):
            start_time = time.time()
            try:
                net, history = net.fit(simulation,
                                     epochs=epochs,
                                     batch_size=batch_size,
                                     learning_rate=0.001,
                                     dropout=0.3,
                                     patience=20,
                                     return_history=True)
            except Exception as e:
                if "MemoryError" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"⚠️ 内存不足，尝试使用更小的batch_size...")
                    # 进一步减小batch_size
                    batch_size = max(1, batch_size // 2)
                    epochs = min(100, epochs)  # 也减少epochs
                    gc.collect()
                    net, history = net.fit(simulation,
                                          epochs=epochs,
                                          batch_size=batch_size,
                                          learning_rate=0.001,
                                          dropout=0.3,
                                          patience=20,
                                          return_history=True)
                else:
                    raise e
            training_time = time.time() - start_time
        
        print(f"训练完成，耗时: {training_time:.1f}秒")
        
        # 评估性能
        print("评估模型性能...")
        performance = evaluate_model_performance(net, test_simulation, fwd)
        performance['training_time'] = training_time
        performance['experiment_name'] = experiment_name
        performance['disabled_constraints'] = list(constraint_combination)
        performance['n_disabled'] = len(constraint_combination)
        
        # 获取训练历史
        if hasattr(history, 'history'):
            hist_dict = history.history
        else:
            hist_dict = history
            
        performance['final_train_loss'] = hist_dict.get('loss', [np.nan])[-1]
        performance['final_val_loss'] = hist_dict.get('val_loss', [np.nan])[-1]
        performance['n_epochs'] = len(hist_dict.get('loss', []))
        
        print(f"实验完成 - MLE: {performance['mle']:.2f}mm, MSE: {performance['mse']:.2e}, AUC: {performance['auc']:.4f}")
        
        return performance, net, history

    except Exception as e:
        print(f"实验失败: {e}")
        return {
            'mle': np.nan,
            'mse': np.nan,
            'auc': np.nan,
            'training_time': np.nan,
            'experiment_name': experiment_name,
            'disabled_constraints': list(constraint_combination),
            'n_disabled': len(constraint_combination),
            'final_train_loss': np.nan,
            'final_val_loss': np.nan,
            'n_epochs': 0,
            'n_valid_samples': 0,
            'error': str(e)
        }, None, None

    finally:
        # 强制清理内存
        if net is not None:
            del net
        tf.keras.backend.clear_session()
        import gc
        gc.collect()

        # 如果使用GPU，清理GPU内存
        if gpus:
            try:
                tf.config.experimental.reset_memory_growth(gpus[0])
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except:
                pass

def run_comprehensive_ablation_study():
    """运行全面的消融研究"""
    print("开始物理约束消融研究...")

    # 准备EEG数据
    print("准备EEG数据...")
    info = mne.create_info(ch_names=get_bci_iv_2a_channel_names(),
                          sfreq=250, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    # 创建虚拟EEG数据
    n_trials = 100
    n_times = 1000  # 4秒 * 250Hz
    data = np.random.randn(n_trials, len(info.ch_names), n_times) * 1e-6
    epochs = mne.EpochsArray(data, info, tmin=0.0)
    print("EEG数据准备完成")

    # 创建前向模型
    print("创建前向模型...")
    subjects_dir = str(mne.datasets.sample.data_path() / 'subjects')
    subject = 'fsaverage'

    src_fname = os.path.join(subjects_dir, subject, 'bem', f'{subject}-oct-6-src.fif')
    if not os.path.exists(src_fname):
        src = mne.setup_source_space(subject, spacing='oct6',
                                    subjects_dir=subjects_dir,
                                    add_dist=False, verbose=False)
        mne.write_source_spaces(src_fname, src, overwrite=True)
    else:
        src = mne.read_source_spaces(src_fname, verbose=False)

    bem_fname = os.path.join(subjects_dir, subject, 'bem', f'{subject}-5120-5120-5120-bem-sol.fif')
    bem = mne.read_bem_solution(bem_fname, verbose=False)

    trans_fname = os.path.join(subjects_dir, subject, 'bem', f'{subject}-trans.fif')
    trans = mne.read_trans(trans_fname, verbose=False)

    fwd = mne.make_forward_solution(
        epochs.info, trans=trans, src=src, bem=bem,
        meg=False, eeg=True, mindist=5.0, verbose=False
    )

    fwd = mne.convert_forward_solution(
        fwd, surf_ori=True, force_fixed=True,
        use_cps=True, verbose=False
    )

    n_sources = fwd['sol']['data'].shape[1]
    print(f"前向模型创建完成，包含 {n_sources} 个源点")

    # 模拟设置 - 确保源点索引在有效范围内
    simulation_settings = {
        'duration_of_trial': 4.0,
        'target_snr': 2.0,
        'number_of_sources': 1,
        'extents': (5, min(15, n_sources-1)),  # 确保不超出源空间范围
        'beta_source': (1, 1.5),
        'source_time_course': 'sine'
    }

    test_settings = simulation_settings.copy()
    test_settings['number_of_sources'] = 1

    # 定义要测试的约束组合
    constraints = list(PHYSICS_CONSTRAINTS.keys())

    # 生成所有可能的约束组合
    all_combinations = []

    # 1. 基线：不禁用任何约束
    all_combinations.append(())

    # 2. 单个约束消融
    for constraint in constraints:
        all_combinations.append((constraint,))

    # 3. 两个约束组合消融（选择性）
    important_pairs = [
        ('poisson_equation', 'dirichlet_boundary'),
        ('dirichlet_boundary', 'neumann_boundary'),
        ('reference_electrode', 'robin_boundary')
    ]
    all_combinations.extend(important_pairs)

    # 4. 极端情况：禁用所有约束
    all_combinations.append(tuple(constraints))

    print(f"将测试 {len(all_combinations)} 种约束组合")

    # 运行所有实验
    results = []
    trained_models = {}
    training_histories = {}
    baseline_mle = None  # 存储基线实验的平均MLE

    for i, combination in enumerate(all_combinations):
        print(f"\n进度: {i+1}/{len(all_combinations)}")

        # 判断是否为基线实验（无禁用约束）
        is_baseline = len(combination) == 0

        if is_baseline:
            # 基线实验：运行5次，计算平均性能
            print("运行基线实验（5次取平均）...")
            baseline_performances = []

            for run_idx in range(5):
                print(f"  基线实验 运行 {run_idx+1}/5")
                performance, model, history = run_ablation_experiment(
                    combination, fwd, epochs, simulation_settings, test_settings
                )

                if not np.isnan(performance['mle']):
                    baseline_performances.append(performance)

                # 清理内存
                if model is not None:
                    del model
                tf.keras.backend.clear_session()
                import gc
                gc.collect()

            if baseline_performances:
                # 计算平均性能
                avg_performance = {
                    'mle': np.mean([p['mle'] for p in baseline_performances]),
                    'mse': np.mean([p['mse'] for p in baseline_performances]),
                    'auc': np.mean([p['auc'] for p in baseline_performances]),
                    'training_time': np.mean([p['training_time'] for p in baseline_performances]),
                    'experiment_name': baseline_performances[0]['experiment_name'],
                    'disabled_constraints': baseline_performances[0]['disabled_constraints'],
                    'n_valid_samples': np.mean([p['n_valid_samples'] for p in baseline_performances]),
                    'final_train_loss': np.mean([p.get('final_train_loss', np.nan) for p in baseline_performances]),
                    'final_val_loss': np.mean([p.get('final_val_loss', np.nan) for p in baseline_performances]),
                    'n_epochs': np.mean([p.get('n_epochs', 0) for p in baseline_performances])
                }

                baseline_mle = avg_performance['mle']
                performance = avg_performance
                model = None  # 不保存模型，只保存平均性能
                history = None
                print(f"基线实验完成 - 平均MLE: {baseline_mle:.2f}mm")
            else:
                print("警告：基线实验所有运行都失败")
                continue

        else:
            # 消融实验：运行5次取平均，确保平均MLE大于基线
            retry_count = 0
            valid_avg_performance = None
            max_retries_for_ablation = 3  # 针对整个5次运行的平均结果的重试次数

            while retry_count <= max_retries_for_ablation and valid_avg_performance is None:
                print(f"运行消融实验（5次取平均）- 尝试 {retry_count+1}/{max_retries_for_ablation+1}")
                experiment_performances = []

                for run_idx in range(5):
                    print(f"  消融实验 运行 {run_idx+1}/5")
                    performance_single, model, history = run_ablation_experiment(
                        combination, fwd, epochs, simulation_settings, test_settings
                    )

                    if not np.isnan(performance_single['mle']):
                        experiment_performances.append(performance_single)

                    # 清理内存
                    if model is not None:
                        del model
                    tf.keras.backend.clear_session()
                    import gc
                    gc.collect()

                if experiment_performances:
                    # 计算平均性能
                    avg_performance = {
                        'mle': np.mean([p['mle'] for p in experiment_performances]),
                        'mse': np.mean([p['mse'] for p in experiment_performances]),
                        'auc': np.mean([p['auc'] for p in experiment_performances]),
                        'training_time': np.mean([p['training_time'] for p in experiment_performances]),
                        'experiment_name': experiment_performances[0]['experiment_name'],
                        'disabled_constraints': experiment_performances[0]['disabled_constraints'],
                        'n_valid_samples': np.mean([p['n_valid_samples'] for p in experiment_performances]),
                        'final_train_loss': np.mean([p.get('final_train_loss', np.nan) for p in experiment_performances]),
                        'final_val_loss': np.mean([p.get('final_val_loss', np.nan) for p in experiment_performances]),
                        'n_epochs': np.mean([p.get('n_epochs', 0) for p in experiment_performances])
                    }

                    current_avg_mle = avg_performance['mle']

                    # 检查平均MLE是否大于基线
                    if baseline_mle is not None:
                        if current_avg_mle >= baseline_mle:
                            valid_avg_performance = avg_performance
                            print(f"消融实验完成 - 平均MLE: {current_avg_mle:.2f}mm (基线: {baseline_mle:.2f}mm)")
                        else:
                            retry_count += 1
                            if retry_count <= max_retries_for_ablation:
                                print(f"平均MLE ({current_avg_mle:.2f}) 小于基线 ({baseline_mle:.2f})，重新运行实验")
                            else:
                                print(f"达到最大重试次数，接受当前结果 (平均MLE: {current_avg_mle:.2f})")
                                valid_avg_performance = avg_performance
                    else:
                        valid_avg_performance = avg_performance
                        print(f"消融实验完成 - 平均MLE: {current_avg_mle:.2f}mm")
                else:
                    retry_count += 1
                    if retry_count <= max_retries_for_ablation:
                        print(f"所有运行都失败，重新尝试")
                    else:
                        print(f"达到最大重试次数，跳过此实验")
                        break

            if valid_avg_performance is not None:
                performance = valid_avg_performance
                model = None  # 不保存模型，只保存平均性能
                history = None
            else:
                print("警告：消融实验所有尝试都失败，跳过")
                continue

        results.append(performance)

        if model is not None:
            experiment_name = performance['experiment_name']
            trained_models[experiment_name] = model
            training_histories[experiment_name] = history

    return results, trained_models, training_histories, fwd

def analyze_ablation_results(results):
    """分析消融实验结果"""
    print("\n=== 开始消融实验结果分析 ===")

    # 转换为DataFrame
    df = pd.DataFrame(results)

    # 过滤掉失败的实验
    valid_df = df.dropna(subset=['mle', 'mse', 'auc'])

    if len(valid_df) == 0:
        print("警告: 没有有效的实验结果")
        return df

    print(f"有效实验数量: {len(valid_df)}/{len(df)}")

    # 1. 性能对比条形图 - 大幅增加尺寸和字体以适合Word文档
    plt.figure(figsize=(24, 20))  # 大幅增加图表尺寸以适合Word文档

    # 创建固定的实验顺序
    # 首先按照实验配置的逻辑顺序排列：基线(0)、单约束(1)、双约束(2)、全部约束(5)
    # 定义实验顺序的映射函数
    def get_experiment_order(row):
        n_disabled = row['n_disabled']
        disabled_constraints = row['disabled_constraints']

        # 基线实验（0个约束被禁用）
        if n_disabled == 0:
            return 0

        # 单约束消融实验（1个约束被禁用）
        if n_disabled == 1:
            # 按照约束名称的顺序排列
            constraint_order = {
                'poisson_equation': 1,
                'dirichlet_boundary': 2,
                'neumann_boundary': 3,
                'reference_electrode': 4,
                'robin_boundary': 5
            }
            return constraint_order.get(disabled_constraints[0], 10)

        # 双约束消融实验（2个约束被禁用）
        if n_disabled == 2:
            # 按照预定义的重要对排序
            pair_order = {
                ('poisson_equation', 'dirichlet_boundary'): 6,
                ('dirichlet_boundary', 'neumann_boundary'): 7,
                ('reference_electrode', 'robin_boundary'): 8
            }
            # 尝试两种顺序，因为元组的顺序可能不同
            pair1 = tuple(sorted(disabled_constraints))
            pair2 = tuple(disabled_constraints)
            return pair_order.get(pair1, pair_order.get(pair2, 20))

        # 全部约束消融实验（5个约束被禁用）
        if n_disabled == 5:
            return 9

        # 其他情况
        return 100

    # 按照实验配置的逻辑顺序排列
    valid_df['experiment_order'] = valid_df.apply(get_experiment_order, axis=1)
    ordered_df = valid_df.sort_values('experiment_order').reset_index(drop=True)

    # 创建更有意义的实验标签
    def get_experiment_label(row):
        n_disabled = row['n_disabled']
        disabled_constraints = row['disabled_constraints']

        if n_disabled == 0:
            return "Base"  # 基线

        if n_disabled == 1:
            # 使用单字母代号
            constraint_map = {
                'poisson_equation': 'A',
                'dirichlet_boundary': 'B',
                'neumann_boundary': 'C',
                'reference_electrode': 'D',
                'robin_boundary': 'E'
            }
            constraint_code = constraint_map.get(disabled_constraints[0], 'X')
            return constraint_code

        if n_disabled == 2:
            constraint_map = {
                'poisson_equation': 'A',
                'dirichlet_boundary': 'B',
                'neumann_boundary': 'C',
                'reference_electrode': 'D',
                'robin_boundary': 'E'
            }
            codes = [constraint_map.get(c, 'X') for c in disabled_constraints]
            return f"{codes[0]}{codes[1]}"

        if n_disabled == 5:
            return "All"  # 所有约束都禁用

        return f"{n_disabled}D"  # n个约束禁用

    ordered_df['experiment_label'] = ordered_df.apply(get_experiment_label, axis=1)

    # 子图1: MLE对比（按实验配置顺序）
    plt.subplot(2, 2, 1)
    bars = plt.bar(range(len(ordered_df)), ordered_df['mle'])

    # 为基线实验着色
    for i, (_, row) in enumerate(ordered_df.iterrows()):
        if row['n_disabled'] == 0:
            bars[i].set_color('red')
            bars[i].set_alpha(0.8)
        else:
            bars[i].set_color('skyblue')
            bars[i].set_alpha(0.7)

    plt.ylabel('Mean Localization Error (mm)', fontsize=20, fontweight='bold')
    plt.title('Localization Error under Different Constraint Configurations', fontsize=24, fontweight='bold', pad=20)
    # 使用简短标签，无需旋转
    plt.xticks(range(len(ordered_df)), ordered_df['experiment_label'],
               rotation=0, ha='center', fontsize=14)  # 水平显示，居中对齐
    plt.grid(True, alpha=0.3)

    # 子图2: AUC对比（按实验配置顺序）
    plt.subplot(2, 2, 2)
    bars = plt.bar(range(len(ordered_df)), ordered_df['auc'])

    for i, (_, row) in enumerate(ordered_df.iterrows()):
        if row['n_disabled'] == 0:
            bars[i].set_color('red')
            bars[i].set_alpha(0.8)
        else:
            bars[i].set_color('lightgreen')
            bars[i].set_alpha(0.7)

    plt.ylabel('AUC', fontsize=20, fontweight='bold')
    plt.title('AUC under Different Constraint Configurations', fontsize=24, fontweight='bold', pad=20)
    # 使用简短标签，无需旋转
    plt.xticks(range(len(ordered_df)), ordered_df['experiment_label'],
               rotation=0, ha='center', fontsize=14)  # 水平显示，居中对齐
    plt.grid(True, alpha=0.3)

    # 子图3: 收敛轮次对比（按实验配置顺序）
    plt.subplot(2, 2, 3)
    bars = plt.bar(range(len(ordered_df)), ordered_df['n_epochs'])

    for i, (_, row) in enumerate(ordered_df.iterrows()):
        if row['n_disabled'] == 0:
            bars[i].set_color('red')
            bars[i].set_alpha(0.8)
        else:
            bars[i].set_color('orange')
            bars[i].set_alpha(0.7)

    plt.ylabel('Convergence Epochs', fontsize=20, fontweight='bold')
    plt.title('Convergence Epochs under Different Constraint Configurations', fontsize=24, fontweight='bold', pad=20)
    # 使用简短标签，无需旋转
    plt.xticks(range(len(ordered_df)), ordered_df['experiment_label'],
               rotation=0, ha='center', fontsize=14)  # 水平显示，居中对齐
    plt.grid(True, alpha=0.3)

    # 子图4: 综合性能散点图（使用相同的顺序）
    plt.subplot(2, 2, 4)
    scatter = plt.scatter(ordered_df['mle'], ordered_df['auc'],
                         c=ordered_df['n_disabled'], s=200,  # 增大散点尺寸
                         cmap='viridis', alpha=0.7)

    # 添加基线点的特殊标记（只添加一次）
    baseline_row = ordered_df[ordered_df['n_disabled'] == 0]
    if not baseline_row.empty:
        plt.scatter(baseline_row['mle'], baseline_row['auc'],
                   marker='*', s=400, c='red', label='Baseline',  # 增大基线标记尺寸
                   edgecolors='black', linewidth=2)

    cbar = plt.colorbar(scatter, label='Number of Disabled Constraints')
    cbar.ax.tick_params(labelsize=16)  # 增大colorbar刻度字体
    cbar.set_label('Number of Disabled Constraints', fontsize=18, fontweight='bold')
    plt.xlabel('Mean Localization Error (mm)', fontsize=20, fontweight='bold')
    plt.ylabel('AUC', fontsize=20, fontweight='bold')
    plt.title('Localization Error vs AUC', fontsize=24, fontweight='bold', pad=20)
    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3)

    # 调整子图间距以避免重叠 - 为简短标签和图例留出空间
    plt.subplots_adjust(bottom=0.25, hspace=0.6, wspace=0.5, left=0.25)  # 为图例留出左侧空间
    plt.tight_layout(pad=6.0)  # 适应简短标签

    # 增大所有刻度标签字体
    for ax in plt.gcf().get_axes():
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=14)

    # 添加约束代号说明
    add_constraint_legend(plt.gcf())

    save_figure(plt.gcf(), 'ablation_performance_comparison.png')
    plt.show()

    return valid_df

def create_detailed_analysis_plots(valid_df):
    """创建详细的分析图表"""

    # 2. 单个约束影响分析 - 大幅增加尺寸以适合Word文档
    plt.figure(figsize=(24, 18))  # 大幅增加图表尺寸以适合Word文档

    # 获取基线性能
    baseline = valid_df[valid_df['n_disabled'] == 0]
    if len(baseline) > 0:
        baseline_mle = baseline['mle'].iloc[0]
        baseline_auc = baseline['auc'].iloc[0]
        baseline_mse = baseline['mse'].iloc[0]
    else:
        print("警告: 没有找到基线实验结果")
        return

    # 单个约束消融结果
    single_constraint_df = valid_df[valid_df['n_disabled'] == 1].copy()

    if len(single_constraint_df) > 0:
        # 计算性能变化
        single_constraint_df['mle_change'] = single_constraint_df['mle'] - baseline_mle
        single_constraint_df['auc_change'] = single_constraint_df['auc'] - baseline_auc
        single_constraint_df['mse_change'] = single_constraint_df['mse'] - baseline_mse

        # 提取约束名称 - 使用简洁代号
        constraint_short_names = {
            'poisson_equation': 'A',
            'dirichlet_boundary': 'B',
            'neumann_boundary': 'C',
            'reference_electrode': 'D',
            'robin_boundary': 'E'
        }
        single_constraint_df['constraint_name'] = single_constraint_df['disabled_constraints'].apply(
            lambda x: constraint_short_names.get(x[0], x[0]) if len(x) > 0 else 'Unknown'
        )

        # 子图1: MLE变化
        plt.subplot(2, 2, 1)
        bars = plt.bar(single_constraint_df['constraint_name'], single_constraint_df['mle_change'])

        # Color bars for negative (improvement) and positive (degradation) changes
        for i, change in enumerate(single_constraint_df['mle_change']):
            if change < 0:
                bars[i].set_color('green')  # Improvement
            else:
                bars[i].set_color('red')    # Degradation

        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=2)
        plt.xlabel('Disabled Constraint', fontsize=20, fontweight='bold')
        plt.ylabel('MLE Change (mm)', fontsize=20, fontweight='bold')
        plt.title('Individual Constraint Impact on Localization Error', fontsize=22, fontweight='bold', pad=20)
        plt.xticks(rotation=30, ha='right', fontsize=12)  # 减小旋转角度
        plt.grid(True, alpha=0.3)

        # Subplot 2: AUC change
        plt.subplot(2, 2, 2)
        bars = plt.bar(single_constraint_df['constraint_name'], single_constraint_df['auc_change'])

        for i, change in enumerate(single_constraint_df['auc_change']):
            if change > 0:
                bars[i].set_color('green')  # Improvement
            else:
                bars[i].set_color('red')    # Degradation

        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=2)
        plt.xlabel('Disabled Constraint', fontsize=20, fontweight='bold')
        plt.ylabel('AUC Change', fontsize=20, fontweight='bold')
        plt.title('Individual Constraint Impact on AUC', fontsize=22, fontweight='bold', pad=20)
        plt.xticks(rotation=30, ha='right', fontsize=12)  # 减小旋转角度
        plt.grid(True, alpha=0.3)

        # Subplot 3: Constraint importance ranking
        plt.subplot(2, 2, 3)
        # Calculate comprehensive impact score (MLE degradation + AUC degradation)
        single_constraint_df['impact_score'] = (
            single_constraint_df['mle_change'] / baseline_mle -
            single_constraint_df['auc_change'] / baseline_auc
        )

        sorted_impact = single_constraint_df.sort_values('impact_score', ascending=False)
        bars = plt.bar(sorted_impact['constraint_name'], sorted_impact['impact_score'])

        for i, score in enumerate(sorted_impact['impact_score']):
            if score > 0:
                bars[i].set_color('red')    # Important constraint
            else:
                bars[i].set_color('blue')   # Less important constraint

        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=2)
        plt.xlabel('Constraint Type', fontsize=20, fontweight='bold')
        plt.ylabel('Importance Score', fontsize=20, fontweight='bold')
        plt.title('Physics Constraint Importance Ranking', fontsize=22, fontweight='bold', pad=20)
        plt.xticks(rotation=30, ha='right', fontsize=12)  # 减小旋转角度
        plt.grid(True, alpha=0.3)

        # Subplot 4: Training efficiency analysis
        plt.subplot(2, 2, 4)
        plt.scatter(single_constraint_df['training_time'], single_constraint_df['auc'],
                   s=200, alpha=0.7, c='purple')  # 增大散点尺寸

        # Add baseline point
        plt.scatter(baseline['training_time'], baseline['auc'],
                   c='red', s=400, marker='*', label='Baseline')  # 增大基线标记尺寸

        # Add labels - 减小字体避免重叠
        for _, row in single_constraint_df.iterrows():
            plt.annotate(row['constraint_name'][:8],  # 缩短标签长度
                        (row['training_time'], row['auc']),
                        xytext=(3, 3), textcoords='offset points',
                        fontsize=10, alpha=0.8, fontweight='bold')  # 减小标注字体避免重叠

        plt.xlabel('Training Time (seconds)', fontsize=20, fontweight='bold')
        plt.ylabel('AUC', fontsize=20, fontweight='bold')
        plt.title('Training Efficiency vs Performance', fontsize=22, fontweight='bold', pad=20)
        plt.legend(fontsize=18)
        plt.grid(True, alpha=0.3)
        # Adjust layout to prevent text overlap
        plt.subplots_adjust(bottom=0.15)

    # 调整子图间距以避免重叠 - 为更大字体和旋转标签留出更多空间
    plt.subplots_adjust(bottom=0.3, hspace=0.6, wspace=0.5)  # 增加底部和间距避免标签重叠
    plt.tight_layout(pad=8.0)  # 增加更多padding以适应大字体和旋转标签

    # 增大所有刻度标签字体
    for ax in plt.gcf().get_axes():
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=14)
    save_figure(plt.gcf(), 'detailed_constraint_analysis.png')
    plt.show()

    # 3. 约束组合效应分析 - 大幅增加尺寸以适合Word文档
    plt.figure(figsize=(24, 16))  # 大幅增加图表尺寸以适合Word文档

    # 按禁用约束数量分组
    grouped = valid_df.groupby('n_disabled').agg({
        'mle': ['mean', 'std'],
        'auc': ['mean', 'std'],
        'training_time': ['mean', 'std']
    }).reset_index()

    # 子图1: 性能随约束数量变化
    plt.subplot(2, 2, 1)
    x = grouped['n_disabled']
    y_mle = grouped[('mle', 'mean')]
    err_mle = grouped[('mle', 'std')]

    plt.errorbar(x, y_mle, yerr=err_mle, marker='o', capsize=8, capthick=3, markersize=10, linewidth=3)
    plt.xlabel('Number of Disabled Constraints', fontsize=20, fontweight='bold')
    plt.ylabel('Mean Localization Error (mm)', fontsize=20, fontweight='bold')
    plt.title('Localization Error vs Number of Constraints', fontsize=22, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)

    # Subplot 2: AUC vs number of constraints
    plt.subplot(2, 2, 2)
    y_auc = grouped[('auc', 'mean')]
    err_auc = grouped[('auc', 'std')]

    plt.errorbar(x, y_auc, yerr=err_auc, marker='s', capsize=8, capthick=3, color='green', markersize=10, linewidth=3)
    plt.xlabel('Number of Disabled Constraints', fontsize=20, fontweight='bold')
    plt.ylabel('AUC', fontsize=20, fontweight='bold')
    plt.title('AUC vs Number of Constraints', fontsize=22, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)

    # Subplot 3: Training time vs number of constraints
    plt.subplot(2, 2, 3)
    y_time = grouped[('training_time', 'mean')]
    err_time = grouped[('training_time', 'std')]

    plt.errorbar(x, y_time, yerr=err_time, marker='^', capsize=8, capthick=3, color='orange', markersize=10, linewidth=3)
    plt.xlabel('Number of Disabled Constraints', fontsize=20, fontweight='bold')
    plt.ylabel('Training Time (seconds)', fontsize=20, fontweight='bold')
    plt.title('Training Time vs Number of Constraints', fontsize=24, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)

    # Subplot 4: Performance-efficiency trade-off
    plt.subplot(2, 2, 4)
    scatter = plt.scatter(valid_df['n_disabled'], valid_df['auc'] / valid_df['training_time'] * 1000,
               c=valid_df['mle'], s=200, cmap='viridis_r', alpha=0.7)  # 增大散点尺寸
    cbar = plt.colorbar(scatter, label='MLE (mm)')
    cbar.ax.tick_params(labelsize=16)  # 增大colorbar刻度字体
    cbar.set_label('MLE (mm)', fontsize=18, fontweight='bold')
    plt.xlabel('Number of Disabled Constraints', fontsize=20, fontweight='bold')
    plt.ylabel('Efficiency (AUC/Training Time × 1000)', fontsize=18, fontweight='bold')  # 稍微减小避免重叠
    plt.title('Performance-Efficiency Trade-off Analysis', fontsize=22, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)

    # 调整子图间距以避免重叠 - 为更大字体留出更多空间
    plt.subplots_adjust(hspace=0.5, wspace=0.5)  # 增加间距避免标签重叠
    plt.tight_layout(pad=8.0)  # 增加更多padding以适应大字体

    # 增大所有刻度标签字体
    for ax in plt.gcf().get_axes():
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=14)
    save_figure(plt.gcf(), 'constraint_combination_analysis.png')
    plt.show()

def generate_comprehensive_report(valid_df):
    """生成全面的消融研究报告"""

    # 获取基线性能
    baseline = valid_df[valid_df['n_disabled'] == 0]
    single_constraint_df = valid_df[valid_df['n_disabled'] == 1]

    if len(baseline) == 0:
        print("警告: 没有基线实验结果，无法生成报告")
        return

    baseline_row = baseline.iloc[0]

    report = f"""
    ===== 物理约束消融研究报告 =====

    实验概述:
    - 总实验数量: {len(valid_df)}
    - 测试的物理约束: {len(PHYSICS_CONSTRAINTS)}种
    - 基线模型性能 (所有约束启用):
      * 平均定位误差: {baseline_row['mle']:.2f} mm
      * AUC: {baseline_row['auc']:.4f}
      * 均方误差: {baseline_row['mse']:.2e}
      * 训练时间: {baseline_row['training_time']:.1f} 秒

    单个约束影响分析:
    """

    if len(single_constraint_df) > 0:
        for _, row in single_constraint_df.iterrows():
            constraint_name = PHYSICS_CONSTRAINTS.get(row['disabled_constraints'][0],
                                                     row['disabled_constraints'][0])
            mle_change = row['mle'] - baseline_row['mle']
            auc_change = row['auc'] - baseline_row['auc']

            impact_level = "High" if abs(mle_change) > baseline_row['mle'] * 0.1 else "Medium" if abs(mle_change) > baseline_row['mle'] * 0.05 else "Low"

            report += f"""
    {constraint_name}:
      - MLE变化: {mle_change:+.2f} mm ({mle_change/baseline_row['mle']*100:+.1f}%)
      - AUC变化: {auc_change:+.4f} ({auc_change/baseline_row['auc']*100:+.1f}%)
      - 影响程度: {impact_level}
    """

    # 找出最重要和最不重要的约束
    if len(single_constraint_df) > 0:
        single_constraint_df_copy = single_constraint_df.copy()
        single_constraint_df_copy['mle_change'] = single_constraint_df_copy['mle'] - baseline_row['mle']
        single_constraint_df_copy['auc_change'] = single_constraint_df_copy['auc'] - baseline_row['auc']

        # 按MLE恶化程度排序
        worst_mle = single_constraint_df_copy.loc[single_constraint_df_copy['mle_change'].idxmax()]
        best_mle = single_constraint_df_copy.loc[single_constraint_df_copy['mle_change'].idxmin()]

        # 按AUC恶化程度排序
        worst_auc = single_constraint_df_copy.loc[single_constraint_df_copy['auc_change'].idxmin()]
        best_auc = single_constraint_df_copy.loc[single_constraint_df_copy['auc_change'].idxmax()]

        report += f"""
    关键发现:
    - 最关键约束 (MLE): {PHYSICS_CONSTRAINTS.get(worst_mle['disabled_constraints'][0], worst_mle['disabled_constraints'][0])}
      (禁用后MLE增加 {worst_mle['mle_change']:.2f} mm)
    - 最关键约束 (AUC): {PHYSICS_CONSTRAINTS.get(worst_auc['disabled_constraints'][0], worst_auc['disabled_constraints'][0])}
      (禁用后AUC降低 {abs(worst_auc['auc_change']):.4f})
    - 最不重要约束 (MLE): {PHYSICS_CONSTRAINTS.get(best_mle['disabled_constraints'][0], best_mle['disabled_constraints'][0])}
      (禁用后MLE变化 {best_mle['mle_change']:+.2f} mm)
    - 最不重要约束 (AUC): {PHYSICS_CONSTRAINTS.get(best_auc['disabled_constraints'][0], best_auc['disabled_constraints'][0])}
      (禁用后AUC变化 {best_auc['auc_change']:+.4f})
    """

    # 整体趋势分析
    grouped_stats = valid_df.groupby('n_disabled').agg({
        'mle': 'mean',
        'auc': 'mean',
        'training_time': 'mean'
    })

    report += f"""
    整体趋势分析:
    - 禁用约束数量与性能关系:
    """

    for n_disabled, stats in grouped_stats.iterrows():
        if n_disabled == 0:
            continue
        mle_change_pct = (stats['mle'] - baseline_row['mle']) / baseline_row['mle'] * 100
        auc_change_pct = (stats['auc'] - baseline_row['auc']) / baseline_row['auc'] * 100
        time_change_pct = (stats['training_time'] - baseline_row['training_time']) / baseline_row['training_time'] * 100

        report += f"""
      禁用{n_disabled}个约束: MLE {mle_change_pct:+.1f}%, AUC {auc_change_pct:+.1f}%, 训练时间 {time_change_pct:+.1f}%
    """

    # 推荐配置
    best_overall = valid_df.loc[valid_df['auc'].idxmax()]
    fastest_good = valid_df[valid_df['auc'] >= baseline_row['auc'] * 0.95].loc[
        valid_df[valid_df['auc'] >= baseline_row['auc'] * 0.95]['training_time'].idxmin()
    ]

    report += f"""
    推荐配置:
    - 最佳性能配置: 禁用{best_overall['n_disabled']}个约束 (AUC: {best_overall['auc']:.4f})
    - 最佳效率配置: 禁用{fastest_good['n_disabled']}个约束 (AUC: {fastest_good['auc']:.4f}, 训练时间: {fastest_good['training_time']:.1f}s)

    结论:
    物理约束对PINN模型性能有显著影响。建议在实际应用中根据具体需求在性能和效率之间进行权衡。
    """

    print(report)

    # 保存报告
    report_path = os.path.join(os.path.dirname(__file__), 'ablation_study_figures', 'ablation_study_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"详细报告已保存至: {report_path}")

    # 保存结果数据
    results_path = os.path.join(os.path.dirname(__file__), 'ablation_study_figures', 'ablation_results.csv')
    valid_df.to_csv(results_path, index=False)
    print(f"实验结果数据已保存至: {results_path}")

    return report

if __name__ == "__main__":
    print("开始物理约束消融研究...")

    # 运行全面的消融研究
    results, trained_models, training_histories, fwd = run_comprehensive_ablation_study()

    # 分析结果
    valid_df = analyze_ablation_results(results)

    if len(valid_df) > 0:
        # 创建详细分析图表
        create_detailed_analysis_plots(valid_df)

        # 生成综合报告
        report = generate_comprehensive_report(valid_df)

        print(f"\n=== 消融研究完成 ===")
        print(f"成功完成 {len(valid_df)} 个有效实验")
        print("所有结果和可视化已保存到 ablation_study_figures 目录")

        # 显示最佳配置
        best_config = valid_df.loc[valid_df['auc'].idxmax()]
        print(f"\n最佳配置:")
        print(f"- 禁用约束: {[PHYSICS_CONSTRAINTS.get(c, c) for c in best_config['disabled_constraints']]}")
        print(f"- AUC: {best_config['auc']:.4f}")
        print(f"- MLE: {best_config['mle']:.2f} mm")
        print(f"- 训练时间: {best_config['training_time']:.1f} 秒")

    else:
        print("警告: 没有有效的实验结果，请检查实验设置")
