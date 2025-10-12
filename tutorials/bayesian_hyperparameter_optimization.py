# 基于BCIIV2a.py的贝叶斯超参数优化脚本
# 使用高斯过程优化PINN模型的关键超参数

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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C
from scipy.optimize import minimize
import pandas as pd
import seaborn as sns
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# 强制设置PyVista 3D后端
mne.viz.set_3d_backend('pyvista')

# # 固定随机种子，保证实验可复现
# np.random.seed(42)
# random.seed(42)
# tf.random.set_seed(42)

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
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14  # 增大默认字体大小
plt.rcParams['axes.titlesize'] = 16  # 增大标题字体
plt.rcParams['axes.labelsize'] = 14  # 增大轴标签字体
plt.rcParams['xtick.labelsize'] = 12  # 增大x轴刻度字体
plt.rcParams['ytick.labelsize'] = 12  # 增大y轴刻度字体
plt.rcParams['legend.fontsize'] = 12  # 增大图例字体

def ensure_font_settings():
    """确保matplotlib字体设置正确，解决负号显示问题"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'sans-serif'
    # 额外设置确保负号正确显示
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams['mathtext.fontset'] = 'stix'
    # 确保字体大小足够大
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12

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

def save_figure(fig, filename, dpi=300):
    """保存图形到指定目录"""
    fig_dir = os.path.join(os.path.dirname(__file__), 'bayesian_optimization_figures')
    os.makedirs(fig_dir, exist_ok=True)
    filepath = os.path.join(fig_dir, filename)
    # 确保保存时字体设置正确
    ensure_font_settings()
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"图形已保存至: {filepath}")
    return filepath

def get_bci_iv_2a_channel_names():
    """获取BCI-IV-2a数据集的电极名称"""
    ch_names = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
    ]
    return ch_names



# 贝叶斯优化类
class BayesianOptimizer:
    def __init__(self, parameter_bounds, acquisition_function='ei', n_initial_points=5):
        """
        初始化贝叶斯优化器
        
        Parameters:
        -----------
        parameter_bounds : dict
            参数边界，格式为 {'param_name': (min_val, max_val)}
        acquisition_function : str
            采集函数类型 ('ei', 'pi', 'ucb')
        n_initial_points : int
            初始随机采样点数量
        """
        self.parameter_bounds = parameter_bounds
        self.param_names = list(parameter_bounds.keys())
        self.bounds = np.array([parameter_bounds[name] for name in self.param_names])
        self.acquisition_function = acquisition_function
        self.n_initial_points = n_initial_points
        
        # 存储历史数据
        self.X_observed = []
        self.y_observed = []
        
        # 高斯过程模型
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=10,
            random_state=42
        )
        
    def _normalize_params(self, params):
        """将参数标准化到[0,1]区间"""
        normalized = []
        for i, param in enumerate(params):
            min_val, max_val = self.bounds[i]
            normalized.append((param - min_val) / (max_val - min_val))
        return np.array(normalized)
    
    def _denormalize_params(self, normalized_params):
        """将标准化参数转换回原始范围"""
        params = []
        for i, norm_param in enumerate(normalized_params):
            min_val, max_val = self.bounds[i]
            params.append(norm_param * (max_val - min_val) + min_val)
        return np.array(params)
    
    def _acquisition_ei(self, X, xi=0.01):
        """期望改进采集函数（针对最小化问题）"""
        mu, sigma = self.gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)

        if len(self.y_observed) == 0:
            return sigma

        f_best = np.min(self.y_observed)  # 最小化问题：找最小值

        with np.errstate(divide='warn'):
            imp = f_best - mu - xi  # 最小化问题：改进是当前值小于最佳值
            Z = imp / sigma.reshape(-1, 1)
            ei = imp * norm.cdf(Z) + sigma.reshape(-1, 1) * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei.flatten()
    
    def _acquisition_pi(self, X, xi=0.01):
        """改进概率采集函数（针对最小化问题）"""
        mu, sigma = self.gp.predict(X, return_std=True)

        if len(self.y_observed) == 0:
            return np.ones(len(X))

        f_best = np.min(self.y_observed)  # 最小化问题：找最小值

        with np.errstate(divide='warn'):
            Z = (f_best - mu - xi) / sigma  # 最小化问题：改进是当前值小于最佳值
            pi = norm.cdf(Z)
            pi[sigma == 0.0] = 0.0

        return pi
    
    def _acquisition_ucb(self, X, kappa=2.576):
        """置信下界采集函数（针对最小化问题）"""
        mu, sigma = self.gp.predict(X, return_std=True)
        return mu - kappa * sigma  # 最小化问题：使用置信下界
    
    def _get_acquisition(self, X):
        """获取采集函数值"""
        if self.acquisition_function == 'ei':
            return self._acquisition_ei(X)
        elif self.acquisition_function == 'pi':
            return self._acquisition_pi(X)
        elif self.acquisition_function == 'ucb':
            return self._acquisition_ucb(X)
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition_function}")
    
    def suggest_next_point(self):
        """建议下一个采样点"""
        if len(self.X_observed) < self.n_initial_points:
            # 随机采样初始点
            random_point = np.random.uniform(0, 1, len(self.param_names))
            return self._denormalize_params(random_point)
        
        # 使用采集函数优化
        def objective(x):
            if self.acquisition_function == 'ucb':
                # UCB已经返回负值（用于最小化），所以直接返回
                return self._get_acquisition(x.reshape(1, -1))[0]
            else:
                # EI和PI需要最大化，所以取负值
                return -self._get_acquisition(x.reshape(1, -1))[0]
        
        # 多起点优化
        best_x = None
        best_val = np.inf
        
        for _ in range(10):
            x0 = np.random.uniform(0, 1, len(self.param_names))
            result = minimize(objective, x0, bounds=[(0, 1)] * len(self.param_names), 
                            method='L-BFGS-B')
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x
        
        return self._denormalize_params(best_x)
    
    def update(self, params, score):
        """更新观测数据"""
        normalized_params = self._normalize_params(params)
        self.X_observed.append(normalized_params)
        self.y_observed.append(score)
        
        # 重新训练高斯过程
        if len(self.X_observed) > 0:
            X = np.array(self.X_observed)
            y = np.array(self.y_observed)
            self.gp.fit(X, y)
    
    def get_best_params(self):
        """获取最佳参数（最小化问题）"""
        if len(self.y_observed) == 0:
            return None, None

        best_idx = np.argmin(self.y_observed)  # 最小化问题：找最小值
        best_params = self._denormalize_params(self.X_observed[best_idx])
        best_score = self.y_observed[best_idx]

        return dict(zip(self.param_names, best_params)), best_score

# 从scipy.stats导入norm
from scipy.stats import norm

def get_or_create_simulation(fwd, epochs, sim_settings, n_samples, data_type="train"):
    """
    获取或创建模拟数据，使用pickle文件进行持久化存储

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
    # ========== 模拟数据保存/加载 (类似BCIIV2a.py) ==========
    sim_filename = f'simulation_{data_type}.pkl'

    if os.path.exists(sim_filename):
        print(f'🔍 检测到已保存的{data_type}模拟数据，尝试加载: {sim_filename}')
        try:
            with open(sim_filename, 'rb') as f:
                simulation = pickle.load(f)

            # 验证数据完整性
            if (hasattr(simulation, 'eeg_data') and hasattr(simulation, 'source_data') and
                len(simulation.eeg_data) > 0 and len(simulation.source_data) > 0):
                print(f'✅ {data_type}模拟数据加载成功 ({len(simulation.eeg_data)} 样本)')
            else:
                print(f'❌ {data_type}模拟数据不完整，将重新生成')
                os.remove(sim_filename)
                raise ValueError("数据不完整")

        except Exception as e:
            print(f'❌ 加载{data_type}模拟数据失败: {e}，将重新生成')
            try:
                os.remove(sim_filename)
            except:
                pass
            # 重新生成数据
            print(f'🔄 重新生成{data_type}模拟数据...')
            simulation = Simulation(fwd, epochs.info, settings=sim_settings, verbose=False)
            simulation.simulate(n_samples=n_samples)

            # 保存新数据
            with open(sim_filename, 'wb') as f:
                pickle.dump(simulation, f)
            print(f'💾 {data_type}模拟数据已保存到: {sim_filename}')
    else:
        print(f'🔄 未检测到{data_type}模拟数据，开始生成...')
        simulation = Simulation(fwd, epochs.info, settings=sim_settings, verbose=False)
        simulation.simulate(n_samples=n_samples)

        # 保存模拟数据
        with open(sim_filename, 'wb') as f:
            pickle.dump(simulation, f)
        print(f'💾 {data_type}模拟数据已保存到: {sim_filename}')

    return simulation

def objective_function(params, fwd, epochs, simulation_settings, n_samples=200):
    """
    目标函数：训练PINN模型并返回MLE（平均定位误差）

    Parameters:
    -----------
    params : dict
        超参数字典
    fwd : mne.Forward
        前向模型
    epochs : mne.Epochs
        EEG数据
    simulation_settings : dict
        模拟设置
    n_samples : int
        模拟样本数量

    Returns:
    --------
    mle : float
        平均定位误差（越小越好）
    """
    net = None
    try:
        print(f"正在评估参数组合: {params}")

        # 清理之前的内存
        tf.keras.backend.clear_session()
        import gc
        gc.collect()

        # 创建训练模拟数据（使用持久化存储）
        # 注意：使用预先生成的模拟数据，SNR已固定在simulation_settings中
        print("📊 准备训练数据...")
        simulation = get_or_create_simulation(fwd, epochs, simulation_settings, n_samples, "train")

        # 验证训练数据
        print(f"✅ 训练数据验证: {len(simulation.eeg_data)} 个EEG样本, {len(simulation.source_data)} 个源数据")
        if len(simulation.eeg_data) == 0 or len(simulation.source_data) == 0:
            raise ValueError("训练数据为空！")
        
        # 创建网络
        net = Net(fwd, verbose=0, model_type='pinn')
        
        # 训练网络（使用固定的训练轮数）
        fixed_epochs = 200  # 固定训练轮数，减少搜索空间
        print(f"🚀 开始训练网络: epochs={fixed_epochs}, batch_size={int(params['batch_size'])}, "
              f"lr={params['learning_rate']:.2e}, dropout={params['dropout']:.2f}")

        with tf.device('/GPU:0' if gpus else '/CPU:0'):
            start_time = time.time()
            try:
                net, history = net.fit(simulation,
                                     epochs=fixed_epochs,
                                     batch_size=int(params['batch_size']),
                                     learning_rate=params['learning_rate'],
                                     dropout=params['dropout'],
                                     patience=int(params['patience']),
                                     return_history=True)
                training_time = time.time() - start_time
                print(f"✅ 训练完成，耗时: {training_time:.1f}s")

                # 检查训练历史
                if hasattr(history, 'history'):
                    final_loss = history.history.get('loss', [])[-1] if history.history.get('loss') else None
                    print(f"最终训练损失: {final_loss}")

            except Exception as train_error:
                print(f"❌ 训练过程失败: {train_error}")
                raise train_error

        # 创建测试数据（使用持久化存储）
        print("📊 准备测试数据...")
        test_settings = simulation_settings.copy()
        test_settings['number_of_sources'] = 1

        test_simulation = get_or_create_simulation(fwd, epochs, test_settings, 50, "test")

        # 验证测试数据
        print(f"✅ 测试数据验证: {len(test_simulation.eeg_data)} 个EEG样本, {len(test_simulation.source_data)} 个源数据")
        if len(test_simulation.eeg_data) == 0 or len(test_simulation.source_data) == 0:
            raise ValueError("测试数据为空！")
        
        # 预测
        print("🔮 开始预测...")
        source_hat = net.predict(test_simulation)
        print(f"✅ 预测完成: {len(source_hat)} 个预测结果")

        # 验证预测结果
        if len(source_hat) == 0:
            raise ValueError("预测结果为空！")

        # 检查预测结果的基本统计信息
        for i in range(min(3, len(source_hat))):
            pred_data = source_hat[i].data
            print(f"预测样本 {i}: shape={pred_data.shape}, 范围=[{pred_data.min():.2e}, {pred_data.max():.2e}]")
        
        # 计算评价指标
        _, _, pos, _ = util.unpack_fwd(fwd)
        
        mle_values = []
        mse_values = []
        auc_values = []
        
        for i in range(len(source_hat)):
            y_true = test_simulation.source_data[i].data[:, 0]
            y_est = source_hat[i].data[:, 0]
            
            mle = eval_mean_localization_error(y_true, y_est, pos)
            mse = eval_mse(y_true, y_est)
            auc_close, auc_far = eval_auc(y_true, y_est, pos)
            
            if not np.isnan(mle):
                mle_values.append(mle)
            if not np.isnan(mse):
                mse_values.append(mse)
            if not np.isnan(auc_close):
                auc_values.append(auc_close)
            if not np.isnan(auc_far):
                auc_values.append(auc_far)
        
        # 计算平均定位误差（越小越好）
        if mle_values:
            avg_mle = np.mean(mle_values)
            print(f"✅ 成功计算 {len(mle_values)} 个有效MLE值")
        else:
            print("❌ 警告：未获得任何有效MLE值！")
            print(f"预测结果数量: {len(source_hat)}")
            print(f"测试数据数量: {len(test_simulation.source_data)}")

            # 诊断问题
            if len(source_hat) == 0:
                print("❌ 模型预测失败：没有预测结果")
                raise ValueError("模型预测失败：没有预测结果")
            else:
                print("🔍 检查预测结果质量...")
                for i in range(min(3, len(source_hat))):  # 检查前3个样本
                    y_true = test_simulation.source_data[i].data[:, 0]
                    y_est = source_hat[i].data[:, 0]
                    print(f"样本 {i}: y_true范围=[{y_true.min():.2e}, {y_true.max():.2e}], "
                          f"y_est范围=[{y_est.min():.2e}, {y_est.max():.2e}]")

                    mle_test = eval_mean_localization_error(y_true, y_est, pos)
                    print(f"样本 {i} MLE: {mle_test} (是否为NaN: {np.isnan(mle_test)})")

            # 抛出异常而不是返回惩罚值
            raise ValueError(f"所有 {len(source_hat)} 个预测结果的MLE都是NaN，训练可能失败")

        avg_mse = np.mean(mse_values) if mse_values else 1.0
        avg_auc = np.mean(auc_values) if auc_values else 0.5

        # 额外的合理性检查
        if avg_mle > 100.0:  # 如果MLE异常高，也认为是失败
            raise ValueError(f"MLE值异常高: {avg_mle:.2f}mm，可能训练失败")

        print(f"参数评估完成 - MLE: {avg_mle:.2f}mm, AUC: {avg_auc:.4f}, MSE: {avg_mse:.2e}, 训练时间: {training_time:.1f}s")

        # 返回MLE作为目标函数（贝叶斯优化将最小化此值）
        return avg_mle

    except Exception as e:
        print(f"❌ 参数评估失败: {e}")
        print(f"失败的参数组合: {params}")
        import traceback
        print("详细错误信息:")
        traceback.print_exc()
        # 重新抛出异常，让上层处理（保险机制会舍弃这次结果）
        raise e

    finally:
        # 强制清理内存和GPU资源
        print("🧹 清理内存和GPU资源...")

        if net is not None:
            try:
                del net
            except:
                pass

        # 清理TensorFlow会话
        tf.keras.backend.clear_session()

        # 强制垃圾回收
        import gc
        gc.collect()

        # 如果使用GPU，强制清理GPU内存
        if gpus:
            try:
                # 重置GPU内存
                tf.config.experimental.reset_memory_growth(gpus[0])
                tf.config.experimental.set_memory_growth(gpus[0], True)

                # 清理GPU缓存
                if hasattr(tf.config.experimental, 'reset_memory_stats'):
                    tf.config.experimental.reset_memory_stats(gpus[0])

                print("✅ GPU内存清理完成")
            except Exception as gpu_error:
                print(f"⚠️ GPU内存清理警告: {gpu_error}")

        print("✅ 内存清理完成")

def run_bayesian_optimization():
    """运行贝叶斯优化主流程"""
    print("开始贝叶斯超参数优化...")
    
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
        src = mne.setup_source_space(subject, spacing='oct3', 
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
    
    print(f"前向模型创建完成，包含 {fwd['sol']['data'].shape[1]} 个源点")
    
    # 定义超参数搜索空间（基于PINN网络特点的合理范围）
    parameter_bounds = {
        'learning_rate': (5e-5, 1e-2),       # 学习率：扩大范围，探索更多可能性
                                              # 5e-5 (0.00005) 到 1e-2 (0.01)
        'batch_size': (8, 32),                # 批大小：扩大范围，适应不同数据规模
                                              # 小批量(8-16)有助于泛化，大批量(32-64)训练稳定
        'dropout': (0.1, 0.5),               # Dropout率：扩大范围，探索正则化效果
                                              # 0.1-0.3轻度正则化，0.4-0.7强正则化
        'patience': (10, 30),                # 早停耐心：适度扩展，平衡训练时间和性能
                                              # 10-15快速收敛，20-30充分训练
        # 注意：移除了target_snr，因为使用预先生成的模拟数据，SNR是固定的
        # 注意：移除了epochs，使用固定的训练轮数以减少搜索空间
    }

    print("📊 超参数搜索空间:")
    for param, (min_val, max_val) in parameter_bounds.items():
        print(f"  {param}: [{min_val}, {max_val}]")
    
    # 模拟设置
    simulation_settings = {
        'duration_of_trial': 4.0,
        'number_of_sources': 1,
        'extents': (5, 15),
        'beta_source': (1, 1.5),
        'source_time_course': 'sine',
        'target_snr': 3.0  # 固定SNR值，因为使用预先生成的模拟数据
    }
    
    # 初始化贝叶斯优化器
    optimizer = BayesianOptimizer(parameter_bounds, acquisition_function='ei', n_initial_points=5)
    
    # 运行优化
    n_iterations = 50  # 总迭代次数
    results = []
    
    consecutive_failures = 0  # 连续失败计数
    max_consecutive_failures = 3  # 最大连续失败次数
    successful_iterations = 0  # 成功迭代计数

    iteration = 0
    while successful_iterations < n_iterations:
        iteration += 1
        print(f"\n=== 贝叶斯优化尝试 {iteration} (成功: {successful_iterations}/{n_iterations}) ===")

        # 在每次迭代前进行内存清理
        if iteration > 1:
            print("🧹 迭代前内存清理...")
            tf.keras.backend.clear_session()
            import gc
            gc.collect()

        # 获取下一个参数组合
        next_params = optimizer.suggest_next_point()
        param_dict = dict(zip(optimizer.param_names, next_params))

        # 评估参数
        training_successful = False
        score = None

        try:
            score = objective_function(param_dict, fwd, epochs, simulation_settings)

            # 检查是否为有效的训练结果
            if score < 50.0:  # 认为MLE < 50mm为成功
                training_successful = True
                consecutive_failures = 0  # 成功时重置计数
                print(f"✅ 训练成功，MLE: {score:.4f}mm")
            else:
                print(f"❌ 训练失败，MLE过高: {score:.4f}mm")
                consecutive_failures += 1

        except Exception as eval_error:
            print(f"❌ 参数评估异常: {eval_error}")
            consecutive_failures += 1

        # 保险机制：只有训练成功的结果才参与优化
        if training_successful and score is not None:
            # 更新优化器（只用成功的结果）
            optimizer.update(next_params, score)

            # 记录成功的结果
            result = param_dict.copy()
            result['mle'] = score
            result['iteration'] = successful_iterations + 1  # 使用成功迭代计数
            result['actual_attempt'] = iteration  # 记录实际尝试次数
            results.append(result)

            successful_iterations += 1

            # 获取当前最佳参数
            best_params, best_score = optimizer.get_best_params()
            if best_params:
                print(f"🏆 历史最佳MLE: {best_score:.4f}mm")
                print(f"📊 成功率: {successful_iterations}/{iteration} = {successful_iterations/iteration*100:.1f}%")
        else:
            print(f"🗑️ 舍弃失败结果，不更新优化器")
            print(f"⚠️ 连续失败: {consecutive_failures}/{max_consecutive_failures}")

            # 如果连续失败太多次，尝试重置环境
            if consecutive_failures >= max_consecutive_failures:
                print("🔄 连续失败过多，尝试重置训练环境...")

                # 强制清理所有资源
                tf.keras.backend.clear_session()
                import gc
                gc.collect()

                # 重置GPU（如果有）
                if gpus:
                    try:
                        tf.config.experimental.reset_memory_growth(gpus[0])
                        tf.config.experimental.set_memory_growth(gpus[0], True)
                    except:
                        pass

                consecutive_failures = 0  # 重置计数
                print("✅ 环境重置完成")

        # 安全机制：防止无限循环
        if iteration > n_iterations * 3:  # 最多尝试3倍的目标迭代数
            print(f"⚠️ 达到最大尝试次数 ({iteration})，停止优化")
            print(f"📊 最终成功率: {successful_iterations}/{iteration} = {successful_iterations/iteration*100:.1f}%")
            break
    
    return results, optimizer

def analyze_and_visualize_results(results, optimizer):
    """Analyze and visualize Bayesian optimization results"""
    print("\n=== Starting Results Analysis and Visualization ===")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # 确保按实验配置的时间顺序排列（而不是按结果排列）
    df = df.sort_values('iteration').reset_index(drop=True)
    print(f"📊 数据验证: {len(df)} 个成功实验，按时间顺序排列")
    print(f"迭代顺序: {df['iteration'].tolist()}")
    print(f"MLE顺序: {df['mle'].round(3).tolist()}")

    # 1. Optimization convergence curve
    ensure_font_settings()  # 确保字体设置正确
    plt.figure(figsize=(16, 10))  # 增大图形尺寸

    # Subplot 1: MLE vs iteration (现在只包含成功的结果)
    plt.subplot(2, 2, 1)

    # 由于保险机制，现在所有结果都应该是有效的
    mle_values = df['mle'].copy()

    plt.plot(df['iteration'], mle_values, 'go-', linewidth=3, markersize=8, label='MLE')

    # Add cumulative best line (minimum MLE)
    cumulative_best = mle_values.cummin()
    plt.plot(df['iteration'], cumulative_best, 'r--', linewidth=3, label='Cumulative Best')

    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Mean Localization Error (mm)', fontsize=14)
    plt.title('Bayesian Optimization Convergence', fontsize=16, pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # 确保x轴显示整数刻度
    plt.xticks(range(1, len(df) + 1, max(1, len(df) // 10)))

    # 设置合理的Y轴范围
    plt.ylim(0, mle_values.max() * 1.1)

    # Subplot 2: Parameter importance analysis
    plt.subplot(2, 2, 2)
    # 只包含真正的超参数，排除记录性变量
    param_cols = [col for col in df.columns if col not in ['mle', 'iteration', 'actual_attempt']]
    correlations = []
    for param in param_cols:
        corr = df[param].corr(df['mle'])
        correlations.append(abs(corr))

    # 创建参数重要性排序
    param_importance = list(zip(param_cols, correlations))
    param_importance.sort(key=lambda x: x[1], reverse=True)

    # 获取最重要的两个参数
    most_important_params = [param_importance[0][0], param_importance[1][0]]
    print(f"📊 最重要的两个超参数: {most_important_params[0]} (相关性: {param_importance[0][1]:.3f}), {most_important_params[1]} (相关性: {param_importance[1][1]:.3f})")

    plt.barh(param_cols, correlations)
    plt.xlabel('Correlation with MLE (Absolute)', fontsize=14)
    plt.title('Hyperparameter Importance Analysis', fontsize=16, pad=20)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)

    # Subplot 3: 最重要参数1 vs performance
    plt.subplot(2, 2, 3)
    param1 = most_important_params[0]
    scatter = plt.scatter(df[param1], df['mle'], c=df['iteration'],
                         cmap='viridis_r', s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    cbar = plt.colorbar(scatter, label='Iteration')
    cbar.ax.tick_params(labelsize=12)
    plt.xlabel(param1.replace('_', ' ').title(), fontsize=14)
    plt.ylabel('Mean Localization Error (mm)', fontsize=14)
    plt.title(f'{param1.replace("_", " ").title()} Impact on Performance', fontsize=16, pad=20)

    # 根据参数类型设置合适的x轴
    if 'learning_rate' in param1.lower():
        plt.xscale('log')
        x_ticks = [5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
        plt.xticks(x_ticks, ['5e-5', '1e-4', '2e-4', '5e-4', '1e-3', '2e-3', '5e-3', '1e-2'])
    elif 'batch_size' in param1.lower():
        x_ticks = [8, 16, 24, 32, 48, 64]
        plt.xticks(x_ticks)

    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)

    # Subplot 4: 最重要参数2 vs performance
    plt.subplot(2, 2, 4)
    param2 = most_important_params[1]
    scatter = plt.scatter(df[param2], df['mle'], c=df['iteration'],
                         cmap='viridis_r', s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    cbar = plt.colorbar(scatter, label='Iteration')
    cbar.ax.tick_params(labelsize=12)
    plt.xlabel(param2.replace('_', ' ').title(), fontsize=14)
    plt.ylabel('Mean Localization Error (mm)', fontsize=14)
    plt.title(f'{param2.replace("_", " ").title()} Impact on Performance', fontsize=16, pad=20)

    # 根据参数类型设置合适的x轴
    if 'learning_rate' in param2.lower():
        plt.xscale('log')
        x_ticks = [5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
        plt.xticks(x_ticks, ['5e-5', '1e-4', '2e-4', '5e-4', '1e-3', '2e-3', '5e-3', '1e-2'])
    elif 'batch_size' in param2.lower():
        x_ticks = [8, 16, 24, 32, 48, 64]
        plt.xticks(x_ticks)

    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout(pad=3.0)  # 增加间距避免重叠
    save_figure(plt.gcf(), 'bayesian_optimization_analysis.png')
    plt.show()

    # 2. Parameter correlation heatmap
    ensure_font_settings()  # 确保字体设置正确
    plt.figure(figsize=(12, 10))  # 增大图形尺寸
    correlation_matrix = df[param_cols + ['mle']].corr()

    # 修复：显示完整的相关性矩阵，而不是只显示下三角
    # mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # 原来只显示下三角
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',
                center=0, square=True, linewidths=0.5, fmt='.2f',
                annot_kws={'size': 12})  # 增大注释字体
    plt.title('Hyperparameter Correlation Matrix', fontsize=16, pad=20)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    save_figure(plt.gcf(), 'parameter_correlation_heatmap.png')
    plt.show()

    # 3. Gaussian process prediction visualization (top 2 most important parameters)
    # 使用之前计算的最重要参数
    param1, param2 = most_important_params[0], most_important_params[1]

    ensure_font_settings()  # 确保字体设置正确
    plt.figure(figsize=(16, 8))  # 增加宽度以避免标题重叠

    # Create grid
    param1_range = np.linspace(df[param1].min(), df[param1].max(), 50)
    param2_range = np.linspace(df[param2].min(), df[param2].max(), 50)
    P1, P2 = np.meshgrid(param1_range, param2_range)

    # 准备预测数据
    grid_points = []
    for i in range(len(param1_range)):
        for j in range(len(param2_range)):
            point = np.zeros(len(param_cols))
            # 使用最佳参数作为其他维度的默认值
            best_params, _ = optimizer.get_best_params()
            for k, param in enumerate(param_cols):
                if param == param1:
                    point[k] = param1_range[i]
                elif param == param2:
                    point[k] = param2_range[j]
                else:
                    point[k] = best_params[param]
            grid_points.append(point)

    grid_points = np.array(grid_points)

    # 标准化网格点
    normalized_grid = []
    for point in grid_points:
        normalized_point = optimizer._normalize_params(point)
        normalized_grid.append(normalized_point)

    normalized_grid = np.array(normalized_grid)

    # 预测
    if len(optimizer.X_observed) > 0:
        predictions, std = optimizer.gp.predict(normalized_grid, return_std=True)
        predictions = predictions.reshape(50, 50)
        std = std.reshape(50, 50)

        # Subplot 1: Prediction mean
        plt.subplot(1, 2, 1)
        contour = plt.contourf(P1, P2, predictions, levels=20, cmap='viridis')
        cbar = plt.colorbar(contour, label='Predicted Objective Value')
        cbar.ax.tick_params(labelsize=12)

        # Add observation points
        plt.scatter(df[param1], df[param2], c=df['mle'],
                   s=100, edgecolors='white', linewidth=2, cmap='viridis_r')

        plt.xlabel(param1.replace('_', ' ').title(), fontsize=14)
        plt.ylabel(param2.replace('_', ' ').title(), fontsize=14)
        plt.title('Gaussian Process Prediction Mean', fontsize=16, pad=20)
        plt.tick_params(axis='both', which='major', labelsize=12)

        # Subplot 2: Prediction uncertainty
        plt.subplot(1, 2, 2)
        contour_std = plt.contourf(P1, P2, std, levels=20, cmap='Reds')
        cbar = plt.colorbar(contour_std, label='Prediction Standard Deviation')
        cbar.ax.tick_params(labelsize=12)

        # Add observation points
        plt.scatter(df[param1], df[param2], c='black', s=80, alpha=0.7, edgecolors='white', linewidth=1)

        plt.xlabel(param1.replace('_', ' ').title(), fontsize=14)
        plt.ylabel(param2.replace('_', ' ').title(), fontsize=14)
        plt.title('Gaussian Process Prediction Uncertainty', fontsize=16, pad=20)
        plt.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout(pad=3.0)  # 增加padding避免重叠
    save_figure(plt.gcf(), f'gaussian_process_prediction_{param1}_{param2}.png')
    plt.show()

    # 4. Optimal parameter configuration summary
    best_params, best_score = optimizer.get_best_params()

    ensure_font_settings()  # 确保字体设置正确
    plt.figure(figsize=(12, 8))

    # Create radar chart for optimal parameters
    angles = np.linspace(0, 2 * np.pi, len(param_cols), endpoint=False).tolist()
    angles += angles[:1]  # Close the plot

    # Normalize parameter values to [0,1]
    normalized_best = []
    for param in param_cols:
        min_val = df[param].min()
        max_val = df[param].max()
        norm_val = (best_params[param] - min_val) / (max_val - min_val)
        normalized_best.append(norm_val)
    normalized_best += normalized_best[:1]

    ax = plt.subplot(111, projection='polar')
    ax.plot(angles, normalized_best, 'o-', linewidth=2, label='Optimal Hyperparameters')
    ax.fill(angles, normalized_best, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(param_cols)
    ax.set_ylim(0, 1)
    ax.set_title('Optimal Hyperparameter Configuration', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    save_figure(plt.gcf(), 'best_parameters_radar.png')
    plt.show()

    # 5. Generate detailed report
    report = f"""
    ===== Bayesian Hyperparameter Optimization Report =====

    Optimization Configuration:
    - Total iterations: {len(results)}
    - Acquisition function: {optimizer.acquisition_function}
    - Initial random points: {optimizer.n_initial_points}

    Best Results:
    - Best MLE (Mean Localization Error): {best_score:.6f} mm
    - Optimal parameter configuration:
    """

    for param, value in best_params.items():
        if param in ['batch_size', 'patience']:
            report += f"      {param}: {int(value)}\n"
        else:
            report += f"      {param}: {value:.6f}\n"

    report += f"""
    Parameter Importance Ranking:
    """

    for i, (param, importance) in enumerate(sorted(zip(param_cols, correlations),
                                                  key=lambda x: x[1], reverse=True)):
        report += f"    {i+1}. {param}: {importance:.4f}\n"

    report += f"""
    Optimization Statistics:
    - Best MLE: {df['mle'].min():.6f} mm
    - Worst MLE: {df['mle'].max():.6f} mm
    - Mean MLE: {df['mle'].mean():.6f} mm
    - MLE standard deviation: {df['mle'].std():.6f} mm

    Convergence Analysis:
    - First 5 iterations mean MLE: {df['mle'][:5].mean():.6f} mm
    - Last 5 iterations mean MLE: {df['mle'][-5:].mean():.6f} mm
    - Improvement: {((df['mle'][:5].mean() - df['mle'][-5:].mean()) / abs(df['mle'][:5].mean()) * 100):.2f}%
    """

    print(report)

    # Save report
    report_path = os.path.join(os.path.dirname(__file__), 'bayesian_optimization_figures', 'optimization_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Detailed report saved to: {report_path}")

    # Save results data
    results_path = os.path.join(os.path.dirname(__file__), 'bayesian_optimization_figures', 'optimization_results.csv')
    df.to_csv(results_path, index=False)
    print(f"Optimization results data saved to: {results_path}")

    return df, best_params, best_score

if __name__ == "__main__":
    # Run Bayesian optimization
    results, optimizer = run_bayesian_optimization()

    # Analyze and visualize results
    df, best_params, best_score = analyze_and_visualize_results(results, optimizer)

    print(f"\n=== Bayesian Optimization Complete ===")
    print(f"Best MLE: {best_score:.4f} mm")
    print(f"Best parameters: {best_params}")

    # Final training and evaluation with optimal parameters
    print(f"\n=== Final Validation with Optimal Parameters ===")

    # Code for complete training and testing with optimal parameters can be added here
    # final_validation(best_params)
