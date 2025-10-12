# BCI-IV-2a源定位分析系统 - GUI版本
# 基于ESINet深度学习框架的图形用户界面

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
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import threading
import warnings
warnings.filterwarnings('ignore')

# 设置高DPI支持
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)  # 启用DPI感知
    print("✓ 已启用高DPI支持")
except:
    print("⚠ 无法启用高DPI支持，可能在高分辨率屏幕上显示模糊")

# 设置3D后端（安全模式）
try:
    import pyvista
    mne.viz.set_3d_backend('pyvista')
    print("✓ PyVista 3D后端加载成功")
    PYVISTA_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"⚠ PyVista加载失败: {e}")
    print("将跳过3D后端设置，使用默认显示")
    PYVISTA_AVAILABLE = False

# 设置matplotlib中文字体和高DPI
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100  # 设置matplotlib的DPI
plt.rcParams['savefig.dpi'] = 300  # 保存图片的DPI

# BCI-IV-2a电极位置定义
def get_bci_iv_2a_channel_names():
    """获取BCI-IV-2a数据集的电极名称"""
    ch_names = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
    ]
    return ch_names

def load_bci_iv_2a_data(data_path, subject_ids=None):
    """加载BCI-IV-2a数据集"""
    if subject_ids is None:
        mat_files = [f for f in os.listdir(data_path) if f.endswith('.mat')]
        print(f"检测到 {len(mat_files)} 个数据文件: {mat_files}")
    else:
        mat_files = [f"subject_{sid}.mat" for sid in subject_ids]
    
    all_epochs = []
    ch_names = get_bci_iv_2a_channel_names()
    
    for mat_file in mat_files:
        file_path = os.path.join(data_path, mat_file)
        if not os.path.exists(file_path):
            print(f"警告: 文件 {file_path} 不存在，跳过")
            continue
        
        try:
            mat_data = scipy.io.loadmat(file_path)
        except Exception as e:
            print(f"加载文件 {mat_file} 失败: {e}")
            continue
        
        # 提取数据
        if 'rawdata' in mat_data and 'label' in mat_data:
            raw_data = mat_data['rawdata']  # (1000, 22, 576)
            labels = mat_data['label'].flatten()  # (576,)
        else:
            data_key = None
            label_key = None
            for key, value in mat_data.items():
                if isinstance(value, np.ndarray):
                    if value.shape == (1000, 22, 576):
                        data_key = key
                    elif value.shape == (576, 1) or value.shape == (576,):
                        label_key = key
            if data_key is None or label_key is None:
                print(f"无法在 {mat_file} 中找到正确格式的数据，跳过")
                continue
            raw_data = mat_data[data_key]
            labels = mat_data[label_key].flatten()
            
        # 转换数据格式
        eeg_data = np.transpose(raw_data, (2, 1, 0))  # (576, 22, 1000)
        sfreq = 250
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        montage = mne.channels.make_standard_montage('standard_1020')
        info.set_montage(montage)
        valid_trials = labels != 0
        valid_eeg_data = eeg_data[valid_trials]
        valid_labels = labels[valid_trials]
        lr_trials = np.isin(valid_labels, [1, 2])
        final_eeg_data = valid_eeg_data[lr_trials]
        final_labels = valid_labels[lr_trials]
        n_final_trials = len(final_eeg_data)
        events = np.zeros((n_final_trials, 3), dtype=int)
        events[:, 0] = np.arange(n_final_trials) * 1000
        events[:, 2] = final_labels.astype(int)
        unique_labels = np.unique(final_labels.astype(int))
        if len(unique_labels) >= 2:
            if 1 in unique_labels and 2 in unique_labels:
                event_id = {'left_hand': 1, 'right_hand': 2}
            else:
                event_id = {f'condition_{i+1}': label for i, label in enumerate(unique_labels)}
        else:
            print(f"警告: 只检测到一种标签 {unique_labels}，跳过此文件")
            continue
            
        # 保留原始数据用于可视化
        eeg_data_for_plot = final_eeg_data.copy()
        
        # 标准化数据
        for trial in range(final_eeg_data.shape[0]):
            for ch in range(final_eeg_data.shape[1]):
                final_eeg_data[trial, ch, :] = (final_eeg_data[trial, ch, :] - np.mean(final_eeg_data[trial, ch, :])) / np.std(final_eeg_data[trial, ch, :])
        
        tmin = 0.0
        tmax = (final_eeg_data.shape[2] - 1) / sfreq
        
        # 创建Epochs对象
        epochs = mne.EpochsArray(
            final_eeg_data, info, events=events, event_id=event_id,
            tmin=tmin, verbose=False
        )
        epochs_plot = mne.EpochsArray(
            eeg_data_for_plot, info, events=events, event_id=event_id,
            tmin=tmin, verbose=False
        )
        all_epochs.append((epochs, epochs_plot))
    return all_epochs

def get_model_name(model_type):
    """返回标准化模型名称"""
    return model_type.upper()

class BCIIV2aGUI:
    """BCI-IV-2a源定位分析GUI界面"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("BCI-IV-2a源定位分析系统")
        
        # 默认窗口大小和位置
        self.default_width = 1400
        self.default_height = 900
        self.root.geometry(f"{self.default_width}x{self.default_height}")
        
        # 正常情况下允许窗口调整
        self.root.resizable(True, True)
        self.root.minsize(1200, 700)  # 设置最小尺寸
        
        # 3D可视化状态标志
        self._is_3d_active = False
        self._saved_geometry = None
        self._saved_resizable_state = None
        
        # 字体和缩放设置
        self.base_font_size = 9
        self.current_scale = 1.0
        self._setup_fonts()
        self._setup_high_dpi_fonts()  # 设置高DPI字体
        
        # 绑定窗口大小变化事件用于字体缩放
        self.root.bind('<Configure>', self._on_window_resize)
        
        # 初始化数据变量
        self.epochs_list = None
        self.epochs = None
        self.epochs_plot = None
        self.fwd = None
        self.simulation = None
        self.simulation_test = None
        self.net = None
        self.source_hat = None
        self.history = None
        self.subjects_dir = None
        self.training_thread = None
        
        # 初始化模拟参数
        self.sim_params = {
            'duration_of_trial': tk.DoubleVar(value=4.0),
            'target_snr': tk.IntVar(value=5),
            'number_of_sources': tk.IntVar(value=2),
            'extents_min': tk.IntVar(value=5),
            'extents_max': tk.IntVar(value=15),
            'beta_source_min': tk.DoubleVar(value=1.0),
            'beta_source_max': tk.DoubleVar(value=1.5),
            'train_samples': tk.IntVar(value=1000),
            'test_samples': tk.IntVar(value=100)
        }
        
        # 初始化训练参数
        self.train_params = {
            'learning_rate': tk.DoubleVar(value=0.001),
            'batch_size': tk.IntVar(value=64),
            'epochs': tk.IntVar(value=1000),
            'patience': tk.IntVar(value=15)
        }
        
        # 初始化评估结果
        self.results = {
            'mle': None,
            'mse': None,
            'auc': None
        }
        
        self.setup_gpu()
        self.create_widgets()
        self.setup_subjects_dir()
        self.create_status_bar()
        
    def _setup_fonts(self):
        """设置字体样式"""
        import tkinter.font as tkFont
        
        # 创建可缩放的字体
        self.default_font = tkFont.nametofont("TkDefaultFont")
        self.text_font = tkFont.nametofont("TkTextFont")
        self.fixed_font = tkFont.nametofont("TkFixedFont")
        
        # 保存原始字体大小
        self.original_default_size = self.default_font.cget("size")
        self.original_text_size = self.text_font.cget("size")
        self.original_fixed_size = self.fixed_font.cget("size")
        
    def _setup_high_dpi_fonts(self):
        """设置高DPI字体，解决全屏模糊问题"""
        try:
            import tkinter.font as tkFont
            
            # 检测系统DPI缩放比例
            try:
                from ctypes import windll
                user32 = windll.user32
                user32.SetProcessDPIAware()
                # 获取DPI缩放比例
                dpi = user32.GetDpiForWindow(self.root.winfo_id())
                scale_factor = dpi / 96.0  # 96是标准DPI
                print(f"检测到DPI缩放比例: {scale_factor:.2f}")
            except:
                scale_factor = 1.0
                print("无法检测DPI缩放比例，使用默认值")
            
            # 根据DPI缩放调整字体大小
            if scale_factor > 1.0:
                # 高DPI屏幕，增大字体
                self.base_font_size = int(9 * scale_factor)
                self.current_scale = scale_factor
                
                # 更新所有字体
                self.default_font.configure(size=self.base_font_size)
                self.text_font.configure(size=self.base_font_size)
                self.fixed_font.configure(size=self.base_font_size)
                
                # 设置ttk样式
                style = ttk.Style()
                style.configure('TLabel', font=('Microsoft YaHei', self.base_font_size))
                style.configure('TButton', font=('Microsoft YaHei', self.base_font_size))
                style.configure('TEntry', font=('Microsoft YaHei', self.base_font_size))
                style.configure('TCombobox', font=('Microsoft YaHei', self.base_font_size))
                
                print(f"✓ 已设置高DPI字体，基础字体大小: {self.base_font_size}")
            else:
                print("✓ 标准DPI屏幕，使用默认字体设置")
                
        except Exception as e:
            print(f"⚠ 设置高DPI字体失败: {e}")
            print("将使用默认字体设置")
        
    def _on_window_resize(self, event):
        """窗口大小变化处理 - 自动缩放字体"""
        if event.widget == self.root and not self._is_3d_active:
            # 计算缩放比例
            current_width = self.root.winfo_width()
            current_height = self.root.winfo_height()
            
            width_scale = current_width / self.default_width
            height_scale = current_height / self.default_height
            new_scale = min(width_scale, height_scale)  # 使用较小的缩放比例保持比例
            
            # 只有缩放比例变化较大时才更新字体
            if abs(new_scale - self.current_scale) > 0.1:
                self.current_scale = new_scale
                self._update_font_sizes()
                
    def _update_font_sizes(self):
        """根据缩放比例更新字体大小"""
        try:
            # 计算新的字体大小
            new_default_size = max(8, int(self.original_default_size * self.current_scale))
            new_text_size = max(8, int(self.original_text_size * self.current_scale))
            new_fixed_size = max(8, int(self.original_fixed_size * self.current_scale))
            
            # 更新字体
            self.default_font.configure(size=new_default_size)
            self.text_font.configure(size=new_text_size)
            self.fixed_font.configure(size=new_fixed_size)
            
            # 强制刷新所有组件
            self.root.update_idletasks()
        except Exception as e:
            print(f"字体更新失败: {e}")
        
    def _on_window_configure(self, event):
        """窗口配置变化事件处理（简化版）"""
        # 由于现在使用离屏渲染，不再需要复杂的窗口锁定机制
        pass
                
    def _unlock_window_from_3d(self):
        """解锁窗口（保留用于菜单调试功能）"""
        self._is_3d_active = False
        self.root.resizable(True, True)
        self.update_window_status(locked=False)
        
    def setup_gpu(self):
        """设置GPU环境"""
        print("🔧 配置GPU环境...")
        self.gpus = tf.config.list_physical_devices('GPU')
        if self.gpus:
            try:
                for gpu in self.gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.set_visible_devices(self.gpus[0], 'GPU')
                print(f"✅ 检测到 {len(self.gpus)} 个GPU设备")
            except RuntimeError as e:
                print(f"⚠️  GPU配置失败: {e}")
        else:
            print("❌ 未检测到GPU设备，将使用CPU训练")
        tf.get_logger().setLevel('ERROR')
        
    def setup_subjects_dir(self):
        """设置FreeSurfer subjects目录"""
        try:
            self.subjects_dir = str(mne.datasets.sample.data_path() / 'subjects')
            mne.set_config('SUBJECTS_DIR', self.subjects_dir)
            print(f"✅ FreeSurfer subjects目录设置完成: {self.subjects_dir}")
        except Exception as e:
            print(f"⚠️ FreeSurfer subjects目录设置失败: {e}")
            
    def create_widgets(self):
        """创建主界面组件"""
        self.create_menu()
        
        # 创建页面选择标签
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 创建三个页面
        self.create_data_page()
        self.create_training_page()
        self.create_results_page()
        
    def create_menu(self):
        """创建顶级菜单"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="退出", command=self.root.quit)
        
        # 窗口菜单
        window_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="窗口", menu=window_menu)
        window_menu.add_command(label="最大化", command=self.maximize_window)
        window_menu.add_command(label="还原默认大小", command=self.restore_default_size)
        window_menu.add_command(label="居中显示", command=self.center_window)
        window_menu.add_separator()
        window_menu.add_command(label="解锁窗口（调试用）", command=self._unlock_window_from_3d)
        window_menu.add_separator()
        window_menu.add_command(label="恢复窗口状态", command=self._restore_gui_state)
        window_menu.add_command(label="解冻GUI窗口", command=self._unfreeze_gui_window)
        window_menu.add_separator()
        window_menu.add_command(label="修复全屏模糊", command=self._fix_dpi_blur)
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="关于", command=self.show_about)
        help_menu.add_separator()
        help_menu.add_command(label="安装PyVista说明", command=self.show_pyvista_install_help)
        
    def show_about(self):
        """显示关于对话框"""
        status = "✓ PyVista已安装" if PYVISTA_AVAILABLE else "⚠ PyVista未安装"
        messagebox.showinfo("关于", f"BCI-IV-2a源定位分析系统\n基于ESINet深度学习框架\nv1.0\n\n3D后端状态: {status}")
        
    def show_pyvista_install_help(self):
        """显示PyVista安装帮助"""
        help_text = """PyVista安装说明

PyVista是可交互3D脑图显示所需的依赖包。

安装方法：
1. 使用conda（推荐）：
   conda install -c conda-forge pyvista

2. 使用pip：
   pip install pyvista

3. 如果遇到问题，可以尝试：
   pip install pyvista[all]

安装完成后请重启程序以使用完整的3D功能。

当前状态：""" + ("✓ 已安装" if PYVISTA_AVAILABLE else "⚠ 未安装")
        
        messagebox.showinfo("PyVista安装说明", help_text)
        
    def _fix_dpi_blur(self):
        """修复全屏模糊问题"""
        try:
            # 重新设置高DPI字体
            self._setup_high_dpi_fonts()
            
            # 强制刷新所有组件
            self.root.update_idletasks()
            
            # 重新创建matplotlib图形
            if hasattr(self, 'loss_fig'):
                dpi = 100 * self.current_scale
                self.loss_fig.set_dpi(dpi)
                self.loss_canvas.draw()
                
            if hasattr(self, 'brain_3d_fig'):
                dpi = 100 * self.current_scale
                self.brain_3d_fig.set_dpi(dpi)
                self.brain_3d_canvas.draw()
            
            messagebox.showinfo("成功", f"已修复全屏模糊问题\n当前DPI缩放: {self.current_scale:.2f}\n字体大小: {self.base_font_size}")
            
        except Exception as e:
            messagebox.showerror("错误", f"修复DPI模糊失败: {str(e)}")
        
    def maximize_window(self):
        """最大化窗口"""
        if not self._is_3d_active:  # 只有在非3D模式下才允许最大化
            self.root.state('zoomed')  # Windows系统最大化
            # 对于其他系统可以用: self.root.attributes('-zoomed', True)
            
    def restore_default_size(self):
        """还原默认窗口大小"""
        if not self._is_3d_active:  # 只有在非3D模式下才允许调整
            self.root.state('normal')
            self.root.geometry(f"{self.default_width}x{self.default_height}")
            self.center_window()
            
    def center_window(self):
        """将窗口居中显示"""
        if not self._is_3d_active:  # 只有在非3D模式下才允许移动
            self.root.update_idletasks()
            width = self.root.winfo_width()
            height = self.root.winfo_height()
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            
            x = (screen_width - width) // 2
            y = (screen_height - height) // 2
            
            self.root.geometry(f"{width}x{height}+{x}+{y}")
            
    def create_status_bar(self):
        """创建状态栏"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side="bottom", fill="x")
        
        self.status_label = ttk.Label(self.status_bar, text="就绪", relief="sunken", anchor="w")
        self.status_label.pack(side="left", fill="x", expand=True)
        
        self.window_status_label = ttk.Label(self.status_bar, text="窗口: 可调整", relief="sunken")
        self.window_status_label.pack(side="right")
        
    def update_window_status(self, locked=False):
        """更新窗口状态显示"""
        if locked:
            self.window_status_label.config(text="窗口: 已锁定（3D可视化中）", foreground="red")
            self.status_label.config(text="3D可视化激活中，窗口已临时锁定")
        else:
            self.window_status_label.config(text="窗口: 可调整", foreground="green")
            self.status_label.config(text="就绪")
        
    def create_data_page(self):
        """创建数据准备页面"""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="数据准备")
        
        # 左侧控制面板
        control_frame = ttk.Frame(data_frame)
        control_frame.pack(side="left", fill="y", padx=5, pady=5)
        
        # 数据导入部分
        import_group = ttk.LabelFrame(control_frame, text="数据导入")
        import_group.pack(fill="x", pady=5)
        
        ttk.Button(import_group, text="导入EEG数据", 
                  command=self.import_eeg_data).pack(pady=5)
        
        # 模拟参数设置
        sim_group = ttk.LabelFrame(control_frame, text="模拟参数设置")
        sim_group.pack(fill="x", pady=5)
        
        # 添加参数控制组件
        self.create_parameter_controls(sim_group)
        
        # 数据生成控制
        gen_group = ttk.LabelFrame(control_frame, text="数据生成")
        gen_group.pack(fill="x", pady=5)
        
        ttk.Button(gen_group, text="生成训练数据", 
                  command=self.generate_training_data).pack(pady=2)
        ttk.Button(gen_group, text="生成测试数据", 
                  command=self.generate_test_data).pack(pady=2)
        ttk.Button(gen_group, text="生成全部数据", 
                  command=self.generate_all_data).pack(pady=2)
        
        # 添加测试按钮
        ttk.Button(gen_group, text="测试3D显示", 
                  command=self.test_3d_display).pack(pady=2)
        
        # 右侧可视化面板
        viz_frame = ttk.Frame(data_frame)
        viz_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        
        # 创建可视化标签页
        self.data_viz_notebook = ttk.Notebook(viz_frame)
        self.data_viz_notebook.pack(fill="both", expand=True)
        
        # EEG信号可视化页面
        self.eeg_viz_frame = ttk.Frame(self.data_viz_notebook)
        self.data_viz_notebook.add(self.eeg_viz_frame, text="EEG信号")
        
        # 3D脑图可视化页面
        self.brain_viz_frame = ttk.Frame(self.data_viz_notebook)
        self.data_viz_notebook.add(self.brain_viz_frame, text="3D脑图")
        
    def create_parameter_controls(self, parent):
        """创建参数控制组件"""
        params = [
            ('试验持续时间(s)', 'duration_of_trial', 1.0, 10.0),
            ('目标信噪比(dB)', 'target_snr', 1, 20),
            ('源数量', 'number_of_sources', 1, 10),
            ('源范围最小值(mm)', 'extents_min', 1, 20),
            ('源范围最大值(mm)', 'extents_max', 5, 50),
            ('Beta源最小值', 'beta_source_min', 0.1, 2.0),
            ('Beta源最大值', 'beta_source_max', 1.0, 3.0),
            ('训练样本数量', 'train_samples', 100, 2000),
            ('测试样本数量', 'test_samples', 50, 500)
        ]
        
        for label, var_name, min_val, max_val in params:
            ttk.Label(parent, text=f"{label}:").pack(anchor="w")
            scale = ttk.Scale(parent, from_=min_val, to=max_val, orient="horizontal",
                            variable=self.sim_params[var_name])
            scale.pack(fill="x")
            value_label = ttk.Label(parent, textvariable=self.sim_params[var_name])
            value_label.pack(anchor="w")
        
    def create_training_page(self):
        """创建网络训练页面"""
        train_frame = ttk.Frame(self.notebook)
        self.notebook.add(train_frame, text="网络训练")
        
        # 左侧参数设置
        param_frame = ttk.Frame(train_frame)
        param_frame.pack(side="left", fill="y", padx=5, pady=5)
        
        # 训练参数设置
        train_group = ttk.LabelFrame(param_frame, text="训练参数")
        train_group.pack(fill="x", pady=5)
        
        # 添加训练参数控制
        train_params = [
            ('学习率', 'learning_rate', 0.0001, 0.01),
            ('批大小', 'batch_size', 8, 128),
            ('训练轮数', 'epochs', 100, 2000),
            ('早停耐心值', 'patience', 5, 50)
        ]
        
        for label, var_name, min_val, max_val in train_params:
            ttk.Label(train_group, text=f"{label}:").pack(anchor="w")
            scale = ttk.Scale(train_group, from_=min_val, to=max_val, orient="horizontal",
                            variable=self.train_params[var_name])
            scale.pack(fill="x")
            value_label = ttk.Label(train_group, textvariable=self.train_params[var_name])
            value_label.pack(anchor="w")
        
        # 训练控制
        control_group = ttk.LabelFrame(param_frame, text="训练控制")
        control_group.pack(fill="x", pady=5)
        
        self.train_button = ttk.Button(control_group, text="开始训练", 
                                      command=self.start_training)
        self.train_button.pack(pady=5)
        
        self.stop_button = ttk.Button(control_group, text="停止训练", 
                                     command=self.stop_training, state="disabled")
        self.stop_button.pack(pady=5)
        
        # 训练状态显示
        status_group = ttk.LabelFrame(param_frame, text="训练状态")
        status_group.pack(fill="x", pady=5)
        
        self.status_label = ttk.Label(status_group, text="未开始训练")
        self.status_label.pack(pady=5)
        
        self.progress_bar = ttk.Progressbar(status_group, mode='indeterminate')
        self.progress_bar.pack(fill="x", pady=5)
        
        # 右侧Loss曲线可视化
        loss_frame = ttk.Frame(train_frame)
        loss_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        
        ttk.Label(loss_frame, text="训练Loss曲线", font=("Arial", 14)).pack(pady=5)
        
        # 创建matplotlib图形（高DPI优化）
        dpi = 100 * self.current_scale if hasattr(self, 'current_scale') else 100
        self.loss_fig = Figure(figsize=(8, 6), dpi=dpi)
        self.loss_ax = self.loss_fig.add_subplot(111)
        self.loss_canvas = FigureCanvasTkAgg(self.loss_fig, loss_frame)
        self.loss_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # 添加matplotlib工具栏
        toolbar_frame = ttk.Frame(loss_frame)
        toolbar_frame.pack(fill="x")
        self.loss_toolbar = NavigationToolbar2Tk(self.loss_canvas, toolbar_frame)
        
    def create_results_page(self):
        """创建结果可视化页面"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="结果可视化")
        
        # 左侧评估指标
        metrics_frame = ttk.Frame(results_frame)
        metrics_frame.pack(side="left", fill="y", padx=5, pady=5)
        
        # 评估指标显示
        metrics_group = ttk.LabelFrame(metrics_frame, text="评估指标")
        metrics_group.pack(fill="x", pady=5)
        
        self.mle_label = ttk.Label(metrics_group, text="定位误差(MLE): 未计算")
        self.mle_label.pack(anchor="w", pady=2)
        
        self.mse_label = ttk.Label(metrics_group, text="均方误差(MSE): 未计算")
        self.mse_label.pack(anchor="w", pady=2)
        
        self.auc_label = ttk.Label(metrics_group, text="AUC值: 未计算")
        self.auc_label.pack(anchor="w", pady=2)
        
        # 结果控制
        control_group = ttk.LabelFrame(metrics_frame, text="结果控制")
        control_group.pack(fill="x", pady=5)
        
        ttk.Button(control_group, text="计算评估指标", 
                  command=self.calculate_metrics).pack(pady=5)
        ttk.Button(control_group, text="保存结果", 
                  command=self.save_results).pack(pady=5)
        
        # 右侧3D脑图可视化
        brain_frame = ttk.Frame(results_frame)
        brain_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        
        ttk.Label(brain_frame, text="源定位结果3D可视化", font=("Arial", 14)).pack(pady=5)
        
        # 3D可视化控制
        viz_control = ttk.Frame(brain_frame)
        viz_control.pack(fill="x", pady=5)
        
        ttk.Button(viz_control, text="显示预测结果", 
                  command=self.show_prediction_brain).pack(side="left", padx=5)
        
        # 视角选择下拉框
        ttk.Label(viz_control, text="视角:").pack(side="left", padx=(10,0))
        self.view_var = tk.StringVar(value="lateral")
        view_combo = ttk.Combobox(viz_control, textvariable=self.view_var, 
                                 values=["lateral", "medial", "dorsal", "ventral"],
                                 state="readonly", width=8)
        view_combo.pack(side="left", padx=5)
        view_combo.bind("<<ComboboxSelected>>", self.on_view_changed)
        
        ttk.Button(viz_control, text="保存3D图像", 
                  command=self.save_brain_images).pack(side="left", padx=5)
        
        # 3D可视化区域
        self.brain_viz_area = ttk.Frame(brain_frame, relief="sunken", borderwidth=2)
        self.brain_viz_area.pack(fill="both", expand=True, pady=5)
        
        # 创建嵌入式3D可视化画布
        self.create_embedded_3d_canvas()
        
    def create_embedded_3d_canvas(self):
        """创建嵌入式3D可视化画布"""
        try:
            # 清空现有内容
            for widget in self.brain_viz_area.winfo_children():
                widget.destroy()
                
            # 创建3D可视化说明标签
            info_frame = ttk.Frame(self.brain_viz_area)
            info_frame.pack(fill="x", pady=5)
            
            self.brain_info_label = ttk.Label(info_frame, 
                                            text="3D脑图将在此区域显示", 
                                            font=("Arial", 12))
            self.brain_info_label.pack(pady=10)
            
            # 创建matplotlib图形用于显示3D脑图截图（高DPI优化）
            dpi = 100 * self.current_scale if hasattr(self, 'current_scale') else 100
            self.brain_3d_fig = Figure(figsize=(8, 6), dpi=dpi)
            self.brain_3d_ax = self.brain_3d_fig.add_subplot(111)
            self.brain_3d_ax.set_title("3D脑源分布", fontsize=12 * self.current_scale)
            self.brain_3d_ax.axis('off')
            
            # 显示默认提示图像
            self.brain_3d_ax.text(0.5, 0.5, '点击"显示预测结果"查看3D脑图', 
                                 ha='center', va='center', fontsize=14, 
                                 transform=self.brain_3d_ax.transAxes)
            
            self.brain_3d_canvas = FigureCanvasTkAgg(self.brain_3d_fig, self.brain_viz_area)
            self.brain_3d_canvas.draw()
            self.brain_3d_canvas.get_tk_widget().pack(fill="both", expand=True)
            
            # 添加工具栏
            brain_toolbar_frame = ttk.Frame(self.brain_viz_area)
            brain_toolbar_frame.pack(fill="x")
            self.brain_3d_toolbar = NavigationToolbar2Tk(self.brain_3d_canvas, brain_toolbar_frame)
            
        except Exception as e:
            print(f"创建3D画布失败: {e}")
        
    def import_eeg_data(self):
        """导入EEG数据"""
        data_path = filedialog.askdirectory(title="选择BCI-IV-2a数据文件夹")
        if data_path:
            try:
                self.epochs_list = load_bci_iv_2a_data(data_path)
                if self.epochs_list:
                    self.epochs, self.epochs_plot = self.epochs_list[0]
                    messagebox.showinfo("成功", f"成功导入数据，包含 {len(self.epochs)} 个试次")
                    self.create_forward_model()
                    self.visualize_eeg_data()
                else:
                    messagebox.showerror("错误", "未能成功加载任何数据文件")
            except Exception as e:
                messagebox.showerror("错误", f"导入数据失败: {str(e)}")
                
    def generate_training_data(self):
        """生成训练数据"""
        if self.fwd is None:
            try:
                self.create_default_forward_model()
            except Exception as e:
                messagebox.showerror("错误", f"创建前向模型失败: {str(e)}")
                return
                
        try:
            settings = self.get_simulation_settings()
            ch_names = get_bci_iv_2a_channel_names()
            info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types='eeg')
            montage = mne.channels.make_standard_montage('standard_1020')
            info.set_montage(montage)
            
            n_samples = int(self.sim_params['train_samples'].get())
            self.simulation = Simulation(self.fwd, info, settings=settings, verbose=True)
            self.simulation.simulate(n_samples=n_samples)
            
            messagebox.showinfo("成功", f"训练数据生成完成，包含{n_samples}个样本")
            self.visualize_simulation_data()
            
        except Exception as e:
            messagebox.showerror("错误", f"生成训练数据失败: {str(e)}")
            
    def generate_test_data(self):
        """生成测试数据"""
        if self.fwd is None:
            messagebox.showwarning("警告", "请先生成训练数据")
            return
            
        try:
            settings = self.get_simulation_settings()
            ch_names = get_bci_iv_2a_channel_names()
            info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types='eeg')
            montage = mne.channels.make_standard_montage('standard_1020')
            info.set_montage(montage)
            
            n_samples = int(self.sim_params['test_samples'].get())
            self.simulation_test = Simulation(self.fwd, info, settings=settings, verbose=True)
            self.simulation_test.simulate(n_samples=n_samples)
            
            messagebox.showinfo("成功", f"测试数据生成完成，包含{n_samples}个样本")
            self.visualize_brain_sources()  # 更新3D脑图显示
            
        except Exception as e:
            messagebox.showerror("错误", f"生成测试数据失败: {str(e)}")
            
    def generate_all_data(self):
        """生成全部数据（训练+测试）"""
        self.generate_training_data()
        if self.simulation is not None:  # 确保训练数据生成成功
            self.generate_test_data()
            
    def test_3d_display(self):
        """测试3D显示功能"""
        try:
            print("==== 开始测试3D显示功能 ====")
            
            # 检查PyVista状态
            print(f"PyVista可用状态: {PYVISTA_AVAILABLE}")
            
            # 如果没有数据，先生成一些测试数据
            if not hasattr(self, 'simulation') or self.simulation is None:
                print("没有模拟数据，先生成测试数据...")
                self.generate_training_data()
                
            if hasattr(self, 'simulation') and self.simulation is not None:
                print("使用现有模拟数据进行3D显示测试")
                source_data = self.simulation.source_data[0]
                print(f"测试数据类型: {type(source_data)}")
                print(f"测试数据形状: {source_data.data.shape}")
                
                # 直接调用最简单的显示函数
                self._show_simple_brain_direct(source_data, "3D显示测试")
            else:
                messagebox.showerror("错误", "无法生成测试数据")
                
        except Exception as e:
            print(f"测试3D显示失败: {e}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
            messagebox.showerror("错误", f"测试3D显示失败: {str(e)}")
            
    def get_simulation_settings(self):
        """获取当前模拟设置"""
        return {
            'duration_of_trial': self.sim_params['duration_of_trial'].get(),
            'target_snr': self.sim_params['target_snr'].get(),
            'number_of_sources': self.sim_params['number_of_sources'].get(),
            'extents': (self.sim_params['extents_min'].get(), self.sim_params['extents_max'].get()),
            'beta_source': (self.sim_params['beta_source_min'].get(), self.sim_params['beta_source_max'].get()),
            'source_time_course': 'sine'
        }
        
    def create_forward_model(self):
        """为真实数据创建前向模型"""
        try:
            if self.subjects_dir is None:
                self.setup_subjects_dir()
                
            subject = 'fsaverage'
            src_fname = os.path.join(self.subjects_dir, subject, 'bem', f'{subject}-oct-6-src.fif')
            if not os.path.exists(src_fname):
                src = mne.setup_source_space(subject, spacing='oct3', 
                                           subjects_dir=self.subjects_dir,
                                           add_dist=False, verbose=True)
                mne.write_source_spaces(src_fname, src, overwrite=True)
            else:
                src = mne.read_source_spaces(src_fname, verbose=True)
                
            bem_fname = os.path.join(self.subjects_dir, subject, 'bem', f'{subject}-5120-5120-5120-bem-sol.fif')
            bem = mne.read_bem_solution(bem_fname, verbose=True)
            
            trans_fname = os.path.join(self.subjects_dir, subject, 'bem', f'{subject}-trans.fif')
            trans = mne.read_trans(trans_fname, verbose=True)
            
            self.fwd = mne.make_forward_solution(
                self.epochs.info, trans=trans, src=src, bem=bem,
                meg=False, eeg=True, mindist=5.0, verbose=True
            )
            
            self.fwd = mne.convert_forward_solution(
                self.fwd, surf_ori=True, force_fixed=True,
                use_cps=True, verbose=False
            )
            
            print(f"✅ 前向模型创建完成，包含 {self.fwd['sol']['data'].shape[1]} 个源点")
            
        except Exception as e:
            print(f"❌ 前向模型创建失败: {e}")
            raise
            
    def create_default_forward_model(self):
        """创建默认前向模型（用于模拟数据）"""
        try:
            if self.subjects_dir is None:
                self.setup_subjects_dir()
                
            subject = 'fsaverage'
            ch_names = get_bci_iv_2a_channel_names()
            info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types='eeg')
            montage = mne.channels.make_standard_montage('standard_1020')
            info.set_montage(montage)
            
            src_fname = os.path.join(self.subjects_dir, subject, 'bem', f'{subject}-oct-6-src.fif')
            if not os.path.exists(src_fname):
                src = mne.setup_source_space(subject, spacing='oct3', 
                                           subjects_dir=self.subjects_dir,
                                           add_dist=False, verbose=True)
                mne.write_source_spaces(src_fname, src, overwrite=True)
            else:
                src = mne.read_source_spaces(src_fname, verbose=True)
                
            bem_fname = os.path.join(self.subjects_dir, subject, 'bem', f'{subject}-5120-5120-5120-bem-sol.fif')
            bem = mne.read_bem_solution(bem_fname, verbose=True)
            
            trans_fname = os.path.join(self.subjects_dir, subject, 'bem', f'{subject}-trans.fif')
            trans = mne.read_trans(trans_fname, verbose=True)
            
            self.fwd = mne.make_forward_solution(
                info, trans=trans, src=src, bem=bem,
                meg=False, eeg=True, mindist=5.0, verbose=True
            )
            
            self.fwd = mne.convert_forward_solution(
                self.fwd, surf_ori=True, force_fixed=True,
                use_cps=True, verbose=False
            )
            
            print(f"✅ 默认前向模型创建完成，包含 {self.fwd['sol']['data'].shape[1]} 个源点")
            
        except Exception as e:
            print(f"❌ 默认前向模型创建失败: {e}")
            raise
            
    def visualize_eeg_data(self):
        """可视化真实EEG数据（所有22个通道）"""
        if self.epochs_plot is None:
            return
            
        try:
            for widget in self.eeg_viz_frame.winfo_children():
                widget.destroy()
                
            # 创建数据选择控制
            control_frame = ttk.Frame(self.eeg_viz_frame)
            control_frame.pack(fill="x", pady=5)
            
            ttk.Label(control_frame, text="真实EEG数据可视化（22通道）", font=("Arial", 12)).pack()
            
            fig = Figure(figsize=(12, 8), dpi=100)
            
            for i, (hand_label, hand_name) in enumerate(zip(['left_hand', 'right_hand'], ['左手', '右手'])):
                if hand_label in self.epochs_plot.event_id:
                    hand_epochs = self.epochs_plot[hand_label]
                    if len(hand_epochs) > 0:
                        ax = fig.add_subplot(2, 1, i+1)
                        evoked = hand_epochs.average()
                        
                        # 显示所有22个通道
                        times = evoked.times
                        data = evoked.data
                        
                        # 绘制所有通道，使用不同颜色
                        colors = plt.cm.tab20(np.linspace(0, 1, len(evoked.ch_names)))
                        for j, ch in enumerate(evoked.ch_names):
                            ax.plot(times, data[j] * 1e6, label=ch, color=colors[j], alpha=0.7)
                                
                        ax.set_title(f'{hand_name}运动想象EEG平均波形（22通道）')
                        ax.set_xlabel('时间 (s)')
                        ax.set_ylabel('幅值 (μV)')
                        # 将图例分两列显示
                        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=8)
                        ax.grid(True, alpha=0.3)
                            
            fig.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, self.eeg_viz_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
            toolbar_frame = ttk.Frame(self.eeg_viz_frame)
            toolbar_frame.pack(fill="x")
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            
        except Exception as e:
            print(f"EEG可视化失败: {e}")
            
    def visualize_simulation_data(self):
        """可视化模拟数据（训练集和测试集分离显示）"""
        try:
            for widget in self.eeg_viz_frame.winfo_children():
                widget.destroy()
                
            # 创建数据选择控制
            control_frame = ttk.Frame(self.eeg_viz_frame)
            control_frame.pack(fill="x", pady=5)
            
            ttk.Label(control_frame, text="模拟EEG数据可视化", font=("Arial", 12)).pack()
            
            # 数据集选择按钮
            button_frame = ttk.Frame(control_frame)
            button_frame.pack(pady=5)
            
            ttk.Button(button_frame, text="显示训练集EEG", 
                      command=lambda: self.show_simulation_eeg('train')).pack(side="left", padx=5)
            if self.simulation_test is not None:
                ttk.Button(button_frame, text="显示测试集EEG", 
                          command=lambda: self.show_simulation_eeg('test')).pack(side="left", padx=5)
            
            # 默认显示训练集
            self.show_simulation_eeg('train')
            
            # 更新3D脑图显示
            self.visualize_brain_sources()
            
        except Exception as e:
            print(f"模拟数据可视化失败: {e}")
            
    def show_simulation_eeg(self, data_type='train'):
        """显示指定数据集的EEG信号"""
        try:
            # 清除之前的图形（保留控制面板）
            for widget in self.eeg_viz_frame.winfo_children():
                if isinstance(widget, FigureCanvasTkAgg):
                    widget.get_tk_widget().destroy()
                elif hasattr(widget, 'toolbar'):
                    widget.destroy()
                    
            simulation_data = self.simulation if data_type == 'train' else self.simulation_test
            if simulation_data is None:
                return
                
            fig = Figure(figsize=(12, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            # 显示第一个样本的所有22个通道
            evoked = simulation_data.eeg_data[0].average()
            times = evoked.times
            data = evoked.data
            
            # 绘制所有通道
            colors = plt.cm.tab20(np.linspace(0, 1, len(evoked.ch_names)))
            for j, ch in enumerate(evoked.ch_names):
                ax.plot(times, data[j] * 1e6, label=ch, color=colors[j], alpha=0.7)
                
            dataset_name = "训练集" if data_type == 'train' else "测试集"
            ax.set_title(f'模拟{dataset_name}EEG信号（22通道，第一个样本）')
            ax.set_xlabel('时间 (s)')
            ax.set_ylabel('幅值 (μV)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=8)
            ax.grid(True, alpha=0.3)
                
            fig.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, self.eeg_viz_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
            toolbar_frame = ttk.Frame(self.eeg_viz_frame)
            toolbar_frame.pack(fill="x")
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            
        except Exception as e:
            print(f"EEG可视化失败: {e}")
            
    def visualize_brain_sources(self):
        """可视化3D脑源分布"""
        try:
            for widget in self.brain_viz_frame.winfo_children():
                widget.destroy()
                
            info_label = ttk.Label(self.brain_viz_frame, 
                                  text="3D脑源分布\n点击下方按钮查看不同视角和数据类型",
                                  font=("Arial", 12))
            info_label.pack(pady=10)
            
            # 数据选择按钮
            data_frame = ttk.Frame(self.brain_viz_frame)
            data_frame.pack(pady=5)
            
            ttk.Button(data_frame, text="显示训练数据源分布",
                      command=lambda: self.show_data_brain('train')).pack(side="left", padx=5)
            if self.simulation_test is not None:
                ttk.Button(data_frame, text="显示测试数据源分布",
                          command=lambda: self.show_data_brain('test')).pack(side="left", padx=5)
            
            # 视角选择按钮
            view_frame = ttk.Frame(self.brain_viz_frame)
            view_frame.pack(pady=5)
            
            views = [('侧视图', 'lateral'), ('背视图', 'dorsal'), ('腹视图', 'ventral'), ('内侧视图', 'medial')]
            
            for view_name, view_code in views:
                ttk.Button(view_frame, text=view_name,
                          command=lambda v=view_code: self.show_brain_view(v)).pack(side="left", padx=2)
                          
        except Exception as e:
            print(f"3D脑源可视化失败: {e}")
            
    def show_data_brain(self, data_type='train'):
        """显示数据源分布的3D脑图（可交互，精确控制GUI窗口）"""
        if data_type == 'train' and self.simulation is None:
            messagebox.showwarning("警告", "请先生成训练数据")
            return
        elif data_type == 'test' and self.simulation_test is None:
            messagebox.showwarning("警告", "请先生成测试数据")
            return
            
        try:
            source_data = self.simulation.source_data[0] if data_type == 'train' else self.simulation_test.source_data[0]
            
            # 直接使用简单可靠的3D显示
            print(f"显示3D脑图: {data_type}数据真实源分布")
            self._show_simple_brain_direct(source_data, f'{data_type}数据真实源分布')
            
        except Exception as e:
            messagebox.showerror("错误", f"显示3D脑图失败: {str(e)}")
            
    def _position_3d_window(self, brain):
        """设置3D窗口位置，并临时锁定主GUI窗口"""
        try:
            # 保存当前主窗口状态
            current_geometry = self.root.geometry()
            current_state = self.root.state()
            
            # 临时禁用主窗口调整
            self.root.resizable(False, False)
            
            # 获取主窗口位置和大小
            self.root.update_idletasks()
            main_x = self.root.winfo_x()
            main_y = self.root.winfo_y()
            main_width = self.root.winfo_width()
            
            # 计算3D窗口位置（主窗口右侧）
            x = main_x + main_width + 20
            y = main_y + 50
            
            # 设置3D窗口位置
            if hasattr(brain, '_renderer') and hasattr(brain._renderer, 'window'):
                brain._renderer.window.SetPosition(x, y)
                # 设置窗口标题
                brain._renderer.window.SetWindowName(f"3D脑图 - MNE")
                
            # 保存状态用于后续恢复
            self._saved_gui_state = {
                'geometry': current_geometry,
                'state': current_state
            }
                
        except Exception as e:
            print(f"设置3D窗口位置失败: {e}")
            
    def _setup_3d_window_cleanup(self, brain):
        """设置3D窗口清理机制"""
        try:
            # 延时恢复主窗口状态（3秒后自动恢复）
            self.root.after(3000, self._restore_gui_state)
            
            # 尝试设置3D窗口关闭回调
            if hasattr(brain, '_renderer') and hasattr(brain._renderer, 'window'):
                try:
                    # 尝试绑定窗口关闭事件
                    def on_3d_close():
                        self._restore_gui_state()
                    # 注意：这个方法可能不适用于所有PyVista版本
                    brain._renderer.window.AddObserver('ExitEvent', lambda obj, event: on_3d_close())
                except:
                    # 如果失败，依赖延时恢复
                    pass
                    
        except Exception as e:
            print(f"设置3D窗口清理失败: {e}")
            
    def _restore_gui_state(self):
        """恢复主GUI窗口状态"""
        try:
            if hasattr(self, '_saved_gui_state'):
                # 恢复窗口可调整性
                self.root.resizable(True, True)
                # 恢复几何状态
                if self._saved_gui_state.get('geometry'):
                    self.root.geometry(self._saved_gui_state['geometry'])
                # 清理保存的状态
                delattr(self, '_saved_gui_state')
                print("主GUI窗口状态已恢复")
        except Exception as e:
            print(f"恢复GUI状态失败: {e}")
            # 确保窗口至少能调整大小
            self.root.resizable(True, True)
            
    def _show_brain_with_matplotlib(self, source_data, title, view='lateral'):
        """使用matplotlib显示3D脑图，完全避免PyVista窗口影响"""
        try:
            import matplotlib.pyplot as plt
            import threading
            import time
            
            # 在单独线程中执行3D渲染，完全隔离GUI
            def render_in_thread():
                try:
                    # 创建临时的Brain对象用于快速截图
                    temp_plot_params = dict(
                        surface='inflated', 
                        cortex="low_contrast", 
                        hemi='both', 
                        verbose=0, 
                        background='white', 
                        foreground='black', 
                        size=(1000, 800),
                        view=view if view != 'lateral' else None
                    )
                    
                    # 快速生成和截图
                    brain = source_data.plot(**temp_plot_params)
                    if view and view != 'lateral':
                        brain.show_view(view)
                    brain.add_text(0.1, 0.9, title, 'title', font_size=16)
                    
                    # 立即截图并关闭
                    screenshot = brain.screenshot(transparent_bg=False, return_img=True)
                    brain.close()
                    
                    # 在主线程中显示matplotlib窗口
                    def show_matplotlib():
                        try:
                            fig, ax = plt.subplots(figsize=(14, 10))
                            ax.imshow(screenshot)
                            ax.set_title(title, fontsize=18, pad=25)
                            ax.axis('off')
                            plt.tight_layout()
                            
                            # 设置窗口位置
                            mngr = fig.canvas.manager
                            try:
                                main_x = self.root.winfo_x()
                                main_width = self.root.winfo_width()
                                if hasattr(mngr, 'window'):
                                    if hasattr(mngr.window, 'wm_geometry'):
                                        mngr.window.wm_geometry(f"+{main_x + main_width + 30}+{80}")
                                    elif hasattr(mngr.window, 'move'):
                                        mngr.window.move(main_x + main_width + 30, 80)
                            except:
                                pass
                            
                            plt.show()
                        except Exception as e:
                            print(f"显示matplotlib窗口失败: {e}")
                    
                    # 在主线程中执行matplotlib显示
                    self.root.after(100, show_matplotlib)
                    
                except Exception as e:
                    print(f"后台渲染失败: {e}")
            
            # 启动渲染线程
            render_thread = threading.Thread(target=render_in_thread, daemon=True)
            render_thread.start()
            
        except Exception as e:
            print(f"启动3D渲染失败: {e}")
            # 备用方案：直接使用matplotlib绘制源数据信息
            self._show_fallback_brain_info(source_data, title)
            
    def _show_fallback_brain_info(self, source_data, title):
        """备用方案：显示源数据的统计信息"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'{title} - 源数据分析', fontsize=16)
            
            # 获取源数据的数值
            data = source_data.data
            
            # 1. 时间序列图
            axes[0,0].plot(source_data.times, np.mean(data, axis=0))
            axes[0,0].set_title('平均激活时间序列')
            axes[0,0].set_xlabel('时间 (s)')
            axes[0,0].set_ylabel('激活强度')
            
            # 2. 激活强度分布
            axes[0,1].hist(data.flatten(), bins=50, alpha=0.7)
            axes[0,1].set_title('激活强度分布')
            axes[0,1].set_xlabel('激活强度')
            axes[0,1].set_ylabel('频次')
            
            # 3. 峰值激活图
            peak_data = np.max(np.abs(data), axis=1)
            axes[1,0].plot(peak_data)
            axes[1,0].set_title('各顶点峰值激活')
            axes[1,0].set_xlabel('顶点索引')
            axes[1,0].set_ylabel('峰值激活强度')
            
            # 4. 统计信息
            stats_text = f"""数据统计信息:
形状: {data.shape}
时间范围: {source_data.times[0]:.3f} - {source_data.times[-1]:.3f} s
最大激活: {np.max(data):.2e}
最小激活: {np.min(data):.2e}
平均激活: {np.mean(data):.2e}
标准差: {np.std(data):.2e}"""
            
            axes[1,1].text(0.1, 0.5, stats_text, transform=axes[1,1].transAxes, 
                          fontsize=10, verticalalignment='center')
            axes[1,1].set_title('统计信息')
            axes[1,1].axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"显示备用源数据信息失败: {e}")
            
    def _show_interactive_brain_protected(self, source_data, title, view='lateral'):
        """显示可交互3D脑图，同时保护GUI窗口不受影响"""
        try:
            # 第一步：保存GUI窗口状态
            self._freeze_gui_window()
            
            # 第二步：创建可交互3D脑图
            plot_params = dict(
                surface='inflated', 
                cortex="low_contrast", 
                hemi='both', 
                verbose=0, 
                background='white', 
                foreground='black', 
                size=(900, 700),
                view=view if view != 'lateral' else None
            )
            
            print(f"正在创建3D脑图: {title}")
            brain = source_data.plot(**plot_params)
            
            if view and view != 'lateral':
                brain.show_view(view)
            brain.add_text(0.1, 0.9, title, 'title', font_size=16)
            
            # 确认3D窗口已创建
            if hasattr(brain, '_renderer') and hasattr(brain._renderer, 'window'):
                print("✓ 3D脑图窗口已成功创建")
            else:
                print("✗ 3D脑图窗口创建失败")
            
            # 第三步：3D窗口创建后，立即设置位置和启动保护
            self.root.after(500, lambda: self._setup_3d_protection(brain))
            
        except Exception as e:
            print(f"显示保护性交互3D脑图失败: {e}")
            # 确保GUI解冻
            self._unfreeze_gui_window()
            
    def _setup_3d_protection(self, brain):
        """在3D窗口创建后设置保护机制"""
        try:
            # 现在启用GUI窗口保护
            self.root.resizable(False, False)
            
            # 精确设置3D窗口位置
            self._position_brain_window_precisely(brain)
            
            # 启动监控
            self._start_geometry_monitoring()
            
            # 设置窗口关闭监听和自动恢复
            self._setup_brain_window_monitoring(brain)
            
            print("3D脑图保护机制已启动")
            
        except Exception as e:
            print(f"设置3D保护机制失败: {e}")
            self._unfreeze_gui_window()
            
    def _freeze_gui_window(self):
        """冻结GUI窗口，防止被3D窗口影响"""
        try:
            # 保存当前窗口状态（避免闪烁）
            self.root.update_idletasks()  # 确保获取准确的窗口信息
            self._saved_gui_geometry = self.root.geometry()
            self._saved_gui_state = self.root.state()
            
            # 记录当前位置和大小
            self._saved_x = self.root.winfo_x()
            self._saved_y = self.root.winfo_y()
            self._saved_width = self.root.winfo_width()
            self._saved_height = self.root.winfo_height()
            
            # 温和地固定窗口（减少闪烁）
            # 不立即禁用resizable，而是在3D窗口创建后
            
            print(f"GUI窗口状态已保存: {self._saved_gui_geometry}")
            
        except Exception as e:
            print(f"保存GUI窗口状态失败: {e}")
            
    def _start_geometry_monitoring(self):
        """开始监控GUI窗口几何变化"""
        try:
            # 只有在有保存的几何信息时才监控
            if hasattr(self, '_saved_gui_geometry'):
                current_geometry = self.root.geometry()
                if current_geometry != self._saved_gui_geometry:
                    # 检测到变化，立即恢复
                    print(f"检测到窗口变化: {current_geometry} -> {self._saved_gui_geometry}")
                    self.root.geometry(self._saved_gui_geometry)
                    
                # 每200ms检查一次（降低频率减少闪烁）
                self._geometry_check_job = self.root.after(200, self._start_geometry_monitoring)
            
        except Exception as e:
            print(f"几何监控失败: {e}")
            
    def _position_brain_window_precisely(self, brain):
        """精确定位3D脑图窗口"""
        try:
            # 计算3D窗口的精确位置（主GUI右侧）
            x = self._saved_x + self._saved_width + 30
            y = self._saved_y + 50
            
            # 设置3D窗口位置
            if hasattr(brain, '_renderer') and hasattr(brain._renderer, 'window'):
                brain._renderer.window.SetPosition(x, y)
                brain._renderer.window.SetWindowName("可交互3D脑图")
                print(f"3D窗口位置设置为: ({x}, {y})")
                
        except Exception as e:
            print(f"设置3D窗口位置失败: {e}")
            
    def _setup_brain_window_monitoring(self, brain):
        """设置3D窗口监控和自动恢复机制"""
        try:
            # 缩短自动恢复时间（5秒后自动解冻）
            self.root.after(5000, self._unfreeze_gui_window)
            
            # 尝试绑定3D窗口关闭事件
            if hasattr(brain, '_renderer') and hasattr(brain._renderer, 'window'):
                try:
                    # PyVista窗口关闭回调
                    def on_brain_close():
                        print("3D窗口已关闭")
                        self.root.after(100, self._unfreeze_gui_window)  # 更快的恢复
                        
                    # 设置关闭回调（不同版本的PyVista可能有不同的API）
                    brain._renderer.window.AddObserver('ExitEvent', lambda obj, event: on_brain_close())
                except:
                    print("无法绑定3D窗口关闭事件，使用延时恢复")
                    
        except Exception as e:
            print(f"设置3D窗口监控失败: {e}")
            
    def _unfreeze_gui_window(self):
        """解冻GUI窗口，恢复正常状态"""
        try:
            # 停止几何监控
            if hasattr(self, '_geometry_check_job') and self._geometry_check_job:
                self.root.after_cancel(self._geometry_check_job)
                self._geometry_check_job = None
                
            # 恢复窗口可调整性
            self.root.resizable(True, True)
            
            # 确保窗口几何正确
            if hasattr(self, '_saved_gui_geometry'):
                self.root.geometry(self._saved_gui_geometry)
                
            # 清理保存的状态
            if hasattr(self, '_saved_gui_geometry'):
                delattr(self, '_saved_gui_geometry')
            if hasattr(self, '_saved_gui_state'):
                delattr(self, '_saved_gui_state')
                
            print("GUI窗口已解冻，恢复正常状态")
            
        except Exception as e:
            print(f"解冻GUI窗口失败: {e}")
            # 确保至少能调整大小
            self.root.resizable(True, True)
            
    def _show_simple_brain(self, source_data, title, view='lateral'):
        """简单的3D脑图显示（备用方案）"""
        try:
            print(f"使用简单方案显示3D脑图: {title}")
            
            # 使用基本参数创建3D脑图
            plot_params = dict(
                surface='inflated', 
                cortex="low_contrast", 
                hemi='both', 
                verbose=0, 
                background='white', 
                foreground='black', 
                size=(800, 600),
                view=view if view != 'lateral' else None
            )
            
            brain = source_data.plot(**plot_params)
            if view and view != 'lateral':
                brain.show_view(view)
            brain.add_text(0.1, 0.9, title, 'title', font_size=14)
            
            # 简单设置窗口位置
            if hasattr(brain, '_renderer') and hasattr(brain._renderer, 'window'):
                try:
                    self.root.update_idletasks()
                    main_x = self.root.winfo_x()
                    main_width = self.root.winfo_width()
                    x = main_x + main_width + 20
                    y = 100
                    brain._renderer.window.SetPosition(x, y)
                    print(f"✓ 简单3D脑图显示成功，位置: ({x}, {y})")
                except:
                    print("设置3D窗口位置失败，但3D脑图已显示")
            
        except Exception as e:
            print(f"简单3D显示也失败: {e}")
            # 最后的备用方案
            messagebox.showinfo("提示", f"3D脑图显示遇到问题，请尝试：\n1. 关闭其他3D窗口\n2. 重启程序\n3. 使用菜单中的恢复功能")
            
    def _show_matplotlib_brain(self, source_data, title, view='lateral'):
        """使用matplotlib后端显示3D脑图"""
        try:
            print(f"使用matplotlib显示3D脑图: {title}")
            
            # 使用matplotlib后端的参数
            plot_params = dict(
                surface='inflated', 
                cortex="low_contrast", 
                hemi='both', 
                verbose=0, 
                background='white', 
                foreground='black',
                view=view if view != 'lateral' else None,
                size=(800, 600)
            )
            
            # 使用matplotlib后端创建3D图
            brain = source_data.plot(**plot_params)
            if view and view != 'lateral':
                brain.show_view(view)
            brain.add_text(0.1, 0.9, title, 'title', font_size=14)
            
            print(f"✓ matplotlib 3D脑图显示成功: {title}")
            
        except Exception as e:
            print(f"matplotlib 3D显示失败: {e}")
            # 使用统计信息作为最终备用方案
            self._show_fallback_brain_info(source_data, title)
            
    def _show_simple_brain_direct(self, source_data, title, view='lateral'):
        """最简单直接的3D脑图显示（去除所有复杂逻辑）"""
        try:
            print(f"==== 3D脑图显示调试信息 ====")
            print(f"标题: {title}")
            print(f"视角: {view}")
            print(f"source_data类型: {type(source_data)}")
            print(f"source_data形状: {source_data.data.shape if hasattr(source_data, 'data') else 'No data attr'}")
            print(f"PyVista可用: {PYVISTA_AVAILABLE}")
            
            # 检查数据是否有效
            if source_data is None:
                raise ValueError("source_data为None")
            
            if not hasattr(source_data, 'plot'):
                raise ValueError("source_data没有plot方法")
            
            print("开始调用source_data.plot()...")
            
            # 使用最基本的MNE plot参数（完全复制原始代码的方式）
            brain = source_data.plot(
                surface='inflated', 
                hemi='both', 
                size=(800, 600),
                background='white',
                foreground='black',
                cortex='low_contrast',
                verbose=False
            )
            
            print("✓ source_data.plot() 调用成功")
            
            # 检查brain对象
            print(f"brain对象类型: {type(brain)}")
            print(f"brain对象属性: {dir(brain)[:10]}...")  # 只显示前10个属性
            
            # 如果指定了视角，设置视角
            if view and view != 'lateral':
                try:
                    brain.show_view(view)
                    print(f"✓ 视角设置成功: {view}")
                except Exception as ve:
                    print(f"⚠ 设置视角失败: {view}, 错误: {ve}")
            
            # 添加标题
            try:
                brain.add_text(0.1, 0.9, title, 'title', font_size=14)
                print("✓ 标题添加成功")
            except Exception as te:
                print(f"⚠ 添加标题失败: {te}")
            
            print(f"✓ 3D脑图创建完成: {title}")
            print("==== 3D脑图显示完成 ====\n")
            
        except Exception as e:
            print(f"✗ 3D脑图创建失败: {e}")
            print(f"错误类型: {type(e)}")
            import traceback
            print(f"详细错误信息:\n{traceback.format_exc()}")
            print("==== 3D脑图显示失败 ====\n")
            
            # 如果3D显示失败，显示统计信息
            try:
                print("尝试显示备用统计信息...")
                self._show_fallback_brain_info(source_data, title)
            except Exception as e2:
                print(f"备用方案也失败: {e2}")
                messagebox.showerror("错误", f"3D显示和备用方案都失败了\n原始错误: {str(e)}\n备用错误: {str(e2)}")
            
    def _show_data_brain_window(self, screenshot, title):
        """在新的matplotlib窗口中显示3D脑图截图"""
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(screenshot)
            ax.set_title(title, fontsize=14)
            ax.axis('off')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"显示3D截图窗口失败: {e}")
            
    def show_brain_view(self, view='lateral'):
        """显示特定视角的3D脑图（可交互，精确控制GUI窗口）"""
        if self.simulation is None:
            messagebox.showwarning("警告", "请先生成训练数据")
            return
            
        try:
            source_data = self.simulation.source_data[0]
            
            # 直接使用简单可靠的3D显示
            print(f"显示3D脑图: 训练数据源分布 - {view}视角")
            self._show_simple_brain_direct(source_data, f'训练数据源分布 - {view}视角', view=view)
            
        except Exception as e:
            messagebox.showerror("错误", f"显示3D脑图失败: {str(e)}")
            
    def start_training(self):
        """开始训练"""
        if self.simulation is None:
            messagebox.showwarning("警告", "请先生成训练数据")
            return
            
        self.train_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.progress_bar.start()
        self.status_label.config(text="正在训练...")
        
        self.training_thread = threading.Thread(target=self.train_network)
        self.training_thread.daemon = True
        self.training_thread.start()
        
    def train_network(self):
        """在后台线程中训练网络"""
        try:
            model_type = 'pinn'
            self.net = Net(self.fwd, verbose=2, model_type=model_type)
            
            learning_rate = self.train_params['learning_rate'].get()
            batch_size = int(self.train_params['batch_size'].get())
            epochs = int(self.train_params['epochs'].get())
            patience = int(self.train_params['patience'].get())
            
            with tf.device('/GPU:0' if self.gpus else '/CPU:0'):
                self.net, self.history = self.net.fit(
                    self.simulation, 
                    patience=patience,
                    epochs=epochs, 
                    batch_size=batch_size, 
                    learning_rate=learning_rate, 
                    return_history=True
                )
                
            self.root.after(0, self.training_completed)
            
        except Exception as e:
            self.root.after(0, lambda: self.training_failed(str(e)))
            
    def training_completed(self):
        """训练完成后的处理"""
        self.train_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.progress_bar.stop()
        self.status_label.config(text="训练完成")
        
        self.plot_loss_curve()
        messagebox.showinfo("成功", "网络训练完成！")
        
    def training_failed(self, error_msg):
        """训练失败后的处理"""
        self.train_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.progress_bar.stop()
        self.status_label.config(text="训练失败")
        
        messagebox.showerror("错误", f"训练失败: {error_msg}")
        
    def stop_training(self):
        """停止训练"""
        self.train_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.progress_bar.stop()
        self.status_label.config(text="训练已停止")
        
    def plot_loss_curve(self):
        """绘制Loss曲线"""
        if self.history is None:
            return
            
        try:
            self.loss_ax.clear()
            
            history_dict = self.history.history if hasattr(self.history, 'history') else self.history
            train_loss = history_dict.get('loss', [])
            val_loss = history_dict.get('val_loss', [])
            
            epochs_range = range(1, len(train_loss) + 1)
            
            self.loss_ax.plot(epochs_range, train_loss, 'b-', label='训练损失', linewidth=2)
            if val_loss:
                self.loss_ax.plot(epochs_range, val_loss, 'r-', label='验证损失', linewidth=2)
                
            self.loss_ax.set_xlabel('Epoch')
            self.loss_ax.set_ylabel('Loss')
            self.loss_ax.set_title('训练Loss曲线')
            self.loss_ax.legend()
            self.loss_ax.grid(True, alpha=0.3)
            
            self.loss_canvas.draw()
            
        except Exception as e:
            print(f"绘制Loss曲线失败: {e}")
            
    def plot_test_loss(self):
        """绘制测试Loss"""
        if 'test_loss' not in self.results or self.results['test_loss'] is None:
            return
            
        try:
            # 在当前Loss图上添加测试Loss线
            test_loss = self.results['test_loss']
            history_dict = self.history.history if hasattr(self.history, 'history') else self.history
            train_loss = history_dict.get('loss', [])
            
            if train_loss:
                # 在最后一个epoch添加测试loss点
                self.loss_ax.axhline(y=test_loss, color='g', linestyle='--', 
                                   label=f'测试损失: {test_loss:.4f}', linewidth=2)
                self.loss_ax.legend()
                self.loss_canvas.draw()
                
        except Exception as e:
            print(f"绘制测试Loss失败: {e}")
            
    def calculate_metrics(self):
        """计算评估指标"""
        if self.net is None:
            messagebox.showwarning("警告", "请先训练网络")
            return
            
        if self.simulation_test is None:
            messagebox.showwarning("警告", "请先生成测试数据")
            return
            
        try:
            # 在测试集上进行预测
            self.source_hat = self.net.predict(self.simulation_test)
            
            # 计算测试集loss - 使用简化方法避免维度问题
            try:
                print("计算测试集loss...")
                # 使用手动计算loss的方法，避免model.evaluate的复杂性
                test_predictions = self.net.predict(self.simulation_test)
                
                # 计算预测误差
                total_mse = 0.0
                n_samples = len(test_predictions)
                
                for i, (pred, true) in enumerate(zip(test_predictions, self.simulation_test.source_data)):
                    pred_data = pred.data[:, 0]  # 取第一个时间点
                    true_data = true.data[:, 0]  # 取第一个时间点
                    mse = np.mean((pred_data - true_data) ** 2)
                    total_mse += mse
                
                test_loss = total_mse / n_samples
                print(f"测试loss计算成功: {test_loss:.6f}")
                
            except Exception as eval_error:
                print(f"计算测试loss失败: {eval_error}")
                print(f"错误类型: {type(eval_error)}")
                import traceback
                traceback.print_exc()
                print("使用默认值...")
                test_loss = 0.0  # 使用默认值
            
            _, _, pos, _ = util.unpack_fwd(self.fwd)
            
            mle_values = []
            mse_values = []
            auc_close_values = []
            auc_far_values = []
            
            for i in range(len(self.source_hat)):
                y_true = self.simulation_test.source_data[i].data[:, 0]
                y_est = self.source_hat[i].data[:, 0]
                
                mle = eval_mean_localization_error(y_true, y_est, pos)
                mse = eval_mse(y_true, y_est)
                auc_close, auc_far = eval_auc(y_true, y_est, pos)
                
                mle_values.append(mle)
                mse_values.append(mse)
                auc_close_values.append(auc_close)
                auc_far_values.append(auc_far)
                
            self.results['mle'] = np.nanmean(mle_values)
            self.results['mse'] = np.nanmean(mse_values)
            self.results['auc'] = np.nanmean(auc_close_values + auc_far_values)
            self.results['test_loss'] = test_loss
            
            self.mle_label.config(text=f"定位误差(MLE): {self.results['mle']:.2f} mm")
            self.mse_label.config(text=f"均方误差(MSE): {self.results['mse']:.2e}")
            self.auc_label.config(text=f"AUC值: {self.results['auc']:.4f}")
            
            # 绘制测试Loss
            self.plot_test_loss()
            
            messagebox.showinfo("成功", "评估指标计算完成")
            
        except Exception as e:
            messagebox.showerror("错误", f"计算评估指标失败: {str(e)}")
            
    def show_prediction_brain(self):
        """显示预测结果的3D脑图（嵌入式）"""
        if self.source_hat is None:
            messagebox.showwarning("警告", "请先计算评估指标")
            return
            
        # 使用当前选择的视角
        current_view = self.view_var.get()
        self.show_prediction_brain_with_view(current_view)
            
    def _display_brain_screenshot(self, screenshot, title):
        """在GUI内部显示3D脑图截图"""
        try:
            # 清空现有内容
            self.brain_3d_ax.clear()
            
            # 显示截图
            self.brain_3d_ax.imshow(screenshot)
            self.brain_3d_ax.set_title(title, fontsize=14)
            self.brain_3d_ax.axis('off')
            
            # 更新画布
            self.brain_3d_canvas.draw()
            
            # 更新信息标签
            self.brain_info_label.config(text=f"当前显示: {title}")
            
        except Exception as e:
            print(f"显示3D截图失败: {e}")
            
    def on_view_changed(self, event=None):
        """视角切换时重新生成3D脑图"""
        if self.source_hat is not None:
            self.show_prediction_brain_with_view(self.view_var.get())
            
    def show_prediction_brain_with_view(self, view="lateral"):
        """显示指定视角的预测结果3D脑图"""
        if self.source_hat is None:
            return
            
        try:
            self.brain_info_label.config(text=f"正在生成3D脑图 ({view}视角)...")
            self.root.update()
            
            if PYVISTA_AVAILABLE:
                # 使用PyVista后台绘图生成3D脑图截图
                plot_params = dict(surface='inflated', cortex="low_contrast", 
                                 hemi='both', verbose=0, background='white', 
                                 foreground='black', size=(800, 600), 
                                 view=view)
                
                # 生成3D脑图
                brain = self.source_hat[0].plot(**plot_params)
                brain.add_text(0.1, 0.9, f'预测源分布结果 - {view}视角', 'title', font_size=14)
                
                # 截图
                screenshot = brain.screenshot(transparent_bg=False, return_img=True)
                brain.close()
                
                # 在GUI内部显示截图
                self._display_brain_screenshot(screenshot, f"预测源分布结果 - {view}视角")
            else:
                # 使用matplotlib后端
                try:
                    plot_params = dict(surface='inflated', cortex="low_contrast", 
                                     hemi='both', verbose=0, background='white', 
                                     foreground='black', size=(800, 600), 
                                     view=view)
                    
                    brain = self.source_hat[0].plot(**plot_params)
                    brain.add_text(0.1, 0.9, f'预测源分布结果 - {view}视角', 'title', font_size=14)
                    
                    # matplotlib后端直接显示，无需截图
                    self.brain_info_label.config(text=f"3D脑图已显示 ({view}视角)")
                    
                except Exception as e_mpl:
                    print(f"matplotlib 3D显示失败: {e_mpl}")
                    # 显示错误信息
                    self.brain_info_label.config(text="3D显示不可用，请安装PyVista")
            
        except Exception as e:
            print(f"显示{view}视角3D脑图失败: {e}")
            self.brain_info_label.config(text="3D脑图显示失败")
            
    def save_brain_images(self):
        """保存3D脑图图像"""
        if self.source_hat is None:
            messagebox.showwarning("警告", "请先计算评估指标")
            return
            
        save_dir = filedialog.askdirectory(title="选择保存目录")
        if save_dir:
            try:
                plot_params = dict(surface='inflated', cortex="low_contrast", 
                                 hemi='both', verbose=0, background='white', 
                                 foreground='black', size=(800, 600))
                                 
                brain_pred = self.source_hat[0].plot(**plot_params)
                img_pred = brain_pred.screenshot()
                brain_pred.close()
                
                pred_path = os.path.join(save_dir, "prediction_brain.png")
                plt.imsave(pred_path, img_pred)
                
                messagebox.showinfo("成功", f"预测结果3D脑图已保存到:\n{pred_path}")
                
            except Exception as e:
                messagebox.showerror("错误", f"保存3D脑图失败: {str(e)}")
                
    def save_results(self):
        """保存结果"""
        if all(v is None for v in self.results.values()):
            messagebox.showwarning("警告", "没有可保存的结果")
            return
            
        filename = filedialog.asksaveasfilename(
            title="保存结果",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("BCI-IV-2a源定位分析结果\n")
                    f.write("=" * 30 + "\n\n")
                    
                    f.write("模拟参数:\n")
                    f.write(f"试验持续时间: {self.sim_params['duration_of_trial'].get():.1f} s\n")
                    f.write(f"目标信噪比: {self.sim_params['target_snr'].get()} dB\n")
                    f.write(f"源数量: {self.sim_params['number_of_sources'].get()}\n")
                    f.write(f"源范围: {self.sim_params['extents_min'].get()}-{self.sim_params['extents_max'].get()} mm\n")
                    f.write(f"Beta参数: {self.sim_params['beta_source_min'].get():.1f}-{self.sim_params['beta_source_max'].get():.1f}\n\n")
                    
                    f.write("训练参数:\n")
                    f.write(f"学习率: {self.train_params['learning_rate'].get():.4f}\n")
                    f.write(f"批大小: {self.train_params['batch_size'].get()}\n")
                    f.write(f"训练轮数: {self.train_params['epochs'].get()}\n")
                    f.write(f"早停耐心值: {self.train_params['patience'].get()}\n\n")
                    
                    f.write("评估结果:\n")
                    if self.results['mle'] is not None:
                        f.write(f"平均定位误差(MLE): {self.results['mle']:.2f} mm\n")
                    if self.results['mse'] is not None:
                        f.write(f"均方误差(MSE): {self.results['mse']:.2e}\n")
                    if self.results['auc'] is not None:
                        f.write(f"AUC值: {self.results['auc']:.4f}\n")
                        
                messagebox.showinfo("成功", f"结果已保存到: {filename}")
                
            except Exception as e:
                messagebox.showerror("错误", f"保存结果失败: {str(e)}")

def run_gui():
    """运行GUI应用"""
    root = tk.Tk()
    app = BCIIV2aGUI(root)
    
    def on_closing():
        if messagebox.askokcancel("退出", "确定要退出吗？"):
            plt.close('all')
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    run_gui()
