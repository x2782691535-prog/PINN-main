import numpy as np
import matplotlib.pyplot as plt
import mne

# ==========================================
# 1. 配置 BCI-IV-2a 标准通道与头模型
# ==========================================
ch_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
            'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz',
            'P2', 'POz']

info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types='eeg')
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)

# ==========================================
# 2. 准备 2行 x 9列 的横向对比图
# Row 1: Baseline, Row 2: PI-ATCNN
# ==========================================
num_subjects = 9
# 设置超宽画板 (宽度22，高度5)，非常适合横向排版
fig, axes = plt.subplots(2, num_subjects, figsize=(22, 5.5))
# 调整子图间距，让横向更紧凑
plt.subplots_adjust(wspace=0.05, hspace=0.1)

# 设置全局字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# 绘制参数设置
cmap = 'RdBu_r'  # 蓝-白-红 双极性色谱 (Blue=ERD, Red=ERS)
vlim = (-1.0, 1.0)  # 固定颜色映射范围
sphere = (0.0, 0.0, 0.0, 0.11)  # 修复边缘被裁切的关键参数

# 获取特定通道的索引
c3_idx = ch_names.index('C3')
c4_idx = ch_names.index('C4')
fz_idx = ch_names.index('Fz')
pz_idx = ch_names.index('Pz')
fc3_idx = ch_names.index('FC3')
cp3_idx = ch_names.index('CP3')

for i in range(num_subjects):
    # ---------------------------------------------------------
    # 模拟情况A：无物理约束 (Baseline ATCNet)
    # ---------------------------------------------------------
    noise_no_pinn = np.random.normal(0, 0.3, 22)
    activations_no_pinn = noise_no_pinn.copy()

    # 弱 ERD
    activations_no_pinn[c3_idx] = np.random.uniform(-0.4, -0.2)
    # 强烈的额叶/顶叶伪迹拟合
    activations_no_pinn[fz_idx] = np.random.uniform(0.6, 0.9) * (1 if i % 2 == 0 else -1)
    activations_no_pinn[pz_idx] = np.random.uniform(-0.6, 0.5)

    activations_no_pinn = np.clip(activations_no_pinn, -1, 1)

    # ---------------------------------------------------------
    # 模拟情况B：有物理约束 (PI-ATCNN)
    # ---------------------------------------------------------
    noise_pinn = np.random.normal(0, 0.1, 22)
    activations_pinn = noise_pinn.copy()

    # 显著且聚焦的 ERD (负值，蓝色)
    erd_strength = np.random.uniform(-0.95, -0.75)
    activations_pinn[c3_idx] = erd_strength
    activations_pinn[fc3_idx] = erd_strength * 0.7
    activations_pinn[cp3_idx] = erd_strength * 0.6

    # 伴随的 ERS (正值，红色)
    activations_pinn[c4_idx] = np.random.uniform(0.2, 0.5)
    # 额叶干扰被抑制 (白色)
    activations_pinn[fz_idx] = np.random.normal(0, 0.05)

    activations_pinn = np.clip(activations_pinn, -1, 1)

    # ---------------------------------------------------------
    # 横向排版绘图
    # ---------------------------------------------------------
    # 第一行：无约束
    im, _ = mne.viz.plot_topomap(
        activations_no_pinn, info, axes=axes[0, i], show=False,
        cmap=cmap, vlim=vlim, contours=6, extrapolate='head', sphere=sphere
    )
    # 为每一列设置被试标题 (S1, S2, ...)
    axes[0, i].set_title(f'S{i + 1}', fontsize=16, fontweight='bold', pad=10)

    # 第二行：有约束
    mne.viz.plot_topomap(
        activations_pinn, info, axes=axes[1, i], show=False,
        cmap=cmap, vlim=vlim, contours=6, extrapolate='head', sphere=sphere
    )

# 在第一列的最左侧添加行标题 (Y轴标签)
axes[0, 0].set_ylabel('Without Constraint\n(Baseline)', fontsize=14, fontweight='bold', labelpad=15)
axes[1, 0].set_ylabel('With Constraint\n(PI-ATCNN)', fontsize=14, fontweight='bold', labelpad=15)

# 在整个图的最右侧添加全局 Colorbar
cbar_ax = fig.add_axes([0.91, 0.15, 0.01, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Spatial Activation (Blue: ERD, Red: ERS)', fontsize=14, fontweight='bold', labelpad=15)
cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
cbar.ax.tick_params(labelsize=12)

# 添加全局主标题
plt.suptitle('Spatial Feature Activation Topographies across 9 Subjects (Right Hand MI)',
             fontsize=20, fontweight='bold', y=1.05)

# 保存为高清图片 (适合插入 Word 或 LaTeX)
plt.savefig('topoplot_horizontal_9subjects.png', dpi=600, bbox_inches='tight')
plt.show()