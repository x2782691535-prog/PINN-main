# 根据用户提供的新表格数据更新图表脚本
# 用于修改图表使其与更新后的表格数据对应

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib全局字体参数以适合Word文档
plt.rcParams.update({
    'font.size': 14,          # 基础字体大小
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

# 设置matplotlib字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def save_figure(fig, filename, dpi=600):
    """保存图形到指定目录，高分辨率适合Word文档"""
    fig_dir = os.path.join(os.path.dirname(__file__), 'updated_figures')
    os.makedirs(fig_dir, exist_ok=True)
    filepath = os.path.join(fig_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"图表已保存至: {filepath}")

def create_updated_comparison_charts():
    """根据用户提供的新表格数据创建对比图表"""
    
    # 用户提供的新表格数据
    updated_data = {
        'Letter_Code': ['A', 'B', 'C', 'D', 'E'],
        'Configuration': [
            'Proposed(Full)',
            'Data-Driven Only', 
            'w/o Physical Consistency',
            'w/o Physical-Inspired Reg',
            'w/L2 Reg.(replaced)'
        ],
        'MLE_mm': [5.18, 7.23, 7.53, 7.11, 7.63],
        'MLE_std': [0.81, 1.94, 2.21, 1.82, 2.32],
        'AUC': [0.9626, 0.9518, 0.9493, 0.9482, 0.9515],
        'AUC_std': [0.0086, 0.0105, 0.0128, 0.0137, 0.0108]
    }
    
    # 转换为DataFrame
    df = pd.DataFrame(updated_data)
    
    print("使用的新数据:")
    print(df.to_string(index=False))
    print()
    
    # 创建对比图表 (类似原始图表的样式)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 子图1: Mean Localization Error (MLE)
    bars1 = ax1.bar(range(len(df)), df['MLE_mm'], yerr=df['MLE_std'], 
                    capsize=6, width=0.6, alpha=0.8)
    
    # 为基线实验(A)着色为红色，其他为蓝色
    for i, letter in enumerate(df['Letter_Code']):
        if letter == 'A':
            bars1[i].set_color('red')
        else:
            bars1[i].set_color('skyblue')
    
    ax1.set_ylabel('Mean Localization Error (mm)', fontsize=18, fontweight='bold')
    ax1.set_title('Mean Localization Error (MLE)', fontsize=20, fontweight='bold', pad=15)
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df['Letter_Code'], fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 10)  # 设置y轴范围以便更好显示
    
    # 子图2: Area Under Curve (AUC)
    bars2 = ax2.bar(range(len(df)), df['AUC'], yerr=df['AUC_std'], 
                    capsize=6, width=0.6, alpha=0.8)
    
    # 为基线实验(A)着色为红色，其他为绿色
    for i, letter in enumerate(df['Letter_Code']):
        if letter == 'A':
            bars2[i].set_color('red')
        else:
            bars2[i].set_color('lightgreen')
    
    ax2.set_ylabel('AUC', fontsize=18, fontweight='bold')
    ax2.set_title('Area Under Curve (AUC)', fontsize=20, fontweight='bold', pad=15)
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels(df['Letter_Code'], fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.9, 1.0)  # 设置y轴范围以便更好显示差异
    
    # 调整子图间距
    plt.tight_layout(pad=3.0)
    
    # 保存图表
    save_figure(fig, 'updated_performance_comparison.png')
    plt.show()
    
    return df

def create_detailed_table_figure():
    """创建详细的表格图形"""
    
    # 用户提供的新表格数据
    table_data = {
        'Letter code': ['A', 'B', 'C', 'D', 'E'],
        'Configuration': [
            'Proposed(Full)',
            'Data-Driven\nOnly', 
            'w/o Physical\nConsistency',
            'w/o Physical-\nInspired Reg',
            'w/L2 Reg.\n(replaced)'
        ],
        'MLE(mm)': ['5.18 ± 0.81', '7.23 ± 1.94', '7.53 ± 2.21', '7.11 ± 1.82', '7.63 ± 2.32'],
        'AUC': ['0.9626 ± 0.0086', '0.9518 ± 0.0105', '0.9493 ± 0.0128', '0.9482 ± 0.0137', '0.9515 ± 0.0108']
    }
    
    # 创建表格图形
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # 创建表格
    table = ax.table(cellText=[[table_data[col][i] for col in table_data.keys()] 
                              for i in range(len(table_data['Letter code']))],
                    colLabels=list(table_data.keys()),
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2.5)
    
    # 设置标题行样式
    for i in range(len(table_data.keys())):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置数据行样式 - 基线行(A)使用不同颜色
    for i in range(1, len(table_data['Letter code']) + 1):
        for j in range(len(table_data.keys())):
            if table_data['Letter code'][i-1] == 'A':  # 基线行
                table[(i, j)].set_facecolor('#FFCDD2')
            else:
                table[(i, j)].set_facecolor('#F5F5F5')
            table[(i, j)].set_text_props(weight='bold')
    
    plt.title('Updated Experimental Results Comparison Table', 
              fontsize=20, fontweight='bold', pad=20)
    
    # 保存表格图形
    save_figure(fig, 'updated_results_table.png')
    plt.show()

def create_combined_figure():
    """创建组合图表(表格+柱状图)"""
    
    # 数据
    updated_data = {
        'Letter_Code': ['A', 'B', 'C', 'D', 'E'],
        'Configuration': [
            'Proposed(Full)',
            'Data-Driven Only', 
            'w/o Physical Consistency',
            'w/o Physical-Inspired Reg',
            'w/L2 Reg.(replaced)'
        ],
        'MLE_mm': [5.18, 7.23, 7.53, 7.11, 7.63],
        'MLE_std': [0.81, 1.94, 2.21, 1.82, 2.32],
        'AUC': [0.9626, 0.9518, 0.9493, 0.9482, 0.9515],
        'AUC_std': [0.0086, 0.0105, 0.0128, 0.0137, 0.0108]
    }
    
    df = pd.DataFrame(updated_data)
    
    # 创建组合图
    fig = plt.figure(figsize=(24, 12))
    
    # 上半部分：表格
    ax_table = plt.subplot2grid((2, 1), (0, 0), rowspan=1)
    ax_table.axis('off')
    
    table_data = [
        ['Letter code', 'A', 'B', 'C', 'D', 'E'],
        ['Configuration', 'Proposed(Full)', 'Data-Driven\nOnly', 'w/o Physical\nConsistency', 'w/o Physical-\nInspired Reg', 'w/L2 Reg.\n(replaced)'],
        ['MLE(mm)', '5.18 ± 0.81', '7.23 ± 1.94', '7.53 ± 2.21', '7.11 ± 1.82', '7.63 ± 2.32'],
        ['AUC', '0.9626 ± 0.0086', '0.9518 ± 0.0105', '0.9493 ± 0.0128', '0.9482 ± 0.0137', '0.9515 ± 0.0108']
    ]
    
    table = ax_table.table(cellText=table_data,
                          cellLoc='center',
                          loc='center',
                          bbox=[0.1, 0.2, 0.8, 0.6])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # 设置表格样式
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            if j == 0:  # 第一列
                table[(i, j)].set_facecolor('#E3F2FD')
                table[(i, j)].set_text_props(weight='bold')
            elif j == 1:  # A列 (基线)
                table[(i, j)].set_facecolor('#FFCDD2')
                table[(i, j)].set_text_props(weight='bold')
            else:
                table[(i, j)].set_facecolor('#F5F5F5')
    
    # 下半部分：柱状图
    ax_charts = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    
    # 创建双y轴图表
    x = np.arange(len(df['Letter_Code']))
    width = 0.35
    
    # MLE柱状图
    bars1 = ax_charts.bar(x - width/2, df['MLE_mm'], width, yerr=df['MLE_std'],
                         capsize=5, alpha=0.8, label='MLE (mm)')
    
    # 为基线着色
    for i, letter in enumerate(df['Letter_Code']):
        if letter == 'A':
            bars1[i].set_color('red')
        else:
            bars1[i].set_color('skyblue')
    
    ax_charts.set_xlabel('Configuration', fontsize=16, fontweight='bold')
    ax_charts.set_ylabel('Mean Localization Error (mm)', fontsize=16, fontweight='bold')
    ax_charts.set_xticks(x)
    ax_charts.set_xticklabels(df['Letter_Code'], fontsize=14, fontweight='bold')
    
    # 创建第二个y轴用于AUC
    ax2 = ax_charts.twinx()
    bars2 = ax2.bar(x + width/2, df['AUC'], width, yerr=df['AUC_std'],
                   capsize=5, alpha=0.8, label='AUC')
    
    for i, letter in enumerate(df['Letter_Code']):
        if letter == 'A':
            bars2[i].set_color('darkred')
        else:
            bars2[i].set_color('green')
    
    ax2.set_ylabel('AUC', fontsize=16, fontweight='bold')
    
    # 添加图例
    lines1, labels1 = ax_charts.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_charts.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.suptitle('Updated Experimental Results: Table and Charts', 
                fontsize=22, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, 'updated_combined_results.png')
    plt.show()

def main():
    """主函数"""
    print("=== 根据新表格数据更新图表 ===\n")
    
    print("1. 创建更新的对比图表...")
    df = create_updated_comparison_charts()
    
    print("\n2. 创建详细表格图形...")
    create_detailed_table_figure()
    
    print("\n3. 创建组合图表...")
    create_combined_figure()
    
    print("\n=== 所有图表已更新完成 ===")
    print("图表保存在 'updated_figures' 目录中")
    
    # 显示数据对比
    print("\n数据摘要:")
    print(f"配置A (基线): MLE={df.loc[0, 'MLE_mm']:.2f}±{df.loc[0, 'MLE_std']:.2f}mm, AUC={df.loc[0, 'AUC']:.4f}±{df.loc[0, 'AUC_std']:.4f}")
    for i in range(1, len(df)):
        mle_change = df.loc[i, 'MLE_mm'] - df.loc[0, 'MLE_mm']
        auc_change = df.loc[i, 'AUC'] - df.loc[0, 'AUC']
        print(f"配置{df.loc[i, 'Letter_Code']}: MLE={df.loc[i, 'MLE_mm']:.2f}±{df.loc[i, 'MLE_std']:.2f}mm ({mle_change:+.2f}), AUC={df.loc[i, 'AUC']:.4f}±{df.loc[i, 'AUC_std']:.4f} ({auc_change:+.4f})")

if __name__ == "__main__":
    main()





