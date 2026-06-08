import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 必须先设置 Seaborn 主题
sns.set_theme(style="whitegrid", font=['SimHei', 'Arial Unicode MS'])

# 2. 然后再强制覆盖字体和所有的字号配置 (防止被 Seaborn 重置)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'font.size': 18,          # 全局默认字号
    'axes.titlesize': 18,     # 子图标题字号
    'axes.labelsize': 18,     # X/Y轴标签字号
    'xtick.labelsize': 18,    # X轴刻度字号
    'ytick.labelsize': 18,    # Y轴刻度字号
    'legend.fontsize': 18     # 图例字号
})

def plot_hyperparameter_results_separated(json_path, target_metric='test_accuracy'):
    # 1. 读取并解析 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_results = data.get('all_results', [])
    if not all_results:
        print("未在 JSON 中找到 'all_results' 键或数据为空，请检查文件。")
        return

    # 展平嵌套的 params 和 metrics 字典
    parsed_data = []
    for i, trial in enumerate(all_results):
        row = {}
        row.update(trial.get('params', {}))
        row.update(trial.get('metrics', {}))
        row['trial_number'] = i + 1
        parsed_data.append(row)

    df = pd.DataFrame(parsed_data)

    if target_metric not in df.columns:
        print(f"目标指标 '{target_metric}' 不存在。")
        return

    # 获取超参数列表
    metric_keys = list(all_results[0].get('metrics', {}).keys())
    param_cols = [col for col in df.columns if col not in metric_keys and col != 'trial_number']

    # ==========================================
    # 图 A：超参数优化历史轨迹图 (独立绘制)
    # ==========================================
    fig_hist, ax_hist = plt.subplots(figsize=(10, 7)) # 设置单张图的画布大小

    sns.scatterplot(x='trial_number', y=target_metric, data=df, ax=ax_hist, color='#4A90E2', alpha=0.6,
                    label='单次试验 (Trial)')

    # 计算并绘制当前最佳值 (最大化目标)
    df['best_value'] = df[target_metric].cummax()
    sns.lineplot(x='trial_number', y='best_value', data=df, ax=ax_hist, color='#E94C3D', linewidth=2,
                 label='最佳值历史 (Best Value)')


    ax_hist.set_xlabel('试验序号 (Trial Number)')
    ax_hist.set_ylabel(f'目标值 ({target_metric})')
    ax_hist.legend()

    plt.tight_layout()
    hist_filename = f'hpo_history_{target_metric}.png'
    plt.savefig(hist_filename, dpi=300, bbox_inches='tight')
    print(f"图 (a) 已成功生成并保存为 '{hist_filename}'")
    plt.show()

    # ==========================================
    # 图 B：参数重要性分析 - 水平柱状图 (独立绘制)
    # ==========================================
    fig_bar, ax_bar = plt.subplots(figsize=(10, 6)) # 设置单张图的画布大小

    # 计算超参数与目标指标的绝对相关性
    corr_dict = {}
    target_series = pd.to_numeric(df[target_metric], errors='coerce')

    for col in param_cols:
        col_data = pd.to_numeric(df[col], errors='coerce')
        # 处理非数值型（类别）超参数：转换为整数后再计算相关性
        if col_data.isna().all() and not df[col].isna().all():
            col_data = pd.Series(pd.factorize(df[col])[0])

        corr = col_data.corr(target_series)
        corr_dict[col] = abs(corr) if pd.notna(corr) else 0.0

    # 转换为DataFrame，并反转顺序让第一个参数在最上方显示
    df_corr = pd.DataFrame(list(corr_dict.items()), columns=['Parameter', 'Absolute Correlation'])
    df_corr = df_corr.iloc[::-1]

    # 绘制匹配图片的水平柱状图
    ax_bar.barh(df_corr['Parameter'], df_corr['Absolute Correlation'], color='#1f77b4', zorder=3)

    # 还原图片中的网格和边框样式 (闭合边框，浅色全网格)
    ax_bar.grid(True, color='#EEEEEE', linestyle='-', linewidth=1, zorder=0)
    for spine in ax_bar.spines.values():
        spine.set_visible(True)
        spine.set_color('#888888')
        spine.set_linewidth(1)

    ax_bar.set_xlabel('参数重要性系数')
    ax_bar.set_ylabel('')


    plt.tight_layout()
    bar_filename = f'hpo_importance_{target_metric}.png'
    plt.savefig(bar_filename, dpi=300, bbox_inches='tight')
    print(f"图 (b) 已成功生成并保存为 '{bar_filename}'")
    plt.show()


# 运行代码
plot_hyperparameter_results_separated('hyperparameter_search_results_20251025_194739.json',
                                     target_metric='test_accuracy')