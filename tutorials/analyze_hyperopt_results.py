# 超参数调优结果分析脚本

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def load_results(json_file):
    """加载超参数搜索结果"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def print_summary(data):
    """打印结果摘要"""
    print("=" * 100)
    print("超参数搜索结果摘要")
    print("=" * 100)
    print(f"搜索类型: {data['search_type']}")
    print(f"评估组合数: {data['total_combinations']}")
    print(f"总耗时: {data['elapsed_time_seconds']/3600:.2f} 小时")
    
    print(f"\n🏆 最佳参数组合:")
    for param, value in data['best_params'].items():
        print(f"  {param:25s} = {value}")
    
    print(f"\n📊 最佳性能指标:")
    for metric, value in data['best_metrics'].items():
        if isinstance(value, float):
            print(f"  {metric:25s} = {value:.4f}")
        else:
            print(f"  {metric:25s} = {value}")
    print("=" * 100)


def create_dataframe(data):
    """创建结果DataFrame"""
    results = data['all_results']
    
    rows = []
    for r in results:
        row = {}
        row.update(r['params'])
        row.update(r['metrics'])
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def top_n_configurations(df, n=10, metric='val_accuracy'):
    """打印Top N配置"""
    print(f"\n{'='*100}")
    print(f"Top {n} 配置（按 {metric} 排序）")
    print('='*100)
    
    df_sorted = df.sort_values(by=metric, ascending=False)
    
    for idx, (i, row) in enumerate(df_sorted.head(n).iterrows()):
        print(f"\n排名 {idx+1}:")
        print(f"  Dropout={row['dropout_rate']:.2f}, PhysicsWeight={row['init_physics_weight']:.2f}, "
              f"LR={row['learning_rate']:.4f}")
        print(f"  BatchSize={int(row['batch_size'])}, WeightDecay={row['weight_decay']:.5f}, "
              f"Repeat={int(row['repeat_factor'])}")
        print(f"  训练={row['train_accuracy']:.2f}%, 验证={row['val_accuracy']:.2f}%, "
              f"测试={row['test_accuracy']:.2f}%")
        print(f"  F1={row['f1_score']:.4f}, Kappa={row['kappa']:.4f}, "
              f"过拟合间隔={row['overfit_gap']:.2f}%")
    
    print('='*100)


def parameter_importance_analysis(df):
    """分析各参数对性能的影响"""
    print(f"\n{'='*100}")
    print("参数重要性分析（相关性）")
    print('='*100)
    
    params = ['dropout_rate', 'init_physics_weight', 'learning_rate', 
              'batch_size', 'weight_decay', 'repeat_factor']
    
    metrics = ['val_accuracy', 'test_accuracy', 'overfit_gap']
    
    print(f"\n{'参数':<25s} ", end='')
    for metric in metrics:
        print(f"{metric:<20s}", end='')
    print()
    print("-" * 100)
    
    for param in params:
        print(f"{param:<25s} ", end='')
        for metric in metrics:
            corr = df[param].corr(df[metric])
            print(f"{corr:>8.4f}{'':>12s}", end='')
        print()
    
    print('='*100)
    print("\n说明:")
    print("  - 正相关: 参数增加 → 指标增加")
    print("  - 负相关: 参数增加 → 指标减少")
    print("  - 对于过拟合间隔，负相关说明参数有助于减少过拟合")


def plot_parameter_effects(df, save_dir='hyperopt_plots'):
    """可视化各参数对性能的影响"""
    Path(save_dir).mkdir(exist_ok=True)
    
    params = ['dropout_rate', 'init_physics_weight', 'learning_rate', 
              'batch_size', 'weight_decay', 'repeat_factor']
    
    metrics = ['val_accuracy', 'test_accuracy', 'overfit_gap']
    
    for param in params:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f'{param} 对性能的影响', fontsize=14, fontweight='bold')
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # 按参数值分组并计算平均值
            grouped = df.groupby(param)[metric].agg(['mean', 'std'])
            
            ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                       marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
            ax.set_xlabel(param, fontsize=11)
            ax.set_ylabel(metric, fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{metric}', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{param}_effects.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\n可视化图表已保存到 {save_dir}/ 目录")


def plot_overfitting_analysis(df, save_dir='hyperopt_plots'):
    """过拟合分析可视化"""
    Path(save_dir).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. 训练-测试准确率散点图
    ax = axes[0, 0]
    scatter = ax.scatter(df['train_accuracy'], df['test_accuracy'], 
                        c=df['overfit_gap'], cmap='RdYlGn_r', alpha=0.6, s=50)
    ax.plot([70, 100], [70, 100], 'k--', alpha=0.3, label='理想线(无过拟合)')
    ax.set_xlabel('训练准确率 (%)', fontsize=11)
    ax.set_ylabel('测试准确率 (%)', fontsize=11)
    ax.set_title('训练 vs 测试准确率', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='过拟合间隔')
    
    # 2. 过拟合间隔分布
    ax = axes[0, 1]
    ax.hist(df['overfit_gap'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(df['overfit_gap'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f'平均值={df["overfit_gap"].mean():.2f}%')
    ax.set_xlabel('过拟合间隔 (%)', fontsize=11)
    ax.set_ylabel('频数', fontsize=11)
    ax.set_title('过拟合间隔分布', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Dropout vs 过拟合
    ax = axes[1, 0]
    grouped = df.groupby('dropout_rate')['overfit_gap'].agg(['mean', 'std'])
    ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
               marker='o', capsize=5, capthick=2, linewidth=2, markersize=10)
    ax.set_xlabel('Dropout率', fontsize=11)
    ax.set_ylabel('过拟合间隔 (%)', fontsize=11)
    ax.set_title('Dropout率 vs 过拟合程度', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 4. 物理权重 vs 过拟合
    ax = axes[1, 1]
    grouped = df.groupby('init_physics_weight')['overfit_gap'].agg(['mean', 'std'])
    ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
               marker='s', capsize=5, capthick=2, linewidth=2, markersize=10, color='orange')
    ax.set_xlabel('物理损失权重', fontsize=11)
    ax.set_ylabel('过拟合间隔 (%)', fontsize=11)
    ax.set_title('物理损失权重 vs 过拟合程度', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/overfitting_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"过拟合分析图已保存到 {save_dir}/overfitting_analysis.png")


def plot_performance_comparison(df, save_dir='hyperopt_plots'):
    """性能指标对比"""
    Path(save_dir).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. 准确率箱线图
    ax = axes[0, 0]
    data_to_plot = [df['train_accuracy'], df['val_accuracy'], df['test_accuracy']]
    bp = ax.boxplot(data_to_plot, labels=['训练', '验证', '测试'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightcoral']):
        patch.set_facecolor(color)
    ax.set_ylabel('准确率 (%)', fontsize=11)
    ax.set_title('准确率分布对比', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 2. F1分数 vs Kappa
    ax = axes[0, 1]
    scatter = ax.scatter(df['f1_score'], df['kappa'], c=df['test_accuracy'], 
                        cmap='viridis', alpha=0.6, s=50)
    ax.set_xlabel('F1分数', fontsize=11)
    ax.set_ylabel('Kappa系数', fontsize=11)
    ax.set_title('F1分数 vs Kappa系数', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='测试准确率(%)')
    
    # 3. 学习率影响
    ax = axes[1, 0]
    grouped = df.groupby('learning_rate')[['train_accuracy', 'val_accuracy', 'test_accuracy']].mean()
    x = np.arange(len(grouped))
    width = 0.25
    ax.bar(x - width, grouped['train_accuracy'], width, label='训练', alpha=0.8)
    ax.bar(x, grouped['val_accuracy'], width, label='验证', alpha=0.8)
    ax.bar(x + width, grouped['test_accuracy'], width, label='测试', alpha=0.8)
    ax.set_xlabel('学习率', fontsize=11)
    ax.set_ylabel('准确率 (%)', fontsize=11)
    ax.set_title('学习率对性能的影响', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{lr:.4f}' for lr in grouped.index])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 批次大小影响
    ax = axes[1, 1]
    grouped = df.groupby('batch_size')[['train_accuracy', 'val_accuracy', 'test_accuracy']].mean()
    x = np.arange(len(grouped))
    ax.bar(x - width, grouped['train_accuracy'], width, label='训练', alpha=0.8)
    ax.bar(x, grouped['val_accuracy'], width, label='验证', alpha=0.8)
    ax.bar(x + width, grouped['test_accuracy'], width, label='测试', alpha=0.8)
    ax.set_xlabel('批次大小', fontsize=11)
    ax.set_ylabel('准确率 (%)', fontsize=11)
    ax.set_title('批次大小对性能的影响', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([int(bs) for bs in grouped.index])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"性能对比图已保存到 {save_dir}/performance_comparison.png")


def generate_report(json_file):
    """生成完整分析报告"""
    print("\n开始分析超参数搜索结果...\n")
    
    # 加载数据
    data = load_results(json_file)
    
    # 打印摘要
    print_summary(data)
    
    # 创建DataFrame
    df = create_dataframe(data)
    
    # Top 10配置
    top_n_configurations(df, n=10, metric='val_accuracy')
    top_n_configurations(df, n=10, metric='test_accuracy')
    
    # 参数重要性分析
    parameter_importance_analysis(df)
    
    # 生成可视化
    print("\n生成可视化图表...")
    plot_parameter_effects(df)
    plot_overfitting_analysis(df)
    plot_performance_comparison(df)
    
    print("\n✅ 分析完成！")


if __name__ == '__main__':
    import sys
    
    # 检查是否提供了JSON文件路径
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        # 自动查找最新的结果文件
        import glob
        files = glob.glob('hyperparameter_search_results_*.json')
        if files:
            json_file = max(files, key=lambda x: Path(x).stat().st_mtime)
            print(f"自动使用最新结果文件: {json_file}\n")
        else:
            print("错误: 未找到超参数搜索结果文件")
            print("用法: python analyze_hyperopt_results.py [结果文件.json]")
            sys.exit(1)
    
    generate_report(json_file)































