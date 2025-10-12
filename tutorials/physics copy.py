import matplotlib.pyplot as plt
import numpy as np

# 数据
labels = ["A", "B", "C", "D", "E"]

# MLE: mean ± std
mle_means = [5.18, 7.53, 7.11, 7.63, 7.23]
mle_stds  = [0.81, 2.21, 1.82, 2.32, 1.94]

# AUC: mean ± std
auc_means = [0.9626, 0.9493, 0.9482, 0.9515, 0.9518]
auc_stds  = [0.0086, 0.0128, 0.0137, 0.0108, 0.0105]

# 自定义颜色，A 用红色，其余保持原色
colors_mle = ["red", "skyblue", "skyblue", "skyblue", "skyblue"]
colors_auc = ["red", "lightgreen", "lightgreen", "lightgreen", "lightgreen"]

# 画图 - MLE
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(labels, mle_means, yerr=mle_stds, capsize=5, color=colors_mle)
plt.ylabel("MLE (mm)")
plt.title("Mean Localization Error (MLE)")

# 画图 - AUC
plt.subplot(1, 2, 2)
plt.bar(labels, auc_means, yerr=auc_stds, capsize=5, color=colors_auc)
plt.ylabel("AUC")
plt.title("Area Under Curve (AUC)")

plt.tight_layout()

# 保存高清图片
plt.savefig("experiment_results.png", dpi=300, bbox_inches="tight")  # png 高清
plt.savefig("experiment_results.pdf", bbox_inches="tight")          # 矢量图（无限放大不失真）

plt.show()
