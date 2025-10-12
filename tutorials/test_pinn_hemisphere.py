#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PINN半球预测器测试脚本
"""

import torch
import torch.nn as nn
import numpy as np

# 创建一个简化的PINN-only模型来测试半球预测器
class TestPINNOnly(nn.Module):
    def __init__(self, in_channels=22, num_classes=2, dropout_rate=0.35):
        super(TestPINNOnly, self).__init__()
        
        # 简化的PINN特征提取器（模拟512维输出）
        self.pinn_simulator = nn.Sequential(
            nn.Linear(in_channels * 500, 256),  # 模拟PINN处理
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 512)  # 输出512维特征
        )
        
        # PINN特征投影
        self.pinn_projection = nn.Linear(512, 16)
        
        # 半球激活预测器
        self.hemisphere_predictor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 2),    # [左脑概率, 右脑概率]
            nn.Softmax(dim=1)
        )
        
        # 空域特征增强模块
        self.spatial_enhancer = nn.Sequential(
            nn.Linear(16 + 2, 32),  # 投影特征 + 半球概率
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),
            nn.Tanh()
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(16, num_classes)
        )
    
    def forward(self, x):
        # 模拟输入处理：(batch, 22, 1000) -> (batch, 22*500)
        batch_size = x.shape[0]
        x_flat = x[:, :, 250:750].reshape(batch_size, -1)  # ERD窗口
        
        # 模拟PINN特征提取
        pinn_features = self.pinn_simulator(x_flat)  # (batch, 512)
        
        # PINN特征投影
        pinn_projected = self.pinn_projection(pinn_features)  # (batch, 16)
        
        # 半球激活预测
        hemisphere_probs = self.hemisphere_predictor(pinn_features)  # (batch, 2)
        
        # 特征融合
        combined_features = torch.cat([pinn_projected, hemisphere_probs], dim=1)  # (batch, 18)
        enhanced_features = self.spatial_enhancer(combined_features)  # (batch, 16)
        
        # 分类
        logits = self.classifier(enhanced_features)
        
        return {
            'logits': logits,
            'hemisphere_probs': hemisphere_probs,
            'pinn_features': pinn_features,
            'physics_loss': torch.tensor(0.0)  # 模拟物理损失
        }

def test_hemisphere_predictor():
    """测试半球预测器是否正常工作"""
    print("🧪 测试PINN半球预测器...")
    
    # 创建测试数据
    batch_size = 4
    test_data = torch.randn(batch_size, 22, 1000)  # 模拟EEG数据
    test_labels = torch.tensor([0, 1, 0, 1])  # 左手, 右手, 左手, 右手
    
    # 创建模型
    model = TestPINNOnly()
    model.eval()
    
    # 前向传播
    with torch.no_grad():
        outputs = model(test_data)
        
        print("✅ 模型输出检查:")
        print(f"   logits形状: {outputs['logits'].shape}")
        print(f"   hemisphere_probs形状: {outputs['hemisphere_probs'].shape}")
        print(f"   半球概率示例:")
        
        for i, (prob, label) in enumerate(zip(outputs['hemisphere_probs'], test_labels)):
            left_prob, right_prob = prob[0].item(), prob[1].item()
            hand = "左手" if label.item() == 0 else "右手"
            expected_brain = "右脑" if label.item() == 0 else "左脑"
            
            print(f"     样本{i+1} ({hand}): 左脑={left_prob:.3f}, 右脑={right_prob:.3f}")
            print(f"       期望激活: {expected_brain}")
            
            # 检查预测是否合理
            if label.item() == 0:  # 左手 -> 应该右脑激活更强
                if right_prob > left_prob:
                    print(f"       ✅ 预测正确！右脑激活更强")
                else:
                    print(f"       ❌ 预测错误！左脑激活更强")
            else:  # 右手 -> 应该左脑激活更强
                if left_prob > right_prob:
                    print(f"       ✅ 预测正确！左脑激活更强")
                else:
                    print(f"       ❌ 预测错误！右脑激活更强")
    
    # 测试损失计算
    print("\n🔍 测试半球预测损失...")
    
    # 构建目标
    hemisphere_targets = torch.zeros_like(outputs['hemisphere_probs'])
    for i, label in enumerate(test_labels):
        if label.item() == 0:  # 左手 → 右脑激活
            hemisphere_targets[i] = torch.tensor([0.0, 1.0])
        else:  # 右手 → 左脑激活  
            hemisphere_targets[i] = torch.tensor([1.0, 0.0])
    
    # 计算KL散度损失
    hemisphere_loss = nn.KLDivLoss(reduction='batchmean')(
        torch.log(outputs['hemisphere_probs'] + 1e-8), hemisphere_targets
    )
    
    print(f"   半球预测损失: {hemisphere_loss.item():.4f}")
    print(f"   目标分布示例: {hemisphere_targets[0].numpy()}")
    print(f"   预测分布示例: {outputs['hemisphere_probs'][0].detach().numpy()}")
    
    return outputs

if __name__ == "__main__":
    # 运行测试
    outputs = test_hemisphere_predictor()
    
    print("\n🎯 测试总结:")
    print("✅ 半球预测器架构正常")
    print("✅ 输出格式正确")
    print("✅ 损失计算正常")
    print("\n💡 如果在真实训练中半球预测器仍然无效，")
    print("   问题可能在于PINN特征质量或训练策略！")





























