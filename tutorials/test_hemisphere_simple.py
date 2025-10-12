#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple hemisphere predictor test
"""

import torch
import torch.nn as nn
import numpy as np

# Test if hemisphere predictor logic works
def test_simple():
    print("Testing hemisphere predictor logic...")
    
    # Create test hemisphere probabilities
    hemisphere_probs = torch.tensor([
        [0.3, 0.7],  # Sample 1: right brain activation (left hand)
        [0.8, 0.2],  # Sample 2: left brain activation (right hand)
        [0.4, 0.6],  # Sample 3: right brain activation (left hand)
        [0.9, 0.1],  # Sample 4: left brain activation (right hand)
    ])
    
    test_labels = torch.tensor([0, 1, 0, 1])  # left, right, left, right
    
    print("Hemisphere probabilities:")
    print("  [left_brain, right_brain]")
    for i, (prob, label) in enumerate(zip(hemisphere_probs, test_labels)):
        hand = "left" if label.item() == 0 else "right"
        expected = "right_brain" if label.item() == 0 else "left_brain"
        print(f"  Sample {i+1} ({hand}): {prob.numpy()} -> expect {expected}")
    
    # Test target construction
    hemisphere_targets = torch.zeros_like(hemisphere_probs)
    for i, label in enumerate(test_labels):
        if label.item() == 0:  # left hand -> right brain
            hemisphere_targets[i] = torch.tensor([0.0, 1.0])
        else:  # right hand -> left brain
            hemisphere_targets[i] = torch.tensor([1.0, 0.0])
    
    print("\nTargets:")
    for i, target in enumerate(hemisphere_targets):
        print(f"  Target {i+1}: {target.numpy()}")
    
    # Test loss calculation
    hemisphere_loss = nn.KLDivLoss(reduction='batchmean')(
        torch.log(hemisphere_probs + 1e-8), hemisphere_targets
    )
    
    print(f"\nHemisphere loss: {hemisphere_loss.item():.4f}")
    
    # Check if the logic makes sense
    print("\nLogic check:")
    for i, (prob, target, label) in enumerate(zip(hemisphere_probs, hemisphere_targets, test_labels)):
        hand = "left" if label.item() == 0 else "right"
        left_prob, right_prob = prob[0].item(), prob[1].item()
        
        if label.item() == 0:  # left hand
            correct = right_prob > left_prob
            print(f"  Sample {i+1} ({hand}): right_brain={right_prob:.3f} > left_brain={left_prob:.3f} ? {correct}")
        else:  # right hand
            correct = left_prob > right_prob
            print(f"  Sample {i+1} ({hand}): left_brain={left_prob:.3f} > right_brain={right_prob:.3f} ? {correct}")

if __name__ == "__main__":
    test_simple()
    print("\nTest completed!")





























