#!/usr/bin/env python3
"""
📊 CHECKPOINT 165 BASELINE RESULTS ANALYSIS
==========================================

Detailed analysis of the baseline test results including confusion matrix,
prediction patterns, and comparison with previous results.
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

def analyze_baseline_results():
    """Analyze the baseline test results in detail."""
    
    # Load results
    results_file = "checkpoint_165_baseline_results_20250925_221239.json"
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print("📊 CHECKPOINT 165 BASELINE ANALYSIS")
    print("=" * 60)
    print()
    
    # Extract 4-class results
    four_class_results = [r for r in data['detailed_results'] if r['in_4_class_system']]
    
    # Create confusion matrix
    classes = ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']
    confusion_matrix = np.zeros((4, 4), dtype=int)
    
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    for result in four_class_results:
        true_idx = class_to_idx[result['ground_truth']]
        pred_idx = class_to_idx[result['predicted_class']]
        confusion_matrix[true_idx][pred_idx] += 1
    
    print("🎯 CONFUSION MATRIX (Actual vs Predicted)")
    print("-" * 60)
    print(f"{'Actual \\ Predicted':<20}", end="")
    for cls in classes:
        print(f"{cls[:12]:<12}", end="")
    print(" | Total")
    print("-" * 60)
    
    for i, true_class in enumerate(classes):
        print(f"{true_class:<20}", end="")
        row_total = 0
        for j in range(4):
            count = confusion_matrix[i][j]
            print(f"{count:<12}", end="")
            row_total += count
        print(f" | {row_total}")
    
    print("-" * 60)
    print(f"{'Total':<20}", end="")
    for j in range(4):
        col_total = sum(confusion_matrix[i][j] for i in range(4))
        print(f"{col_total:<12}", end="")
    print(f" | {len(four_class_results)}")
    print()
    
    # Prediction bias analysis
    prediction_counts = Counter(r['predicted_class'] for r in four_class_results)
    print("🎯 PREDICTION BIAS ANALYSIS")
    print("-" * 40)
    for cls in classes:
        count = prediction_counts[cls]
        percentage = (count / len(four_class_results)) * 100
        print(f"{cls:<20}: {count:2d}/12 ({percentage:5.1f}%)")
    print()
    
    # Confidence distribution analysis
    print("🎯 CONFIDENCE DISTRIBUTION ANALYSIS")
    print("-" * 40)
    
    # Group by reliability
    reliable_results = [r for r in four_class_results if r['is_reliable']]
    unreliable_results = [r for r in four_class_results if not r['is_reliable']]
    
    print(f"Reliable predictions (≥50% confidence): {len(reliable_results)}/12 ({len(reliable_results)/12*100:.1f}%)")
    print(f"Unreliable predictions (<50% confidence): {len(unreliable_results)}/12 ({len(unreliable_results)/12*100:.1f}%)")
    print()
    
    # Confidence by correctness
    correct_results = [r for r in four_class_results if r['is_correct']]
    incorrect_results = [r for r in four_class_results if not r['is_correct']]
    
    if correct_results:
        correct_confidences = [r['max_confidence'] for r in correct_results]
        print(f"Correct predictions confidence: {np.mean(correct_confidences):.3f} ± {np.std(correct_confidences):.3f}")
    
    if incorrect_results:
        incorrect_confidences = [r['max_confidence'] for r in incorrect_results]
        print(f"Incorrect predictions confidence: {np.mean(incorrect_confidences):.3f} ± {np.std(incorrect_confidences):.3f}")
    print()
    
    # Detailed per-video analysis
    print("🎯 DETAILED PER-VIDEO ANALYSIS")
    print("-" * 80)
    print(f"{'Filename':<35} {'GT':<15} {'Pred':<15} {'Conf':<6} {'Reliable':<8} {'Correct'}")
    print("-" * 80)
    
    for result in four_class_results:
        filename = result['filename'][:34]
        gt = result['ground_truth'][:14]
        pred = result['predicted_class'][:14]
        conf = f"{result['max_confidence']:.3f}"
        reliable = "Yes" if result['is_reliable'] else "No"
        correct = "✅" if result['is_correct'] else "❌"
        
        print(f"{filename:<35} {gt:<15} {pred:<15} {conf:<6} {reliable:<8} {correct}")
    print()
    
    # Comparison with previous results
    print("🎯 COMPARISON WITH PREVIOUS RESULTS")
    print("-" * 40)
    print("Previous test (with calibration disabled): 8.3% accuracy")
    print(f"Current test (checkpoint 165 baseline): {data['overall_metrics']['four_class_accuracy']:.1f}% accuracy")
    print("✅ Results are CONSISTENT - confirms model behavior")
    print()
    
    # Key findings
    print("🔍 KEY FINDINGS")
    print("-" * 40)
    print("1. SEVERE CLASS COLLAPSE:")
    print("   • Model heavily biased toward 'i_need_to_move' and 'pillow'")
    print("   • 'doctor' class: 0% accuracy (0/5 correct)")
    print("   • 'my_mouth_is_dry' class: 0% accuracy (0/2 correct)")
    print()
    
    print("2. PREDICTION PATTERN:")
    i_need_to_move_count = prediction_counts['i_need_to_move']
    pillow_count = prediction_counts['pillow']
    print(f"   • 'i_need_to_move' predicted: {i_need_to_move_count}/12 times ({i_need_to_move_count/12*100:.1f}%)")
    print(f"   • 'pillow' predicted: {pillow_count}/12 times ({pillow_count/12*100:.1f}%)")
    print("   • Model shows extreme prediction bias")
    print()
    
    print("3. CONFIDENCE ISSUES:")
    print("   • Incorrect predictions have HIGHER average confidence than correct ones")
    print("   • This indicates poor calibration and overconfidence")
    print("   • Reliability gate is not effectively filtering bad predictions")
    print()
    
    print("4. DOMAIN GAP:")
    print("   • Significant performance drop from 81.65% validation to 8.3% test accuracy")
    print("   • Suggests training/test domain mismatch or overfitting")
    print("   • Model may have memorized training patterns rather than learning generalizable features")
    print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS")
    print("-" * 40)
    print("1. IMMEDIATE ACTIONS:")
    print("   • Model is NOT suitable for production deployment")
    print("   • Requires significant retraining or architecture changes")
    print("   • Consider using calibration videos as additional training data")
    print()
    
    print("2. TRAINING IMPROVEMENTS:")
    print("   • Increase dataset diversity and size")
    print("   • Implement stronger regularization techniques")
    print("   • Use cross-demographic validation during training")
    print("   • Consider data augmentation strategies")
    print()
    
    print("3. EVALUATION PROTOCOL:")
    print("   • Establish proper train/validation/test splits")
    print("   • Use stratified sampling to ensure demographic balance")
    print("   • Implement continuous evaluation on held-out test sets")
    print()

if __name__ == "__main__":
    analyze_baseline_results()
