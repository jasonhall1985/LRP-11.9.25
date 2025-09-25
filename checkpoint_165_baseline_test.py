#!/usr/bin/env python3
"""
🎯 CHECKPOINT 165 BASELINE ACCURACY TEST
=====================================

Comprehensive baseline accuracy assessment of the 81.65% validation accuracy model
on the test set 24.9.25 with uncalibrated predictions and Enhanced Reliability Gate V2.0.

This script establishes the production baseline performance for checkpoint 165.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict, Counter
import pandas as pd
from datetime import datetime

# Import the model loading function
from enhanced_lightweight_training_pipeline import load_enhanced_checkpoint

def apply_enhanced_reliability_gate_v2(probabilities, confidence_threshold=0.5):
    """
    Enhanced Reliability Gate V2.0 with mandatory 50% confidence floor.
    
    Args:
        probabilities: Softmax probabilities from model
        confidence_threshold: Minimum confidence required (default: 0.5)
    
    Returns:
        tuple: (is_reliable, max_confidence, predicted_class)
    """
    max_confidence = float(torch.max(probabilities))
    predicted_class = int(torch.argmax(probabilities))
    
    # Enhanced Reliability Gate V2.0: Mandatory 50% confidence floor
    is_reliable = max_confidence >= confidence_threshold
    
    return is_reliable, max_confidence, predicted_class

def apply_test_time_augmentation(model, video_tensor, device):
    """
    Apply Test-Time Augmentation with 5 temporal windows + horizontal flips.
    
    Args:
        model: Loaded model
        video_tensor: Input video tensor (1, 32, 64, 96)
        device: Torch device
    
    Returns:
        torch.Tensor: Averaged logits from all augmentations
    """
    model.eval()
    all_logits = []
    
    # Original video
    with torch.no_grad():
        logits = model(video_tensor.to(device))
        all_logits.append(logits)
    
    # Horizontal flip
    flipped_video = torch.flip(video_tensor, dims=[3])  # Flip width dimension
    with torch.no_grad():
        logits = model(flipped_video.to(device))
        all_logits.append(logits)
    
    # Temporal windows (5 different starting points)
    frames = video_tensor.shape[1]  # Should be 32
    for start_frame in [0, 2, 4, 6, 8]:
        if start_frame + frames <= video_tensor.shape[1]:
            windowed_video = video_tensor[:, start_frame:start_frame+frames]
        else:
            # Pad if needed
            windowed_video = video_tensor
        
        with torch.no_grad():
            logits = model(windowed_video.to(device))
            all_logits.append(logits)
        
        # Also test flipped version of windowed video
        flipped_windowed = torch.flip(windowed_video, dims=[3])
        with torch.no_grad():
            logits = model(flipped_windowed.to(device))
            all_logits.append(logits)
    
    # Average all logits
    averaged_logits = torch.mean(torch.stack(all_logits), dim=0)
    return averaged_logits

def test_checkpoint_165_baseline():
    """
    Execute comprehensive baseline testing of checkpoint 165 model.
    """
    print("🎯 CHECKPOINT 165 BASELINE ACCURACY TEST")
    print("=" * 50)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load preprocessing manifest
    manifest_path = "preprocessed_test_set_24925/preprocessing_manifest.json"
    if not os.path.exists(manifest_path):
        print(f"❌ ERROR: Preprocessing manifest not found at {manifest_path}")
        return
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    print(f"📊 Test Set Overview:")
    print(f"   • Total videos: {manifest['total_videos']}")
    print(f"   • Successfully preprocessed: {manifest['successful']}")
    print(f"   • Failed: {manifest['failed']}")
    print()
    
    # Load checkpoint 165 model
    print("🔄 Loading Checkpoint 165 Model...")
    try:
        model, class_to_idx, idx_to_class, checkpoint = load_enhanced_checkpoint()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        print(f"✅ Model loaded successfully on {device}")
        print(f"   • Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   • Classes: {list(class_to_idx.keys())}")
        print()
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        return
    
    # Test configuration
    CONFIDENCE_THRESHOLD = 0.5  # Enhanced Reliability Gate V2.0
    ENABLE_TTA = True  # Test-Time Augmentation
    TEMPERATURE = 1.0  # Checkpoint 165 setting
    
    print(f"⚙️  Test Configuration:")
    print(f"   • Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"   • TTA Enabled: {ENABLE_TTA}")
    print(f"   • Temperature: {TEMPERATURE}")
    print(f"   • Calibration: DISABLED (uncalibrated baseline)")
    print()
    
    # Initialize results tracking
    results = []
    class_counts = defaultdict(int)
    class_correct = defaultdict(int)
    reliable_predictions = 0
    total_predictions = 0
    
    print("🧪 Testing Videos...")
    print("-" * 50)
    
    # Test each preprocessed video
    for i, video_info in enumerate(manifest['results']):
        if not video_info['success']:
            continue
            
        # Load preprocessed video
        video_path = video_info['preprocessed_path']
        ground_truth = video_info['ground_truth']
        in_4_class = video_info['in_4_class_system']
        
        if not os.path.exists(video_path):
            print(f"⚠️  Skipping {video_path} - file not found")
            continue
        
        # Load and prepare video tensor
        video_data = np.load(video_path)  # Shape: (32, 64, 96)
        video_tensor = torch.FloatTensor(video_data).unsqueeze(0).unsqueeze(0)  # (1, 1, 32, 64, 96)
        
        # Apply Test-Time Augmentation if enabled
        if ENABLE_TTA:
            logits = apply_test_time_augmentation(model, video_tensor, device)
        else:
            with torch.no_grad():
                logits = model(video_tensor.to(device))
        
        # Apply temperature scaling
        logits = logits / TEMPERATURE
        
        # Convert to probabilities
        probabilities = F.softmax(logits, dim=1)
        
        # Apply Enhanced Reliability Gate V2.0
        is_reliable, max_confidence, predicted_class_idx = apply_enhanced_reliability_gate_v2(
            probabilities[0], CONFIDENCE_THRESHOLD
        )
        
        # Get predicted class name
        predicted_class = idx_to_class[predicted_class_idx]
        
        # Determine if prediction is correct (only for 4-class system)
        is_correct = None
        if in_4_class:
            is_correct = (predicted_class == ground_truth)
            class_counts[ground_truth] += 1
            if is_correct:
                class_correct[ground_truth] += 1
        
        # Track reliability
        total_predictions += 1
        if is_reliable:
            reliable_predictions += 1
        
        # Store detailed results
        result = {
            'filename': video_info['original_filename'],
            'ground_truth': ground_truth,
            'predicted_class': predicted_class,
            'max_confidence': max_confidence,
            'is_reliable': is_reliable,
            'in_4_class_system': in_4_class,
            'is_correct': is_correct,
            'raw_probabilities': probabilities[0].cpu().numpy().tolist(),
            'raw_logits': logits[0].cpu().numpy().tolist()
        }
        results.append(result)
        
        # Print progress
        status_icon = "✅" if (is_correct if in_4_class else True) else "❌"
        reliability_icon = "🔒" if is_reliable else "🔓"
        class_system = "4-class" if in_4_class else "OOD"
        
        print(f"{status_icon} {reliability_icon} [{i+1:2d}/18] {video_info['original_filename'][:40]:<40} "
              f"GT: {ground_truth:<15} Pred: {predicted_class:<15} "
              f"Conf: {max_confidence:.3f} ({class_system})")
    
    print()
    print("📊 COMPREHENSIVE BASELINE RESULTS")
    print("=" * 50)
    
    # Calculate overall metrics for 4-class system
    four_class_results = [r for r in results if r['in_4_class_system']]
    four_class_correct = sum(1 for r in four_class_results if r['is_correct'])
    four_class_total = len(four_class_results)
    
    overall_accuracy = (four_class_correct / four_class_total * 100) if four_class_total > 0 else 0
    reliability_rate = (reliable_predictions / total_predictions * 100) if total_predictions > 0 else 0
    
    print(f"🎯 OVERALL PERFORMANCE:")
    print(f"   • 4-Class Accuracy: {four_class_correct}/{four_class_total} = {overall_accuracy:.2f}%")
    print(f"   • Reliability Gate Pass Rate: {reliable_predictions}/{total_predictions} = {reliability_rate:.2f}%")
    print()
    
    # Per-class breakdown
    print(f"📈 PER-CLASS PERFORMANCE (4-Class System):")
    for class_name in sorted(class_counts.keys()):
        correct = class_correct[class_name]
        total = class_counts[class_name]
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"   • {class_name:<15}: {correct:2d}/{total:2d} = {accuracy:5.1f}%")
    print()
    
    # Confidence analysis
    correct_confidences = [r['max_confidence'] for r in four_class_results if r['is_correct']]
    incorrect_confidences = [r['max_confidence'] for r in four_class_results if not r['is_correct']]
    
    avg_correct_conf = np.mean(correct_confidences) if correct_confidences else 0
    avg_incorrect_conf = np.mean(incorrect_confidences) if incorrect_confidences else 0
    
    print(f"🎯 CONFIDENCE ANALYSIS:")
    print(f"   • Average confidence (correct): {avg_correct_conf:.3f}")
    print(f"   • Average confidence (incorrect): {avg_incorrect_conf:.3f}")
    print(f"   • Confidence separation: {avg_correct_conf - avg_incorrect_conf:.3f}")
    print()
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"checkpoint_165_baseline_results_{timestamp}.json"
    
    summary_results = {
        'timestamp': datetime.now().isoformat(),
        'model_info': {
            'checkpoint': 'checkpoint_165',
            'validation_accuracy': '81.65%',
            'parameters': sum(p.numel() for p in model.parameters()),
            'classes': list(class_to_idx.keys())
        },
        'test_configuration': {
            'confidence_threshold': CONFIDENCE_THRESHOLD,
            'tta_enabled': ENABLE_TTA,
            'temperature': TEMPERATURE,
            'calibration': 'DISABLED'
        },
        'overall_metrics': {
            'four_class_accuracy': overall_accuracy,
            'reliability_rate': reliability_rate,
            'total_videos_tested': total_predictions,
            'four_class_videos': four_class_total
        },
        'per_class_performance': {
            class_name: {
                'correct': class_correct[class_name],
                'total': class_counts[class_name],
                'accuracy': (class_correct[class_name] / class_counts[class_name] * 100) if class_counts[class_name] > 0 else 0
            }
            for class_name in sorted(class_counts.keys())
        },
        'confidence_analysis': {
            'avg_correct_confidence': avg_correct_conf,
            'avg_incorrect_confidence': avg_incorrect_conf,
            'confidence_separation': avg_correct_conf - avg_incorrect_conf
        },
        'detailed_results': results
    }
    
    with open(results_file, 'w') as f:
        json.dump(summary_results, f, indent=2)
    
    print(f"💾 Detailed results saved to: {results_file}")
    print()
    print(f"✅ Baseline testing complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return summary_results

if __name__ == "__main__":
    test_checkpoint_165_baseline()
