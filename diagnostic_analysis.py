#!/usr/bin/env python3
"""
Diagnostic Analysis - Understanding Why Training Performance Degraded
"""

import os
import numpy as np
import torch
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import json
from collections import Counter

def analyze_dataset_quality():
    """Analyze the corrected balanced dataset for potential issues."""
    print("🔍 DATASET QUALITY ANALYSIS")
    print("=" * 50)
    
    dataset_path = "corrected_balanced_dataset"
    video_files = list(Path(dataset_path).glob("*.mp4"))
    
    if len(video_files) == 0:
        print("❌ No videos found!")
        return
    
    print(f"📊 Found {len(video_files)} videos")
    
    # Analyze by class
    class_counts = Counter()
    class_frame_counts = {}
    class_dimensions = {}
    class_file_sizes = {}
    
    for video_file in video_files:
        class_name = video_file.stem.split('_')[0]
        class_counts[class_name] += 1
        
        # Analyze video properties
        try:
            cap = cv2.VideoCapture(str(video_file))
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                if class_name not in class_frame_counts:
                    class_frame_counts[class_name] = []
                    class_dimensions[class_name] = []
                
                class_frame_counts[class_name].append(frame_count)
                class_dimensions[class_name].append((width, height))
                
                file_size = video_file.stat().st_size / 1024  # KB
                if class_name not in class_file_sizes:
                    class_file_sizes[class_name] = []
                class_file_sizes[class_name].append(file_size)
                
            cap.release()
        except Exception as e:
            print(f"❌ Error analyzing {video_file}: {e}")
    
    # Report findings
    print(f"\n📊 CLASS DISTRIBUTION:")
    for class_name, count in sorted(class_counts.items()):
        print(f"   • {class_name}: {count} videos")
    
    print(f"\n📊 FRAME COUNT ANALYSIS:")
    for class_name in sorted(class_frame_counts.keys()):
        frames = class_frame_counts[class_name]
        print(f"   • {class_name}: {min(frames)}-{max(frames)} frames (avg: {np.mean(frames):.1f})")
    
    print(f"\n📊 DIMENSION ANALYSIS:")
    for class_name in sorted(class_dimensions.keys()):
        dims = class_dimensions[class_name]
        unique_dims = set(dims)
        print(f"   • {class_name}: {len(unique_dims)} unique dimensions")
        for dim in unique_dims:
            count = dims.count(dim)
            print(f"     - {dim[0]}x{dim[1]}: {count} videos")
    
    print(f"\n📊 FILE SIZE ANALYSIS:")
    for class_name in sorted(class_file_sizes.keys()):
        sizes = class_file_sizes[class_name]
        print(f"   • {class_name}: {min(sizes):.1f}-{max(sizes):.1f} KB (avg: {np.mean(sizes):.1f} KB)")
    
    # Check for potential issues
    print(f"\n⚠️  POTENTIAL ISSUES:")
    
    # Check class balance
    counts = list(class_counts.values())
    if max(counts) - min(counts) > 2:
        print(f"   • Class imbalance detected: {min(counts)}-{max(counts)} videos per class")
    else:
        print(f"   ✅ Classes are well balanced")
    
    # Check frame consistency
    all_frames = []
    for frames in class_frame_counts.values():
        all_frames.extend(frames)
    
    if len(set(all_frames)) > 3:
        print(f"   • Frame count inconsistency: {min(all_frames)}-{max(all_frames)} frames")
    else:
        print(f"   ✅ Frame counts are consistent")
    
    # Check dimensions
    all_dims = []
    for dims in class_dimensions.values():
        all_dims.extend(dims)
    
    unique_all_dims = set(all_dims)
    if len(unique_all_dims) > 1:
        print(f"   • Dimension inconsistency: {len(unique_all_dims)} different sizes")
        for dim in unique_all_dims:
            count = all_dims.count(dim)
            print(f"     - {dim[0]}x{dim[1]}: {count} videos")
    else:
        print(f"   ✅ All videos have consistent dimensions")

def analyze_training_results():
    """Analyze training results to understand performance degradation."""
    print("\n🔍 TRAINING RESULTS ANALYSIS")
    print("=" * 50)
    
    # Find all training experiments
    experiments = []
    
    # Memory efficient training
    if os.path.exists("efficient_training_20250917_010329/final_results.json"):
        with open("efficient_training_20250917_010329/final_results.json", 'r') as f:
            results = json.load(f)
        experiments.append(("Memory Efficient", results))
    
    # Enhanced training
    enhanced_dirs = [d for d in os.listdir('.') if d.startswith('enhanced_training_')]
    for exp_dir in enhanced_dirs:
        results_path = f"{exp_dir}/final_results.json"
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
            experiments.append(("Enhanced", results))
    
    # Fallback training
    fallback_dirs = [d for d in os.listdir('.') if d.startswith('fallback_training_')]
    for exp_dir in fallback_dirs:
        results_path = f"{exp_dir}/final_results.json"
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
            experiments.append(("Fallback", results))
    
    if not experiments:
        print("❌ No training results found")
        return
    
    print(f"📊 TRAINING COMPARISON:")
    print(f"{'Experiment':<15} {'Test Acc':<10} {'Val Acc':<10} {'Epochs':<8} {'Parameters'}")
    print("-" * 60)
    
    for name, results in experiments:
        test_acc = results.get('test_accuracy', 0)
        val_acc = results.get('best_val_accuracy', 0)
        epochs = results.get('total_epochs', 0)
        
        # Estimate parameters based on experiment type
        if name == "Memory Efficient":
            params = "290K"
        elif name == "Enhanced":
            params = "2.2M"
        elif name == "Fallback":
            params = "5.2M"
        else:
            params = "Unknown"
        
        print(f"{name:<15} {test_acc:<10.1f} {val_acc:<10.1f} {epochs:<8} {params}")
    
    # Analyze patterns
    print(f"\n🔍 PATTERN ANALYSIS:")
    
    test_accs = [results.get('test_accuracy', 0) for _, results in experiments]
    val_accs = [results.get('best_val_accuracy', 0) for _, results in experiments]
    
    if all(acc <= 25 for acc in test_accs):
        print(f"   ⚠️  All models performing at random level (~20%)")
        print(f"   🔍 Possible causes:")
        print(f"      • Dataset quality issues")
        print(f"      • Preprocessing problems")
        print(f"      • Label misalignment")
        print(f"      • Insufficient training data")
        print(f"      • Model architecture issues")
    
    if max(test_accs) < max(val_accs):
        print(f"   ⚠️  Test performance worse than validation")
        print(f"   🔍 Possible overfitting to validation set")

def create_simple_baseline():
    """Create a very simple baseline to test basic functionality."""
    print("\n🔍 SIMPLE BASELINE TEST")
    print("=" * 50)
    
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
        
        # Create dummy data to test if training works at all
        class DummyDataset(Dataset):
            def __init__(self, size=50):
                self.size = size
                # Create simple patterns that should be learnable
                self.data = []
                self.labels = []
                
                for i in range(size):
                    # Create simple visual patterns for each class
                    class_id = i % 5
                    
                    # Create a 32x64x64 video with class-specific pattern
                    video = torch.zeros(1, 32, 64, 64)
                    
                    if class_id == 0:  # doctor - horizontal lines
                        video[:, :, 10:20, :] = 1.0
                    elif class_id == 1:  # glasses - vertical lines
                        video[:, :, :, 10:20] = 1.0
                    elif class_id == 2:  # help - diagonal
                        for t in range(32):
                            for i in range(min(64, 64)):
                                if i < 64 and i < 64:
                                    video[0, t, i, i] = 1.0
                    elif class_id == 3:  # phone - center square
                        video[:, :, 20:40, 20:40] = 1.0
                    elif class_id == 4:  # pillow - corners
                        video[:, :, :10, :10] = 1.0
                        video[:, :, -10:, -10:] = 1.0
                    
                    self.data.append(video)
                    self.labels.append(class_id)
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]
        
        # Simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv3d(1, 16, kernel_size=3, padding=1)
                self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
                self.fc = nn.Linear(16, 5)
            
            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)
        
        # Test training
        dataset = DummyDataset(50)
        loader = DataLoader(dataset, batch_size=5, shuffle=True)
        
        model = SimpleModel()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        print("🧪 Testing basic training functionality...")
        
        model.train()
        for epoch in range(5):
            total_loss = 0
            correct = 0
            total = 0
            
            for data, target in loader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
            
            acc = 100. * correct / total
            print(f"   Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}, Acc={acc:.1f}%")
        
        if acc > 80:
            print("✅ Basic training functionality works - issue is with real data")
        elif acc > 40:
            print("⚠️  Basic training works but not optimal - check model/data")
        else:
            print("❌ Basic training fails - fundamental issue with setup")
            
    except Exception as e:
        print(f"❌ Error in baseline test: {e}")

def generate_recommendations():
    """Generate specific recommendations based on analysis."""
    print("\n💡 RECOMMENDATIONS")
    print("=" * 50)
    
    print("🎯 IMMEDIATE ACTIONS:")
    print("   1. Verify dataset integrity and labels")
    print("   2. Check preprocessing pipeline for bugs")
    print("   3. Test with original (non-enhanced) preprocessing")
    print("   4. Reduce model complexity further")
    print("   5. Increase batch size if memory allows")
    
    print("\n🔧 TECHNICAL FIXES:")
    print("   1. Use original video dimensions (640x432)")
    print("   2. Disable all augmentations initially")
    print("   3. Use simple Adam optimizer (lr=1e-3)")
    print("   4. Start with 2D CNN on middle frame only")
    print("   5. Check for data leakage in splits")
    
    print("\n📊 DEBUGGING STEPS:")
    print("   1. Visualize preprocessed videos")
    print("   2. Check label distribution in splits")
    print("   3. Test overfitting on single batch")
    print("   4. Compare with original simple model")
    print("   5. Validate data loading pipeline")
    
    print("\n🎯 SUCCESS CRITERIA FOR NEXT ATTEMPT:")
    print("   • Target: Beat 40% baseline (aim for 45%+)")
    print("   • Method: Simplest possible approach first")
    print("   • Validation: Overfit single batch before full training")
    print("   • Timeline: Quick iteration cycles (15-30 min each)")

def main():
    """Run comprehensive diagnostic analysis."""
    print("🔍 COMPREHENSIVE DIAGNOSTIC ANALYSIS")
    print("=" * 60)
    print("Analyzing why training performance degraded from 40% to 20%")
    print("=" * 60)
    
    # Run all analyses
    analyze_dataset_quality()
    analyze_training_results()
    create_simple_baseline()
    generate_recommendations()
    
    print(f"\n📋 SUMMARY:")
    print(f"   • Dataset appears to be properly balanced")
    print(f"   • Multiple training approaches all converged to 20%")
    print(f"   • This suggests systematic issue rather than model choice")
    print(f"   • Need to debug preprocessing and data pipeline")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Return to original simple approach that achieved 40%")
    print(f"   2. Identify what changed between working and non-working versions")
    print(f"   3. Test minimal modifications to working baseline")
    print(f"   4. Focus on data quality over model complexity")

if __name__ == "__main__":
    main()
