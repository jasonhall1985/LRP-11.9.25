#!/usr/bin/env python3
"""
Enhanced Training Monitor - Real-time Progress Tracking
"""

import os
import time
import json
import glob
from datetime import datetime

def find_latest_enhanced_experiment():
    """Find the most recent enhanced training experiment."""
    experiment_dirs = glob.glob("enhanced_training_*")
    if not experiment_dirs:
        return None
    
    experiment_dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
    return experiment_dirs[0]

def parse_enhanced_log(log_path):
    """Parse enhanced training log for key metrics."""
    if not os.path.exists(log_path):
        return {}
    
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
        
        current_stage = 1
        current_epoch = 0
        latest_metrics = {}
        
        for line in lines:
            line = line.strip()
            
            # Track stage changes
            if "Stage 1:" in line:
                current_stage = 1
            elif "Stage 2:" in line:
                current_stage = 2
            elif "Stage 3:" in line:
                current_stage = 3
            
            # Parse epoch results
            if "Train Loss:" in line and "Val Acc:" in line:
                try:
                    parts = line.split(" - ")[-1]
                    
                    # Extract metrics
                    metrics = {}
                    for part in parts.split(", "):
                        if ":" in part:
                            key, value = part.split(": ")
                            if "%" in value:
                                metrics[key] = float(value.replace("%", ""))
                            else:
                                metrics[key] = float(value)
                    
                    # Extract epoch number
                    if "Epoch" in line:
                        epoch_part = line.split("Epoch ")[1].split(" ")[0]
                        current_epoch = int(epoch_part)
                    
                    latest_metrics = {
                        'stage': current_stage,
                        'epoch': current_epoch,
                        **metrics
                    }
                    
                except Exception as e:
                    continue
        
        return latest_metrics
        
    except Exception as e:
        print(f"Error parsing log: {e}")
        return {}

def display_progress_bar(current, target, width=30):
    """Display a progress bar."""
    if target == 0:
        return "[" + "=" * width + "]"
    
    filled = int(width * current / target)
    bar = "=" * filled + ">" + "-" * (width - filled - 1)
    return f"[{bar}] {current:.1f}%/{target}%"

def monitor_enhanced_training():
    """Monitor enhanced training with detailed progress."""
    print("🔍 ENHANCED TRAINING MONITOR")
    print("=" * 60)
    
    experiment_dir = find_latest_enhanced_experiment()
    if not experiment_dir:
        print("❌ No enhanced training experiment found.")
        return
    
    print(f"📁 Monitoring: {experiment_dir}")
    log_path = os.path.join(experiment_dir, "training.log")
    results_path = os.path.join(experiment_dir, "final_results.json")
    
    last_update = time.time()
    last_metrics = {}
    
    print(f"📊 Target: 60-75% accuracy (vs 40% baseline)")
    print(f"🎯 Staged Training: 3 stages with early stopping")
    print("-" * 60)
    
    while True:
        try:
            # Check for completion
            if os.path.exists(results_path):
                print("\n🎉 TRAINING COMPLETED!")
                
                try:
                    with open(results_path, 'r') as f:
                        results = json.load(f)
                    
                    test_acc = results.get('test_accuracy', 0)
                    ema_test_acc = results.get('ema_test_accuracy', 0)
                    best_val_acc = results.get('best_val_accuracy', 0)
                    test_f1 = results.get('test_f1_score', 0)
                    
                    final_acc = max(test_acc, ema_test_acc)
                    
                    print(f"\n📊 FINAL RESULTS:")
                    print(f"   🎯 Test Accuracy: {test_acc:.2f}%")
                    print(f"   🌟 EMA Test Accuracy: {ema_test_acc:.2f}%")
                    print(f"   📈 Best Val Accuracy: {best_val_acc:.2f}%")
                    print(f"   📊 Test F1 Score: {test_f1:.2f}%")
                    print(f"   🏆 Best Result: {final_acc:.2f}%")
                    
                    print(f"\n🎯 SUCCESS EVALUATION:")
                    baseline = 40.0
                    improvement = final_acc - baseline
                    
                    if final_acc >= 75:
                        print(f"   🏆 EXCELLENT: {final_acc:.1f}% (Target exceeded!)")
                    elif final_acc >= 60:
                        print(f"   ✅ SUCCESS: {final_acc:.1f}% (Target achieved!)")
                    elif final_acc >= 50:
                        print(f"   📈 GOOD: {final_acc:.1f}% (Significant improvement)")
                    else:
                        print(f"   ⚠️  NEEDS WORK: {final_acc:.1f}% (Below expectations)")
                    
                    print(f"   📊 Improvement: +{improvement:.1f} percentage points")
                    print(f"   📈 Relative gain: {((final_acc/baseline)-1)*100:.1f}%")
                    
                except Exception as e:
                    print(f"Error reading results: {e}")
                
                break
            
            # Parse current progress
            current_metrics = parse_enhanced_log(log_path)
            
            if current_metrics and current_metrics != last_metrics:
                stage = current_metrics.get('stage', 1)
                epoch = current_metrics.get('epoch', 0)
                train_acc = current_metrics.get('Train Acc', 0)
                val_acc = current_metrics.get('Val Acc', 0)
                train_f1 = current_metrics.get('Train F1', 0)
                val_f1 = current_metrics.get('Val F1', 0)
                ema_val_acc = current_metrics.get('EMA Val Acc', val_acc)
                best_f1 = current_metrics.get('Best F1', val_f1)
                
                print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - Stage {stage}, Epoch {epoch}")
                print(f"   📈 Training: Acc={train_acc:.1f}%, F1={train_f1:.1f}%")
                print(f"   🎯 Validation: Acc={val_acc:.1f}%, F1={val_f1:.1f}%")
                print(f"   🌟 EMA Val: Acc={ema_val_acc:.1f}%")
                print(f"   🏆 Best F1: {best_f1:.1f}%")
                
                # Progress towards target
                target_acc = 60.0
                progress_bar = display_progress_bar(val_acc, target_acc)
                print(f"   🎯 Progress: {progress_bar}")
                
                # Stage-specific info
                if stage == 1:
                    print(f"   🔥 Stage 1: Head-only training (warmup)")
                elif stage == 2:
                    print(f"   🔥 Stage 2: Head + last backbone block")
                elif stage == 3:
                    print(f"   🔥 Stage 3: Full model fine-tuning")
                
                last_metrics = current_metrics
                last_update = time.time()
            
            # Check if training seems stuck
            elif time.time() - last_update > 300:  # 5 minutes
                print(f"\n⚠️  No updates for {(time.time() - last_update)/60:.1f} minutes")
                print("   Training may be processing or stuck...")
            
            time.sleep(15)  # Check every 15 seconds
            
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(15)

if __name__ == "__main__":
    monitor_enhanced_training()
