#!/usr/bin/env python3
"""
Training Monitor for Phase 2 Lip-Reading Training

This script monitors the training progress and provides real-time updates.
"""

import os
import time
import json
import glob
from datetime import datetime
import matplotlib.pyplot as plt

def find_latest_experiment():
    """Find the most recent training experiment directory."""
    experiment_dirs = glob.glob("training_experiment_*")
    if not experiment_dirs:
        return None
    
    # Sort by creation time
    experiment_dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
    return experiment_dirs[0]

def read_log_file(log_path):
    """Read and parse the training log file."""
    if not os.path.exists(log_path):
        return []
    
    log_entries = []
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if "Epoch" in line and "Train Loss" in line:
                # Parse training progress line
                parts = line.strip().split(" - ")[-1]  # Get the message part
                if "Train Loss:" in parts:
                    log_entries.append(parts)
    except Exception as e:
        print(f"Error reading log: {e}")
    
    return log_entries

def monitor_training():
    """Monitor training progress in real-time."""
    print("🔍 TRAINING MONITOR - Phase 2 Lip-Reading Classifier")
    print("=" * 70)
    
    # Find latest experiment
    experiment_dir = find_latest_experiment()
    if not experiment_dir:
        print("❌ No training experiment found. Make sure training is running.")
        return
    
    print(f"📁 Monitoring experiment: {experiment_dir}")
    log_path = os.path.join(experiment_dir, "training.log")
    
    last_log_size = 0
    last_update = time.time()
    
    while True:
        try:
            # Check if log file exists and has new content
            if os.path.exists(log_path):
                current_size = os.path.getsize(log_path)
                
                if current_size > last_log_size:
                    # Read new log entries
                    log_entries = read_log_file(log_path)
                    
                    # Display latest progress
                    if log_entries:
                        print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - Latest Progress:")
                        for entry in log_entries[-3:]:  # Show last 3 entries
                            print(f"   📊 {entry}")
                    
                    last_log_size = current_size
                    last_update = time.time()
                
                # Check for completion indicators
                if os.path.exists(os.path.join(experiment_dir, "final_results.json")):
                    print("\n🎉 TRAINING COMPLETED!")
                    
                    # Load and display final results
                    try:
                        with open(os.path.join(experiment_dir, "final_results.json"), 'r') as f:
                            results = json.load(f)
                        
                        print(f"📊 FINAL RESULTS:")
                        print(f"   • Test Accuracy: {results.get('test_accuracy', 'N/A'):.2f}%")
                        print(f"   • Best Val Accuracy: {results.get('best_val_accuracy', 'N/A'):.2f}%")
                        print(f"   • Total Epochs: {results.get('total_epochs', 'N/A')}")
                        print(f"   • Training Time: {results.get('training_time_minutes', 'N/A'):.1f} minutes")
                        
                    except Exception as e:
                        print(f"Error reading final results: {e}")
                    
                    break
                
                # Check if training seems stuck (no updates for 10 minutes)
                if time.time() - last_update > 600:
                    print(f"\n⚠️  Warning: No log updates for {(time.time() - last_update)/60:.1f} minutes")
                    print("   Training may be stuck or completed without final results.")
            
            else:
                print(f"⏳ Waiting for log file: {log_path}")
            
            # Wait before next check
            time.sleep(30)  # Check every 30 seconds
            
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"❌ Error during monitoring: {e}")
            time.sleep(30)

def display_training_summary():
    """Display a summary of all training experiments."""
    print("\n📈 TRAINING EXPERIMENTS SUMMARY")
    print("=" * 50)
    
    experiment_dirs = glob.glob("training_experiment_*")
    if not experiment_dirs:
        print("No training experiments found.")
        return
    
    experiment_dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
    
    for i, exp_dir in enumerate(experiment_dirs[:5]):  # Show last 5 experiments
        print(f"\n{i+1}. {exp_dir}")
        
        # Check for final results
        results_path = os.path.join(exp_dir, "final_results.json")
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    results = json.load(f)
                
                print(f"   ✅ Status: Completed")
                print(f"   📊 Test Accuracy: {results.get('test_accuracy', 'N/A'):.2f}%")
                print(f"   🎯 Best Val Accuracy: {results.get('best_val_accuracy', 'N/A'):.2f}%")
                print(f"   ⏱️  Training Time: {results.get('training_time_minutes', 'N/A'):.1f} min")
                
            except Exception as e:
                print(f"   ❌ Error reading results: {e}")
        else:
            # Check if still running
            log_path = os.path.join(exp_dir, "training.log")
            if os.path.exists(log_path):
                mod_time = os.path.getmtime(log_path)
                if time.time() - mod_time < 300:  # Modified within 5 minutes
                    print(f"   🔄 Status: Running")
                else:
                    print(f"   ⏸️  Status: Stopped/Incomplete")
            else:
                print(f"   ❓ Status: Unknown")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        display_training_summary()
    else:
        monitor_training()
