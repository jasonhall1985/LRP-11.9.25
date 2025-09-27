#!/usr/bin/env python3
"""
Execute Dual-Track Evaluation
=============================

Master execution script for the complete dual-track lip-reading evaluation system.
This script orchestrates both honest cross-speaker generalization validation and
practical bedside personalization capabilities.

Features:
- Automated LOSO cross-validation across all 6 speakers
- Few-shot personalization with K=10 and K=20 shots
- Cross-adaptation validation to detect overfitting
- Comprehensive dual-track reporting
- Scientific integrity with practical deployment readiness

Author: Augment Agent
Date: 2025-09-27
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import time

def run_command(command, description, timeout=3600):
    """Run a command with timeout and error handling."""
    print(f"\n🚀 {description}")
    print(f"Command: {command}")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully in {elapsed_time:.1f}s")
            if result.stdout:
                print("Output:")
                print(result.stdout[-1000:])  # Show last 1000 chars
        else:
            print(f"❌ {description} failed with return code {result.returncode}")
            if result.stderr:
                print("Error:")
                print(result.stderr[-1000:])  # Show last 1000 chars
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"💥 {description} failed with exception: {e}")
        return False
    
    return True

def check_prerequisites():
    """Check if all required files and directories exist."""
    print("🔍 CHECKING PREREQUISITES")
    print("-" * 30)
    
    required_files = [
        "loso_cross_validation_framework.py",
        "calibrate.py", 
        "dual_track_evaluation.py",
        "advanced_training_components.py"
    ]
    
    required_dirs = [
        "data/speaker sets"
    ]
    
    missing_items = []
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_items.append(f"File: {file_path}")
        else:
            print(f"✅ {file_path}")
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_items.append(f"Directory: {dir_path}")
        else:
            print(f"✅ {dir_path}")
    
    if missing_items:
        print(f"\n❌ Missing prerequisites:")
        for item in missing_items:
            print(f"   {item}")
        return False
    
    print(f"\n✅ All prerequisites satisfied")
    return True

def create_base_model_if_needed(output_dir):
    """Create a base model if none exists."""
    base_model_path = Path(output_dir) / "base_model.pth"
    
    if base_model_path.exists():
        print(f"✅ Base model found: {base_model_path}")
        return str(base_model_path)
    
    print("🏗️  Creating base model from current training...")
    
    # Check if we have a current training checkpoint
    current_training_dir = "comprehensive_training_20250926_144725"
    if Path(current_training_dir).exists():
        best_model = Path(current_training_dir) / "best_model.pth"
        if best_model.exists():
            print(f"📋 Using existing best model: {best_model}")
            
            # Copy to base model location
            import shutil
            shutil.copy2(best_model, base_model_path)
            return str(base_model_path)
    
    # If no existing model, train a quick base model using LOSO
    print("🎯 Training new base model using LOSO framework...")
    
    command = f"python3 loso_cross_validation_framework.py --max-epochs 10 --output-dir {output_dir}/base_training"
    success = run_command(command, "Training base model", timeout=7200)  # 2 hours
    
    if success:
        # Find the best model from LOSO training
        base_training_dir = Path(output_dir) / "base_training"
        for fold_dir in base_training_dir.glob("fold_*"):
            fold_model = fold_dir / "best_model.pth"
            if fold_model.exists():
                import shutil
                shutil.copy2(fold_model, base_model_path)
                print(f"✅ Base model created: {base_model_path}")
                return str(base_model_path)
    
    print("❌ Failed to create base model")
    return None

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Execute Dual-Track Evaluation')
    parser.add_argument('--output-dir', default='dual_track_evaluation_results',
                       help='Output directory for all results')
    parser.add_argument('--base-model', help='Path to base model checkpoint (optional)')
    parser.add_argument('--loso-epochs', type=int, default=15,
                       help='Epochs for LOSO training')
    parser.add_argument('--skip-loso', action='store_true',
                       help='Skip LOSO evaluation (use existing results)')
    parser.add_argument('--skip-personalization', action='store_true',
                       help='Skip personalization evaluation')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with minimal epochs')
    
    args = parser.parse_args()
    
    # Adjust parameters for quick test
    if args.quick_test:
        args.loso_epochs = 2
        print("⚡ QUICK TEST MODE - Using minimal epochs")
    
    print("🎯 DUAL-TRACK EVALUATION EXECUTION")
    print("=" * 60)
    print("Comprehensive evaluation of lip-reading model with:")
    print("• Track 1: LOSO Cross-Validation (Honest Generalization)")
    print("• Track 2: Few-Shot Personalization (Bedside Calibration)")
    print("• Cross-Adaptation Validation (Overfitting Detection)")
    print("• Scientific Dual-Track Reporting")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites not met. Please ensure all required files exist.")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle base model
    base_model_path = args.base_model
    if not base_model_path:
        base_model_path = create_base_model_if_needed(output_dir)
        if not base_model_path:
            print("❌ Could not create or find base model")
            return 1
    
    # Execution log
    execution_log = {
        'start_time': datetime.now().isoformat(),
        'parameters': vars(args),
        'base_model': base_model_path,
        'steps': []
    }
    
    success_count = 0
    total_steps = 0
    
    # Step 1: LOSO Cross-Validation
    if not args.skip_loso:
        total_steps += 1
        print(f"\n📊 STEP 1: LOSO CROSS-VALIDATION")
        print("=" * 40)
        
        loso_command = (f"python3 loso_cross_validation_framework.py "
                       f"--max-epochs {args.loso_epochs} "
                       f"--output-dir {output_dir}/loso_results")
        
        loso_success = run_command(loso_command, "LOSO Cross-Validation", timeout=14400)  # 4 hours
        execution_log['steps'].append({
            'step': 'loso_cross_validation',
            'success': loso_success,
            'command': loso_command
        })
        
        if loso_success:
            success_count += 1
    else:
        print("⏭️  Skipping LOSO evaluation (using existing results)")
    
    # Step 2: Few-Shot Personalization
    if not args.skip_personalization:
        total_steps += 1
        print(f"\n🎯 STEP 2: FEW-SHOT PERSONALIZATION")
        print("=" * 40)
        
        # Test personalization on a few speakers
        test_speakers = ['speaker 1 ', 'speaker 2 ', 'speaker 4']
        personalization_success = True
        
        for speaker in test_speakers:
            for k_shots in [10, 20]:
                personalization_command = (f"python3 calibrate.py "
                                         f"--checkpoint {base_model_path} "
                                         f"--speaker '{speaker}' "
                                         f"--shots-per-class {k_shots} "
                                         f"--epochs 5 "
                                         f"--freeze-encoder "
                                         f"--output-dir {output_dir}/personalization_results")
                
                step_success = run_command(
                    personalization_command, 
                    f"Personalization {speaker} K={k_shots}", 
                    timeout=300  # 5 minutes
                )
                
                if not step_success:
                    personalization_success = False
        
        execution_log['steps'].append({
            'step': 'few_shot_personalization',
            'success': personalization_success,
            'speakers_tested': test_speakers
        })
        
        if personalization_success:
            success_count += 1
    else:
        print("⏭️  Skipping personalization evaluation")
    
    # Step 3: Comprehensive Dual-Track Report
    total_steps += 1
    print(f"\n📋 STEP 3: DUAL-TRACK REPORTING")
    print("=" * 40)
    
    report_command = (f"python3 dual_track_evaluation.py "
                     f"--base-model {base_model_path} "
                     f"--loso-epochs {args.loso_epochs} "
                     f"--k-shots 10 20 "
                     f"--output-dir {output_dir}")
    
    report_success = run_command(report_command, "Dual-Track Reporting", timeout=1800)  # 30 minutes
    execution_log['steps'].append({
        'step': 'dual_track_reporting',
        'success': report_success,
        'command': report_command
    })
    
    if report_success:
        success_count += 1
    
    # Final summary
    execution_log['end_time'] = datetime.now().isoformat()
    execution_log['success_rate'] = f"{success_count}/{total_steps}"
    
    # Save execution log
    log_path = output_dir / 'execution_log.json'
    with open(log_path, 'w') as f:
        json.dump(execution_log, f, indent=2)
    
    print(f"\n🎯 DUAL-TRACK EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Success Rate: {success_count}/{total_steps} steps completed")
    print(f"Output Directory: {output_dir}")
    print(f"Execution Log: {log_path}")
    
    if success_count == total_steps:
        print("✅ All steps completed successfully!")
        print(f"\n📊 RESULTS AVAILABLE:")
        print(f"  • LOSO Results: {output_dir}/loso_results/")
        print(f"  • Personalization: {output_dir}/personalization_results/")
        print(f"  • Dual-Track Report: {output_dir}/dual_track_evaluation_report.png")
        return 0
    else:
        print(f"⚠️  {total_steps - success_count} steps failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
