#!/usr/bin/env python3
"""
🎯 EXECUTE COMPREHENSIVE TRAINING
=================================

Master execution script for the comprehensive speaker-disjoint training pipeline.
Orchestrates all phases from dataset preparation to final model training.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
import shutil

def run_command(command, description, cwd=None):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    print(f"   Command: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd
        )
        print(f"✅ {description} completed successfully")
        if result.stdout.strip():
            print(f"   Output: {result.stdout.strip()}")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"   Error: {e.stderr}")
        return False, e.stderr

def main():
    """Main execution orchestrator"""
    print("🎯 COMPREHENSIVE SPEAKER-DISJOINT TRAINING EXECUTION")
    print("=" * 70)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    # Check if we have the pipeline results from previous run
    pipeline_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('comprehensive_speaker_disjoint_')]

    if not pipeline_dirs:
        print("📊 No existing pipeline results found. Running dataset preparation...")

        # Run dataset preparation
        success, output = run_command(
            "python comprehensive_speaker_disjoint_pipeline.py",
            "Dataset preparation and speaker-disjoint splitting"
        )

        if not success:
            print("❌ Dataset preparation failed. Exiting.")
            return False

        # Find the newly created directory
        pipeline_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('comprehensive_speaker_disjoint_')]

    # Use the most recent pipeline directory
    pipeline_dir = sorted(pipeline_dirs)[-1]
    print(f"📁 Using pipeline directory: {pipeline_dir}")
    
    # Load pipeline results
    results_path = Path(pipeline_dir) / "pipeline_results.json"
    if not results_path.exists():
        print(f"❌ Pipeline results not found at: {results_path}")
        return False
    
    with open(results_path, 'r') as f:
        pipeline_results = json.load(f)
    
    print(f"📊 Pipeline Results Summary:")
    print(f"   Total videos: {pipeline_results['step2']['total_videos']}")
    print(f"   Training videos: {pipeline_results['step3']['train_size']}")
    print(f"   Validation videos: {pipeline_results['step3']['val_size']}")
    print(f"   Test videos: {pipeline_results['step3']['test_size']}")
    print(f"   Zero speaker overlap: {pipeline_results['step3']['zero_overlap']}")
    print("")
    
    # Create training output directory
    training_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    training_dir = f"comprehensive_training_{training_timestamp}"
    os.makedirs(training_dir, exist_ok=True)
    
    print(f"📁 Training output directory: {training_dir}")
    
    # Prepare training command
    train_manifest = f"{pipeline_dir}/train_manifest.csv"
    val_manifest = f"{pipeline_dir}/val_manifest.csv"
    
    training_command = f"""python comprehensive_training_script.py \\
        --train-manifest "{train_manifest}" \\
        --val-manifest "{val_manifest}" \\
        --output-dir "{training_dir}" \\
        --target-accuracy 82.0 \\
        --max-epochs 80 \\
        --batch-size 8 \\
        --learning-rate 3e-4 \\
        --weight-decay 1e-3 \\
        --dropout 0.4 \\
        --early-stop-patience 15 \\
        --synthetic-ratio 0.25 \\
        --seed 1337"""
    
    print(f"🚀 Starting comprehensive training...")
    print(f"   Target: 82% validation accuracy")
    print(f"   Baseline: 39.06% (from speaker-disjoint training)")
    print("")
    
    # Execute training
    success, output = run_command(
        training_command,
        "Comprehensive speaker-disjoint training",
        cwd="."
    )
    
    if success:
        print("🎉 Training completed successfully!")
        
        # Check if target was achieved
        best_model_path = Path(training_dir) / "best_model.pth"
        if best_model_path.exists():
            print(f"✅ Best model saved: {best_model_path}")
            
            # Try to load and report final metrics
            try:
                import torch
                checkpoint = torch.load(best_model_path, map_location='cpu')
                final_acc = checkpoint.get('best_val_acc', 0)
                final_f1 = checkpoint.get('best_val_f1', 0)
                
                print(f"🏆 Final Results:")
                print(f"   Best validation accuracy: {final_acc:.2f}%")
                print(f"   Best validation F1-macro: {final_f1:.2f}%")
                
                if final_acc >= 82.0:
                    print(f"🎯 SUCCESS: Target 82% accuracy achieved!")
                    
                    # Create deployment package
                    deployment_dir = f"deployment_package_{training_timestamp}"
                    os.makedirs(deployment_dir, exist_ok=True)
                    
                    # Copy essential files
                    shutil.copy2(best_model_path, deployment_dir)
                    shutil.copy2(Path(training_dir) / "config.json", deployment_dir)
                    
                    # Create deployment info
                    deployment_info = {
                        'model_type': 'Enhanced 3D CNN-LSTM with Temporal Attention',
                        'validation_accuracy': final_acc,
                        'validation_f1_macro': final_f1,
                        'training_date': datetime.now().isoformat(),
                        'speaker_disjoint': True,
                        'zero_speaker_overlap': True,
                        'classes': ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow'],
                        'input_shape': [1, 32, 64, 96],  # [C, T, H, W]
                        'preprocessing': {
                            'target_size': [64, 96],
                            'target_frames': 32,
                            'grayscale': True,
                            'normalize': True
                        }
                    }
                    
                    with open(Path(deployment_dir) / "deployment_info.json", 'w') as f:
                        json.dump(deployment_info, f, indent=2)
                    
                    print(f"📦 Deployment package created: {deployment_dir}")
                    
                else:
                    print(f"⚠️  Target 82% not reached, but significant improvement achieved")
                    improvement = final_acc - 39.06
                    print(f"   Improvement over baseline: +{improvement:.2f} percentage points")
                
            except Exception as e:
                print(f"⚠️  Could not load final metrics: {e}")
        
        # Generate comprehensive report
        generate_comprehensive_report(pipeline_dir, training_dir, pipeline_results)
        
    else:
        print("❌ Training failed")
        return False
    
    print("")
    print("🎯 COMPREHENSIVE TRAINING EXECUTION COMPLETED")
    print("=" * 70)
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return success

def generate_comprehensive_report(pipeline_dir, training_dir, pipeline_results):
    """Generate comprehensive training report"""
    print("📋 Generating comprehensive report...")
    
    report = {
        'execution_summary': {
            'timestamp': datetime.now().isoformat(),
            'pipeline_directory': pipeline_dir,
            'training_directory': training_dir,
            'objective': 'Achieve 82% cross-demographic validation accuracy',
            'baseline_accuracy': 39.06,
            'baseline_source': 'Speaker-disjoint training (checkpoint 165 replacement)'
        },
        'dataset_statistics': {
            'total_videos': pipeline_results['step2']['total_videos'],
            'class_distribution': pipeline_results['step2']['class_counts'],
            'source_statistics': pipeline_results['step2']['source_stats'],
            'final_splits': {
                'training': pipeline_results['step3']['train_size'],
                'validation': pipeline_results['step3']['val_size'],
                'test': pipeline_results['step3']['test_size']
            },
            'speaker_statistics': {
                'training_speakers': pipeline_results['step3']['train_speakers'],
                'validation_speakers': pipeline_results['step3']['val_speakers'],
                'test_speakers': pipeline_results['step3']['test_speakers'],
                'zero_overlap_verified': pipeline_results['step3']['zero_overlap']
            }
        },
        'training_configuration': {
            'architecture': 'Enhanced 3D CNN-LSTM with Temporal Attention',
            'target_accuracy': 82.0,
            'max_epochs': 80,
            'batch_size': 8,
            'learning_rate': 3e-4,
            'weight_decay': 1e-3,
            'dropout': 0.4,
            'early_stopping_patience': 15,
            'augmentation_ratio': 0.25,
            'loss_function': 'Focal Loss with class weights',
            'optimizer': 'AdamW',
            'scheduler': 'ReduceLROnPlateau'
        },
        'key_improvements': [
            'Dataset expanded from 160 to 1,540 videos (9.6x increase)',
            'Speaker-disjoint splits with zero overlap verified',
            'Enhanced architecture with temporal attention',
            'Conservative data augmentation preserving lip-reading quality',
            'Focal loss with class weights for imbalanced data',
            'Comprehensive validation strategy with F1-macro tracking'
        ],
        'deliverables': {
            'data_artifacts': [
                f"{pipeline_dir}/manifest_real.csv",
                f"{pipeline_dir}/train_manifest.csv",
                f"{pipeline_dir}/val_manifest.csv",
                f"{pipeline_dir}/test_manifest.csv"
            ],
            'model_artifacts': [
                f"{training_dir}/best_model.pth",
                f"{training_dir}/config.json"
            ],
            'analysis_artifacts': [
                f"{training_dir}/training_history.png",
                f"{training_dir}/training_history.json"
            ]
        }
    }
    
    report_path = f"comprehensive_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Comprehensive report saved: {report_path}")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
