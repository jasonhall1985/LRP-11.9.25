#!/usr/bin/env python3
"""
Three-Stage Training Pipeline Executor
======================================

Master execution script for the complete three-stage training pipeline:
1. GRID Corpus Pretraining - Learn visual features from viseme-matched words
2. ICU Fine-tuning with LOSO - Speaker-disjoint validation for honest metrics  
3. Few-shot Personalization - Rapid adaptation for bedside deployment

This script orchestrates the entire pipeline with proper dependency checking,
automated execution, and comprehensive reporting.

Author: Augment Agent
Date: 2025-09-27
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import argparse
import time

class ThreeStageExecutor:
    """Orchestrates the complete three-stage training pipeline."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.start_time = datetime.now()
        self.execution_log = []
        
        # Pipeline stages
        self.stages = {
            1: "GRID Pretraining",
            2: "ICU Fine-tuning (LOSO)",
            3: "Few-shot Personalization"
        }
        
        # Required files and directories
        self.required_files = [
            "utils/viseme_mapper.py",
            "tools/build_grid_manifest.py", 
            "tools/select_grid_subset.py",
            "train_grid_pretrain.py",
            "train_icu_finetune.py",
            "calibrate.py",
            "advanced_training_components.py",
            "loso_cross_validation_framework.py"
        ]
        
        self.required_dirs = [
            "data/speaker sets",
            "tools",
            "utils",
            "manifests",
            "checkpoints"
        ]
    
    def log_step(self, stage: int, step: str, status: str, details: str = ""):
        """Log execution step."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'stage': stage,
            'step': step,
            'status': status,
            'details': details
        }
        self.execution_log.append(log_entry)
        
        status_emoji = "✅" if status == "SUCCESS" else "❌" if status == "ERROR" else "🔄"
        print(f"{status_emoji} [{timestamp}] Stage {stage} - {step}: {status}")
        if details:
            print(f"    {details}")
    
    def check_prerequisites(self) -> bool:
        """Check all prerequisites for the pipeline."""
        print("🔍 CHECKING PREREQUISITES")
        print("=" * 40)
        
        all_good = True
        
        # Check required files
        print("📄 Required files:")
        for file_path in self.required_files:
            if Path(file_path).exists():
                print(f"  ✅ {file_path}")
            else:
                print(f"  ❌ {file_path}")
                all_good = False
        
        # Check required directories
        print("\n📁 Required directories:")
        for dir_path in self.required_dirs:
            if Path(dir_path).exists():
                print(f"  ✅ {dir_path}")
            else:
                print(f"  ❌ {dir_path}")
                all_good = False
        
        # Check speaker sets
        speaker_sets_dir = Path("data/speaker sets")
        if speaker_sets_dir.exists():
            speakers = [d for d in speaker_sets_dir.iterdir() if d.is_dir() and d.name.startswith('speaker')]
            print(f"  📊 Found {len(speakers)} speaker sets")
            if len(speakers) < 3:
                print(f"  ⚠️  Warning: Only {len(speakers)} speakers found, recommend ≥6 for robust LOSO")
        
        # Check Python packages
        print("\n📦 Python packages:")
        required_packages = ['torch', 'pandas', 'numpy', 'sklearn', 'cv2', 'tqdm']
        for package in required_packages:
            try:
                __import__(package)
                print(f"  ✅ {package}")
            except ImportError:
                print(f"  ❌ {package}")
                all_good = False
        
        return all_good
    
    def run_command(self, command: List[str], stage: int, step: str, 
                   timeout: Optional[int] = None) -> bool:
        """Run a command and log the result."""
        self.log_step(stage, step, "RUNNING", f"Command: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path.cwd()
            )
            
            if result.returncode == 0:
                self.log_step(stage, step, "SUCCESS", f"Completed successfully")
                return True
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                self.log_step(stage, step, "ERROR", f"Exit code {result.returncode}: {error_msg}")
                return False
                
        except subprocess.TimeoutExpired:
            self.log_step(stage, step, "ERROR", f"Command timed out after {timeout} seconds")
            return False
        except Exception as e:
            self.log_step(stage, step, "ERROR", f"Exception: {str(e)}")
            return False
    
    def stage_1_grid_pretraining(self) -> bool:
        """Execute Stage 1: GRID Pretraining."""
        print(f"\n🎯 STAGE 1: {self.stages[1]}")
        print("=" * 50)
        
        # Step 1: Build GRID manifest (if GRID data available)
        grid_dir = Path(self.config.get('grid_dir', 'data/grid'))
        if grid_dir.exists():
            print("📋 Building GRID corpus manifest...")
            if not self.run_command([
                'python3', 'tools/build_grid_manifest.py',
                '--grid-dir', str(grid_dir),
                '--output-dir', 'manifests'
            ], 1, "Build GRID Manifest", timeout=300):
                return False
            
            # Step 2: Select GRID subset based on viseme similarity
            print("🎯 Selecting GRID subset for pretraining...")
            if not self.run_command([
                'python3', 'tools/select_grid_subset.py',
                '--manifest-dir', 'manifests',
                '--output-dir', 'manifests/pretraining',
                '--words-per-class', str(self.config.get('words_per_class', 20))
            ], 1, "Select GRID Subset", timeout=300):
                return False
            
            # Step 3: GRID pretraining
            print("🚀 Starting GRID pretraining...")
            grid_output_dir = f"{self.config.get('output_dir', 'runs/pipeline_output')}/checkpoints/grid_pretraining"
            if not self.run_command([
                'python3', 'train_grid_pretrain.py',
                '--manifest-path', 'manifests/pretraining/grid_pretraining_manifest.csv',
                '--output-dir', grid_output_dir,
                '--max-epochs', str(self.config.get('grid_epochs', 30)),
                '--batch-size', str(self.config.get('batch_size', 16)),
                '--learning-rate', str(self.config.get('grid_lr', 0.001))
            ], 1, "GRID Pretraining", timeout=7200):  # 2 hours timeout
                return False
        else:
            print("⚠️  GRID directory not found, skipping GRID pretraining")
            self.log_step(1, "GRID Pretraining", "SKIPPED", "GRID data not available")
        
        return True
    
    def stage_2_icu_finetuning(self) -> bool:
        """Execute Stage 2: ICU Fine-tuning with LOSO."""
        print(f"\n🎯 STAGE 2: {self.stages[2]}")
        print("=" * 50)
        
        # Check for pretrained encoder
        base_output_dir = self.config.get('output_dir', 'runs/pipeline_output')
        pretrained_path = Path(f'{base_output_dir}/checkpoints/grid_pretraining/grid_pretrain_best.pth')
        icu_output_dir = f"{base_output_dir}/checkpoints/icu_finetuning"

        command = [
            'python3', 'train_icu_finetune.py',
            '--data-dir', 'data/speaker sets',
            '--output-dir', icu_output_dir,
            '--max-epochs', str(self.config.get('icu_epochs', 20)),
            '--batch-size', str(self.config.get('batch_size', 8)),
            '--learning-rate', str(self.config.get('icu_lr', 0.0005))
        ]

        if pretrained_path.exists():
            command.extend(['--pretrained-encoder', str(pretrained_path)])
            print("🔗 Using GRID pretrained encoder")
        else:
            command.append('--no-pretrained')
            print("⚠️  No pretrained encoder found, training from scratch")
        
        print("🚀 Starting ICU fine-tuning with LOSO validation...")
        return self.run_command(command, 2, "ICU Fine-tuning (LOSO)", timeout=10800)  # 3 hours timeout
    
    def stage_3_personalization(self) -> bool:
        """Execute Stage 3: Few-shot Personalization."""
        print(f"\n🎯 STAGE 3: {self.stages[3]}")
        print("=" * 50)
        
        # Find best ICU model from LOSO training
        base_output_dir = self.config.get('output_dir', 'runs/pipeline_output')
        icu_checkpoint_dir = Path(f'{base_output_dir}/checkpoints/icu_finetuning')
        if not icu_checkpoint_dir.exists():
            self.log_step(3, "Find ICU Model", "ERROR", "ICU checkpoint directory not found")
            return False

        # Look for best fold checkpoint
        best_checkpoints = list(icu_checkpoint_dir.glob('icu_finetune_fold_*_best.pth'))
        if not best_checkpoints:
            self.log_step(3, "Find ICU Model", "ERROR", "No ICU checkpoints found")
            return False

        # Use first available checkpoint (in practice, you might want to select best performing fold)
        base_model_path = best_checkpoints[0]
        personalization_output_dir = f"{base_output_dir}/checkpoints/personalization"

        print(f"🎯 Using base model: {base_model_path}")
        print("🚀 Starting few-shot personalization...")

        return self.run_command([
            'python3', 'calibrate.py',
            '--base-model', str(base_model_path),
            '--data-dir', 'data/speaker sets',
            '--output-dir', personalization_output_dir,
            '--k-shot', str(self.config.get('k_shot', 10)),
            '--max-epochs', str(self.config.get('personalization_epochs', 5)),
            '--learning-rate', str(self.config.get('personalization_lr', 0.0001))
        ], 3, "Few-shot Personalization", timeout=1800)  # 30 minutes timeout
    
    def generate_final_report(self) -> str:
        """Generate comprehensive final report."""
        end_time = datetime.now()
        total_duration = end_time - self.start_time
        
        report = []
        report.append("THREE-STAGE TRAINING PIPELINE REPORT")
        report.append("=" * 60)
        report.append(f"Execution Date: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Duration: {total_duration}")
        report.append("")
        
        # Configuration
        report.append("CONFIGURATION:")
        for key, value in self.config.items():
            report.append(f"  {key}: {value}")
        report.append("")
        
        # Execution log
        report.append("EXECUTION LOG:")
        for entry in self.execution_log:
            report.append(f"[{entry['timestamp']}] Stage {entry['stage']} - {entry['step']}: {entry['status']}")
            if entry['details']:
                report.append(f"    {entry['details']}")
        report.append("")
        
        # Stage summary
        report.append("STAGE SUMMARY:")
        for stage_num, stage_name in self.stages.items():
            stage_entries = [e for e in self.execution_log if e['stage'] == stage_num]
            success_count = len([e for e in stage_entries if e['status'] == 'SUCCESS'])
            error_count = len([e for e in stage_entries if e['status'] == 'ERROR'])
            
            status = "✅ COMPLETED" if error_count == 0 and success_count > 0 else "❌ FAILED" if error_count > 0 else "⏭️ SKIPPED"
            report.append(f"  Stage {stage_num} ({stage_name}): {status}")
        
        report.append("")
        report.append("NEXT STEPS:")
        report.append("1. Review individual stage logs in respective output directories")
        report.append("2. Evaluate model performance using validation metrics")
        report.append("3. Deploy personalized models for bedside use")
        report.append("4. Monitor real-world performance and collect feedback")
        
        return "\n".join(report)
    
    def execute_pipeline(self) -> bool:
        """Execute the complete three-stage pipeline."""
        print("🚀 THREE-STAGE TRAINING PIPELINE EXECUTION")
        print("=" * 60)
        print("Pipeline Overview:")
        for stage_num, stage_name in self.stages.items():
            print(f"  Stage {stage_num}: {stage_name}")
        print("=" * 60)
        
        # Check prerequisites
        if not self.check_prerequisites():
            print("\n❌ Prerequisites not met. Please resolve issues before continuing.")
            return False
        
        print("\n✅ Prerequisites satisfied. Starting pipeline execution...")
        
        # Execute stages
        success = True
        
        if self.config.get('skip_grid', False):
            print("\n⏭️  Skipping Stage 1 (GRID Pretraining) as requested")
            self.log_step(1, "GRID Pretraining", "SKIPPED", "User requested skip")
        else:
            success = success and self.stage_1_grid_pretraining()
        
        if success and not self.config.get('skip_icu', False):
            success = success and self.stage_2_icu_finetuning()
        elif self.config.get('skip_icu', False):
            print("\n⏭️  Skipping Stage 2 (ICU Fine-tuning) as requested")
            self.log_step(2, "ICU Fine-tuning", "SKIPPED", "User requested skip")
        
        if success and not self.config.get('skip_personalization', False):
            success = success and self.stage_3_personalization()
        elif self.config.get('skip_personalization', False):
            print("\n⏭️  Skipping Stage 3 (Personalization) as requested")
            self.log_step(3, "Personalization", "SKIPPED", "User requested skip")
        
        # Generate final report
        report = self.generate_final_report()
        
        # Save report
        report_path = Path('pipeline_execution_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n📊 PIPELINE EXECUTION COMPLETE")
        print("=" * 40)
        if success:
            print("✅ All stages completed successfully!")
        else:
            print("❌ Pipeline completed with errors. Check logs for details.")
        
        print(f"📋 Full report saved to: {report_path}")
        print("\nSUMMARY:")
        print(report.split("STAGE SUMMARY:")[1].split("NEXT STEPS:")[0])
        
        return success

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Execute Three-Stage Training Pipeline')
    parser.add_argument('--grid-dir', default='data/grid',
                       help='Path to GRID corpus directory')
    parser.add_argument('--grid-epochs', type=int, default=30,
                       help='Epochs for GRID pretraining')
    parser.add_argument('--icu-epochs', type=int, default=20,
                       help='Epochs for ICU fine-tuning')
    parser.add_argument('--personalization-epochs', type=int, default=5,
                       help='Epochs for personalization')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--words-per-class', type=int, default=20,
                       help='GRID words per ICU class')
    parser.add_argument('--k-shot', type=int, default=10,
                       help='K-shot examples for personalization')
    parser.add_argument('--output-dir', default='runs/pipeline_output',
                       help='Base output directory for all stages')
    parser.add_argument('--skip-grid', action='store_true',
                       help='Skip GRID pretraining stage')
    parser.add_argument('--skip-icu', action='store_true',
                       help='Skip ICU fine-tuning stage')
    parser.add_argument('--skip-personalization', action='store_true',
                       help='Skip personalization stage')
    
    args = parser.parse_args()
    
    # Build configuration
    config = {
        'grid_dir': args.grid_dir,
        'grid_epochs': args.grid_epochs,
        'icu_epochs': args.icu_epochs,
        'personalization_epochs': args.personalization_epochs,
        'batch_size': args.batch_size,
        'words_per_class': args.words_per_class,
        'k_shot': args.k_shot,
        'output_dir': args.output_dir,
        'grid_lr': 0.001,
        'icu_lr': 0.0005,
        'personalization_lr': 0.0001,
        'skip_grid': args.skip_grid,
        'skip_icu': args.skip_icu,
        'skip_personalization': args.skip_personalization
    }
    
    # Execute pipeline
    executor = ThreeStageExecutor(config)
    success = executor.execute_pipeline()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
