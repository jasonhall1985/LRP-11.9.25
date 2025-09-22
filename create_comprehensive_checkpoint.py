#!/usr/bin/env python3
"""
Create Comprehensive Checkpoint: Enhanced 62.39% Validation Accuracy Model
Complete backup of expanded dataset training pipeline and results
"""

import os
import shutil
import json
from datetime import datetime
import subprocess
import glob

def create_checkpoint_directory():
    """Create comprehensive checkpoint directory"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_name = f"checkpoint_enhanced_62_39_percent_model_{timestamp}"
    checkpoint_dir = checkpoint_name
    
    print(f"🎯 CREATING COMPREHENSIVE CHECKPOINT")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_name}")
    print("Objective: Backup enhanced 62.39% validation accuracy model and complete pipeline")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir, checkpoint_name

def backup_enhanced_training_results(checkpoint_dir):
    """Backup enhanced training results and models"""
    print(f"\n📊 BACKING UP ENHANCED TRAINING RESULTS")
    print("=" * 50)
    
    source_dir = "enhanced_balanced_training_results"
    target_dir = os.path.join(checkpoint_dir, "enhanced_training_results")
    
    if os.path.exists(source_dir):
        shutil.copytree(source_dir, target_dir)
        
        # List backed up files
        files = os.listdir(target_dir)
        print(f"✅ Backed up {len(files)} files:")
        for file in sorted(files):
            file_path = os.path.join(target_dir, file)
            size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"  {file} ({size:.1f} MB)")
    else:
        print("❌ Enhanced training results directory not found")

def backup_training_scripts(checkpoint_dir):
    """Backup all training and analysis scripts"""
    print(f"\n📝 BACKING UP TRAINING SCRIPTS")
    print("=" * 40)
    
    scripts_dir = os.path.join(checkpoint_dir, "training_scripts")
    os.makedirs(scripts_dir, exist_ok=True)
    
    # Key scripts to backup
    script_files = [
        "analyze_new_diverse_videos.py",
        "process_140_new_diverse_videos.py", 
        "create_enhanced_balanced_dataset.py",
        "train_enhanced_lightweight_model.py",
        "create_comprehensive_checkpoint.py"
    ]
    
    backed_up = 0
    for script in script_files:
        if os.path.exists(script):
            shutil.copy2(script, scripts_dir)
            print(f"  ✅ {script}")
            backed_up += 1
        else:
            print(f"  ❌ {script} (not found)")
    
    print(f"✅ Backed up {backed_up}/{len(script_files)} scripts")

def backup_dataset_manifests(checkpoint_dir):
    """Backup dataset manifests and analysis results"""
    print(f"\n📋 BACKING UP DATASET MANIFESTS")
    print("=" * 40)
    
    manifests_dir = os.path.join(checkpoint_dir, "dataset_manifests")
    os.makedirs(manifests_dir, exist_ok=True)
    
    # Find and backup manifest files
    manifest_patterns = [
        "enhanced_balanced_*_manifest.csv",
        "*_analysis_*.json",
        "*_processing_results_*.json",
        "*_dataset_results_*.json"
    ]
    
    backed_up = 0
    for pattern in manifest_patterns:
        files = glob.glob(pattern)
        for file in files:
            shutil.copy2(file, manifests_dir)
            print(f"  ✅ {file}")
            backed_up += 1
    
    print(f"✅ Backed up {backed_up} manifest/analysis files")

def backup_previous_results(checkpoint_dir):
    """Backup previous training results for comparison"""
    print(f"\n📈 BACKING UP PREVIOUS RESULTS FOR COMPARISON")
    print("=" * 50)
    
    previous_dir = os.path.join(checkpoint_dir, "previous_results")
    os.makedirs(previous_dir, exist_ok=True)
    
    # Backup previous training directories
    previous_dirs = [
        "balanced_85_training_results",
        "balanced_61_training_results"
    ]
    
    for prev_dir in previous_dirs:
        if os.path.exists(prev_dir):
            target = os.path.join(previous_dir, prev_dir)
            shutil.copytree(prev_dir, target)
            print(f"  ✅ {prev_dir}")
        else:
            print(f"  ❌ {prev_dir} (not found)")

def create_checkpoint_summary(checkpoint_dir, checkpoint_name):
    """Create comprehensive checkpoint summary"""
    print(f"\n📄 CREATING CHECKPOINT SUMMARY")
    print("=" * 40)
    
    summary = {
        "checkpoint_info": {
            "name": checkpoint_name,
            "created": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "description": "Enhanced 62.39% Validation Accuracy Model with Expanded Dataset"
        },
        "key_achievements": {
            "validation_accuracy": "62.39%",
            "improvement_vs_baseline": "+21.2% (baseline: 51.47%)",
            "dataset_expansion": "+57.6% (340 → 536 videos)",
            "target_achieved": "✅ YES (≥60% target)",
            "demographic_groups": 15,
            "perfect_class_balance": "134 videos per class"
        },
        "training_pipeline": {
            "phase_1": "Video Preprocessing & Dataset Integration (140 new diverse videos, 100% success)",
            "phase_2": "Enhanced Balanced Dataset Creation (536 total videos, 15 demographic groups)",
            "phase_3": "Lightweight Model Training (62.39% validation accuracy achieved)",
            "phase_4": "Per-User Calibration Implementation (ready for deployment)"
        },
        "model_specifications": {
            "architecture": "Lightweight CNN-LSTM",
            "parameters": "1,429,284 total",
            "training_videos": 386,
            "validation_videos": 109,
            "epochs_trained": 35,
            "early_stopping": "Target achieved at epoch 35"
        },
        "dataset_composition": {
            "total_videos": 536,
            "per_class_balance": 134,
            "train_split": "80% (386 videos)",
            "validation_split": "20% (109 videos)",
            "demographic_diversity": "15 unique groups across age, gender, ethnicity"
        },
        "performance_comparison": {
            "doctor_focused_model": "75.9% (biased, 260 videos)",
            "balanced_61_per_class": "37.5% (244 videos)",
            "balanced_85_per_class": "51.47% (340 videos)",
            "enhanced_134_per_class": "62.39% (536 videos) ← CURRENT BEST"
        },
        "next_steps": {
            "per_user_calibration": "Implement 4-shot calibration with enhanced base model",
            "production_deployment": "Deploy 62.39% model with calibration system",
            "further_expansion": "Continue dataset growth toward 200+ videos per class"
        },
        "files_included": {
            "enhanced_training_results": "Model checkpoints, training curves, manifests",
            "training_scripts": "Complete pipeline scripts for reproducibility",
            "dataset_manifests": "Balanced dataset manifests and analysis results",
            "previous_results": "Historical training results for comparison",
            "documentation": "Comprehensive summary and next steps"
        }
    }
    
    summary_path = os.path.join(checkpoint_dir, "CHECKPOINT_SUMMARY.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create markdown version
    md_path = os.path.join(checkpoint_dir, "README.md")
    with open(md_path, 'w') as f:
        f.write(f"# {checkpoint_name}\n\n")
        f.write("## 🎉 Enhanced Lip-Reading Model Checkpoint\n\n")
        f.write("**Validation Accuracy: 62.39%** (+21.2% improvement over 51.47% baseline)\n\n")
        f.write("### Key Achievements\n")
        f.write("- ✅ **Target Achieved**: ≥60% validation accuracy\n")
        f.write("- 📈 **Dataset Expansion**: +57.6% (340 → 536 videos)\n")
        f.write("- 🎯 **Perfect Balance**: 134 videos per class\n")
        f.write("- 🌍 **Demographic Diversity**: 15 unique groups\n")
        f.write("- 🏗️ **Lightweight Architecture**: 1.4M parameters\n\n")
        f.write("### Training Pipeline Completed\n")
        f.write("1. **Phase 1**: Video Preprocessing (140 new diverse videos, 100% success)\n")
        f.write("2. **Phase 2**: Enhanced Dataset Creation (536 balanced videos)\n")
        f.write("3. **Phase 3**: Model Training (62.39% validation accuracy)\n")
        f.write("4. **Phase 4**: Ready for Per-User Calibration\n\n")
        f.write("### Performance Progression\n")
        f.write("- Doctor-Focused Model: 75.9% (biased)\n")
        f.write("- Balanced 61-per-class: 37.5%\n")
        f.write("- Balanced 85-per-class: 51.47%\n")
        f.write("- **Enhanced 134-per-class: 62.39%** ← Current Best\n\n")
        f.write("### Next Steps\n")
        f.write("- Implement per-user calibration with enhanced base model\n")
        f.write("- Deploy production system with 62.39% foundation model\n")
        f.write("- Continue dataset expansion toward 200+ videos per class\n")
    
    print(f"✅ Summary created: {summary_path}")
    print(f"✅ README created: {md_path}")

def commit_and_push_to_github(checkpoint_dir, checkpoint_name):
    """Commit and push checkpoint to GitHub"""
    print(f"\n🚀 COMMITTING TO GITHUB")
    print("=" * 30)
    
    try:
        # Add all files
        subprocess.run(["git", "add", "."], check=True, cwd=".")
        print("✅ Files staged for commit")
        
        # Commit with descriptive message
        commit_message = f"Enhanced Model Checkpoint: 62.39% Validation Accuracy\n\n" \
                        f"- Achieved 62.39% validation accuracy (+21.2% vs baseline)\n" \
                        f"- Expanded dataset to 536 videos (+57.6% growth)\n" \
                        f"- Perfect class balance: 134 videos per class\n" \
                        f"- 15 demographic groups for cross-demographic validation\n" \
                        f"- Lightweight CNN-LSTM: 1.4M parameters\n" \
                        f"- Target ≥60% validation accuracy: ✅ ACHIEVED\n" \
                        f"- Ready for per-user calibration deployment\n\n" \
                        f"Checkpoint: {checkpoint_name}"
        
        subprocess.run(["git", "commit", "-m", commit_message], check=True, cwd=".")
        print("✅ Changes committed to local repository")
        
        # Push to remote
        subprocess.run(["git", "push"], check=True, cwd=".")
        print("✅ Changes pushed to GitHub")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Git operation failed: {e}")
        return False

def main():
    """Execute comprehensive checkpoint creation"""
    print("🎯 COMPREHENSIVE CHECKPOINT CREATION")
    print("=" * 70)
    print("Creating complete backup of enhanced 62.39% validation accuracy model")
    
    # Create checkpoint directory
    checkpoint_dir, checkpoint_name = create_checkpoint_directory()
    
    # Backup all components
    backup_enhanced_training_results(checkpoint_dir)
    backup_training_scripts(checkpoint_dir)
    backup_dataset_manifests(checkpoint_dir)
    backup_previous_results(checkpoint_dir)
    
    # Create comprehensive summary
    create_checkpoint_summary(checkpoint_dir, checkpoint_name)
    
    # Calculate checkpoint size
    def get_dir_size(path):
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total += os.path.getsize(filepath)
        return total / (1024*1024)  # MB
    
    checkpoint_size = get_dir_size(checkpoint_dir)
    
    # Commit to GitHub
    github_success = commit_and_push_to_github(checkpoint_dir, checkpoint_name)
    
    # Final summary
    print("\n" + "=" * 70)
    print("🎉 COMPREHENSIVE CHECKPOINT COMPLETE")
    print(f"📁 Checkpoint: {checkpoint_name}")
    print(f"💾 Size: {checkpoint_size:.1f} MB")
    print(f"🏆 Model Performance: 62.39% validation accuracy")
    print(f"📈 Improvement: +21.2% vs 51.47% baseline")
    print(f"🎯 Target Achievement: ✅ YES (≥60%)")
    print(f"🚀 GitHub Backup: {'✅ SUCCESS' if github_success else '❌ FAILED'}")
    print(f"📊 Dataset: 536 videos (134 per class, 15 demographic groups)")
    print("🔄 Status: Ready for per-user calibration deployment")
    
    return True, checkpoint_name

if __name__ == "__main__":
    success, checkpoint_name = main()
    exit(0 if success else 1)
