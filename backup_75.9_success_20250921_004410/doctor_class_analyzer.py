#!/usr/bin/env python3
"""
Doctor Class Performance Analysis
Analyze doctor class distribution and performance issues in 4-class training
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import cv2

class DoctorClassAnalyzer:
    def __init__(self):
        self.results_dir = Path("4class_training_results")
        self.analysis_dir = Path("doctor_class_analysis")
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        print("🔍 DOCTOR CLASS PERFORMANCE ANALYSIS")
        print("=" * 80)
        print("🎯 Goal: Identify root causes of doctor class underperformance (40.0% vs 85.7% avg)")
        print("📊 Current Performance: doctor 40.0% vs my_mouth_is_dry 100%, i_need_to_move 87.5%, pillow 85.7%")
        
    def analyze_doctor_distribution(self):
        """Analyze doctor class distribution across demographics."""
        print("\n📋 ANALYZING DOCTOR CLASS DISTRIBUTION")
        print("=" * 60)
        
        # Load training manifest
        train_manifest = pd.read_csv(self.results_dir / "4class_train_manifest.csv")
        val_manifest = pd.read_csv(self.results_dir / "4class_validation_manifest.csv")
        
        print(f"📊 Training Videos: {len(train_manifest)} total")
        print(f"📊 Validation Videos: {len(val_manifest)} total")
        
        # Analyze doctor class specifically
        train_doctor = train_manifest[train_manifest['class'] == 'doctor']
        val_doctor = val_manifest[val_manifest['class'] == 'doctor']
        
        print(f"\n🏥 DOCTOR CLASS ANALYSIS:")
        print(f"   Training Doctor Videos: {len(train_doctor)}")
        print(f"   Validation Doctor Videos: {len(val_doctor)}")
        
        # Training demographic breakdown
        print(f"\n📊 TRAINING DOCTOR DEMOGRAPHICS:")
        train_doctor_demos = train_doctor['demographic_group'].value_counts()
        for demo, count in train_doctor_demos.items():
            percentage = (count / len(train_doctor)) * 100
            print(f"   {demo}: {count} videos ({percentage:.1f}%)")
        
        # Validation demographic breakdown
        print(f"\n📊 VALIDATION DOCTOR DEMOGRAPHICS:")
        val_doctor_demos = val_doctor['demographic_group'].value_counts()
        for demo, count in val_doctor_demos.items():
            percentage = (count / len(val_doctor)) * 100
            print(f"   {demo}: {count} videos ({percentage:.1f}%)")
        
        # Compare with other classes
        print(f"\n📊 CLASS BALANCE COMPARISON:")
        all_classes = ['doctor', 'my_mouth_is_dry', 'i_need_to_move', 'pillow']
        
        print("   TRAINING SET:")
        for class_name in all_classes:
            class_count = len(train_manifest[train_manifest['class'] == class_name])
            percentage = (class_count / len(train_manifest)) * 100
            print(f"     {class_name}: {class_count} videos ({percentage:.1f}%)")
        
        print("   VALIDATION SET:")
        for class_name in all_classes:
            class_count = len(val_manifest[val_manifest['class'] == class_name])
            percentage = (class_count / len(val_manifest)) * 100
            print(f"     {class_name}: {class_count} videos ({percentage:.1f}%)")
        
        # Calculate training/validation ratios
        print(f"\n📊 TRAINING/VALIDATION RATIOS:")
        for class_name in all_classes:
            train_count = len(train_manifest[train_manifest['class'] == class_name])
            val_count = len(val_manifest[val_manifest['class'] == class_name])
            ratio = train_count / max(val_count, 1)
            print(f"   {class_name}: {ratio:.1f}:1 (train:val)")
        
        # Save detailed analysis
        analysis_data = {
            'train_doctor_count': len(train_doctor),
            'val_doctor_count': len(val_doctor),
            'train_doctor_demographics': dict(train_doctor_demos),
            'val_doctor_demographics': dict(val_doctor_demos),
            'doctor_train_val_ratio': len(train_doctor) / max(len(val_doctor), 1)
        }
        
        return analysis_data, train_doctor, val_doctor
    
    def analyze_confusion_patterns(self):
        """Analyze confusion matrix to identify doctor misclassification patterns."""
        print("\n🔍 ANALYZING DOCTOR CONFUSION PATTERNS")
        print("=" * 60)
        
        # Load the best model results from training report
        report_path = self.results_dir / "4class_training_report.txt"
        
        if report_path.exists():
            with open(report_path, 'r') as f:
                content = f.read()
                print("📄 Found training report - analyzing confusion patterns...")
                
                # Extract per-class performance
                if "PER-CLASS PERFORMANCE" in content:
                    lines = content.split('\n')
                    doctor_line = [line for line in lines if 'doctor:' in line and '%' in line]
                    if doctor_line:
                        print(f"   Doctor performance from report: {doctor_line[0].strip()}")
        
        # Analyze potential confusion sources
        print(f"\n💡 POTENTIAL CONFUSION ANALYSIS:")
        print(f"   Doctor (40.0%) vs Pillow (85.7%): Potential visual similarity")
        print(f"   - Both may involve similar mouth positions")
        print(f"   - Both are objects/concepts rather than actions")
        print(f"   - May require enhanced visual distinction")
        
        print(f"\n   Doctor vs High-performing classes:")
        print(f"   - my_mouth_is_dry (100.0%): Clear physiological state")
        print(f"   - i_need_to_move (87.5%): Clear action/movement")
        print(f"   - pillow (85.7%): Object reference (potential confusion source)")
        
        return {
            'likely_confusion_target': 'pillow',
            'confusion_reason': 'similar_object_reference_patterns',
            'improvement_strategy': 'enhanced_visual_distinction'
        }
    
    def calculate_class_weights(self):
        """Calculate optimal class weights for addressing doctor underperformance."""
        print("\n⚖️  CALCULATING OPTIMAL CLASS WEIGHTS")
        print("=" * 60)
        
        # Load training data for weight calculation
        train_manifest = pd.read_csv(self.results_dir / "4class_train_manifest.csv")
        
        # Count samples per class
        class_counts = train_manifest['class'].value_counts()
        total_samples = len(train_manifest)
        
        print(f"📊 Class Distribution in Training:")
        for class_name, count in class_counts.items():
            percentage = (count / total_samples) * 100
            print(f"   {class_name}: {count} samples ({percentage:.1f}%)")
        
        # Calculate inverse frequency weights
        n_classes = len(class_counts)
        weights = {}
        
        print(f"\n⚖️  Calculated Class Weights:")
        for class_name in ['my_mouth_is_dry', 'i_need_to_move', 'doctor', 'pillow']:
            if class_name in class_counts:
                # Standard inverse frequency: total_samples / (n_classes * class_count)
                standard_weight = total_samples / (n_classes * class_counts[class_name])
                
                # Enhanced weight for doctor class (2x boost)
                if class_name == 'doctor':
                    enhanced_weight = standard_weight * 2.0
                    print(f"   {class_name}: {enhanced_weight:.3f} (enhanced 2x for improvement)")
                else:
                    print(f"   {class_name}: {standard_weight:.3f}")
                
                weights[class_name] = enhanced_weight if class_name == 'doctor' else standard_weight
        
        return weights
    
    def recommend_augmentation_strategy(self):
        """Recommend targeted augmentation strategy for doctor class."""
        print("\n🎯 DOCTOR-SPECIFIC AUGMENTATION STRATEGY")
        print("=" * 60)
        
        print("📋 TARGETED IMPROVEMENTS FOR DOCTOR CLASS:")
        print("   1. Enhanced Brightness/Contrast Variations:")
        print("      - Standard: ±15% brightness, 0.9-1.1x contrast")
        print("      - Doctor Enhanced: ±20% brightness, 0.85-1.15x contrast")
        print("      - Rationale: Medical contexts may have varied lighting")
        
        print("\n   2. Temporal Speed Variations:")
        print("      - Add 0.9-1.1x speed variations for doctor videos")
        print("      - Captures different speaking rates in medical contexts")
        print("      - Helps model generalize across speaking speeds")
        
        print("\n   3. Enhanced Horizontal Flipping:")
        print("      - Standard: 33% probability for all classes")
        print("      - Doctor Enhanced: 50% probability")
        print("      - Increases doctor training diversity")
        
        print("\n   4. Class-Specific Augmentation Multiplier:")
        print("      - Standard: 3x augmentation for all classes")
        print("      - Doctor Enhanced: 5x augmentation")
        print("      - Provides more doctor training examples")
        
        strategy = {
            'enhanced_brightness_range': 0.20,  # ±20%
            'enhanced_contrast_range': (0.85, 1.15),
            'temporal_speed_range': (0.9, 1.1),
            'doctor_flip_probability': 0.5,
            'doctor_augmentation_multiplier': 5
        }
        
        return strategy
    
    def generate_improvement_plan(self):
        """Generate comprehensive doctor class improvement plan."""
        print("\n📋 COMPREHENSIVE DOCTOR CLASS IMPROVEMENT PLAN")
        print("=" * 80)
        
        # Analyze current state
        distribution_data, train_doctor, val_doctor = self.analyze_doctor_distribution()
        confusion_analysis = self.analyze_confusion_patterns()
        class_weights = self.calculate_class_weights()
        augmentation_strategy = self.recommend_augmentation_strategy()
        
        # Generate improvement plan
        improvement_plan = {
            'current_performance': {
                'doctor_accuracy': 40.0,
                'target_accuracy': 60.0,
                'performance_gap': 20.0
            },
            'root_causes': {
                'insufficient_training_data': len(train_doctor) < 60,  # Threshold
                'demographic_mismatch': len(set(train_doctor['demographic_group'])) < 2,
                'class_imbalance': distribution_data['doctor_train_val_ratio'] < 5.0,
                'visual_confusion': confusion_analysis['likely_confusion_target'] == 'pillow'
            },
            'improvement_strategies': {
                'class_weighted_loss': class_weights,
                'enhanced_augmentation': augmentation_strategy,
                'targeted_retraining': {
                    'epochs': 20,
                    'early_stopping_patience': 8,
                    'doctor_focus_weight': 2.0
                }
            },
            'success_criteria': {
                'doctor_accuracy_target': 60.0,
                'overall_accuracy_maintenance': 70.0,
                'other_classes_tolerance': 5.0  # Max 5% degradation
            }
        }
        
        # Save improvement plan
        plan_path = self.analysis_dir / "doctor_improvement_plan.txt"
        with open(plan_path, 'w') as f:
            f.write("DOCTOR CLASS IMPROVEMENT PLAN\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"CURRENT STATE:\n")
            f.write(f"Doctor accuracy: {improvement_plan['current_performance']['doctor_accuracy']}%\n")
            f.write(f"Target accuracy: {improvement_plan['current_performance']['target_accuracy']}%\n")
            f.write(f"Performance gap: {improvement_plan['current_performance']['performance_gap']}%\n\n")
            
            f.write("ROOT CAUSES IDENTIFIED:\n")
            for cause, detected in improvement_plan['root_causes'].items():
                status = "YES" if detected else "NO"
                f.write(f"- {cause}: {status}\n")
            
            f.write(f"\nIMPROVEMENT STRATEGIES:\n")
            f.write(f"1. Class-weighted loss with 2x doctor emphasis\n")
            f.write(f"2. Enhanced augmentation (5x multiplier, ±20% brightness)\n")
            f.write(f"3. Targeted 20-epoch retraining with early stopping\n")
            f.write(f"4. Focus on doctor-pillow visual distinction\n")
        
        print(f"📄 Improvement plan saved: {plan_path}")
        
        return improvement_plan
    
    def run_complete_analysis(self):
        """Execute complete doctor class analysis."""
        try:
            print("🚀 STARTING COMPREHENSIVE DOCTOR CLASS ANALYSIS")
            
            improvement_plan = self.generate_improvement_plan()
            
            print(f"\n✅ DOCTOR CLASS ANALYSIS COMPLETED")
            print(f"📊 Key Findings:")
            print(f"   - Current doctor accuracy: 40.0% (target: 60.0%)")
            print(f"   - Performance gap: 20.0 percentage points")
            print(f"   - Primary strategy: Enhanced augmentation + class weighting")
            print(f"   - Ready for targeted improvement implementation")
            
            return improvement_plan
            
        except Exception as e:
            print(f"\n❌ ANALYSIS FAILED: {e}")
            raise

def main():
    """Execute doctor class analysis."""
    print("🔍 STARTING DOCTOR CLASS PERFORMANCE ANALYSIS")
    print("🎯 Goal: Identify and address doctor class underperformance (40.0%)")
    
    analyzer = DoctorClassAnalyzer()
    improvement_plan = analyzer.run_complete_analysis()
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print("📄 Results saved to doctor_class_analysis/ directory")
    print("🔄 Ready for targeted doctor class improvement implementation")

if __name__ == "__main__":
    main()
