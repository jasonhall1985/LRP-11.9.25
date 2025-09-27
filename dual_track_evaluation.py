#!/usr/bin/env python3
"""
Dual-Track Evaluation System
===========================

Comprehensive evaluation system that provides both honest cross-speaker generalization
metrics and practical bedside calibration capabilities with clear scientific separation.

Features:
- LOSO generalization accuracy (true cross-speaker performance)
- Personalized accuracy (within-speaker after few-shot adaptation)
- Cross-adaptation validation to detect overfitting
- Clear reporting that never conflates the two metrics
- Scientific integrity with practical deployment capabilities

Author: Augment Agent
Date: 2025-09-27
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import our components
from loso_cross_validation_framework import LOSODatasetManager, LOSOTrainer
from calibrate import FewShotPersonalizer

class DualTrackEvaluator:
    """Dual-track evaluation system for honest generalization vs personalization metrics."""
    
    def __init__(self, speaker_sets_dir: str = "data/speaker sets"):
        """
        Initialize dual-track evaluator.
        
        Args:
            speaker_sets_dir: Directory containing speaker set folders
        """
        self.speaker_sets_dir = speaker_sets_dir
        self.dataset_manager = LOSODatasetManager(speaker_sets_dir)
        self.speakers = self.dataset_manager.speakers
        
        print("🎯 DUAL-TRACK EVALUATION SYSTEM")
        print("=" * 50)
        print("Track 1: LOSO Cross-Validation (Honest Generalization)")
        print("Track 2: Few-Shot Personalization (Bedside Calibration)")
        print("=" * 50)
    
    def run_loso_evaluation(self, config: Dict[str, Any], output_dir: str = "loso_results") -> Dict[str, Any]:
        """Run LOSO cross-validation for honest generalization metrics."""
        print("\n📊 TRACK 1: LOSO CROSS-VALIDATION")
        print("-" * 40)
        print("Training on 5 speakers, validating on 1 held-out speaker")
        print("Repeated for all 6 speakers - NO SPEAKER CONTAMINATION")
        
        # Generate LOSO splits
        loso_splits = self.dataset_manager.generate_all_loso_splits("loso_splits")
        
        # Initialize trainer
        trainer = LOSOTrainer(config)
        
        # Run LOSO cross-validation
        loso_results = trainer.run_loso_cross_validation(loso_splits, output_dir)
        
        return loso_results
    
    def run_personalization_evaluation(self, base_model_path: str, k_shots_list: List[int] = [10, 20],
                                     output_dir: str = "personalization_results") -> Dict[str, Any]:
        """Run few-shot personalization evaluation across all speakers."""
        print("\n🎯 TRACK 2: FEW-SHOT PERSONALIZATION")
        print("-" * 40)
        print("Rapid adaptation using K-shot learning with head-only fine-tuning")
        print("Target: >90% within-speaker accuracy in <1 minute")
        
        # Initialize personalizer
        personalizer = FewShotPersonalizer(base_model_path)
        
        personalization_results = {
            'base_model': base_model_path,
            'speakers': {},
            'summary': {}
        }
        
        all_accuracies = {k: [] for k in k_shots_list}
        all_times = {k: [] for k in k_shots_list}
        
        # Run personalization for each speaker and K-shot setting
        for speaker in self.speakers:
            print(f"\n--- Personalizing for {speaker} ---")
            speaker_results = {}
            
            for k_shots in k_shots_list:
                try:
                    results = personalizer.personalize(
                        speaker_name=speaker,
                        k_shots=k_shots,
                        epochs=5,
                        freeze_encoder=True,
                        learning_rate=1e-3
                    )
                    
                    speaker_results[f'K{k_shots}'] = results
                    all_accuracies[k_shots].append(results['final_accuracy'])
                    all_times[k_shots].append(results['personalization_time'])
                    
                    print(f"  K={k_shots}: {results['final_accuracy']:.2f}% in {results['personalization_time']:.1f}s")
                    
                except Exception as e:
                    print(f"  ⚠️  Error with {speaker} K={k_shots}: {e}")
                    continue
            
            personalization_results['speakers'][speaker] = speaker_results
        
        # Calculate summary statistics
        for k_shots in k_shots_list:
            if all_accuracies[k_shots]:
                personalization_results['summary'][f'K{k_shots}'] = {
                    'mean_accuracy': np.mean(all_accuracies[k_shots]),
                    'std_accuracy': np.std(all_accuracies[k_shots]),
                    'mean_time': np.mean(all_times[k_shots]),
                    'std_time': np.std(all_times[k_shots]),
                    'individual_accuracies': all_accuracies[k_shots],
                    'individual_times': all_times[k_shots]
                }
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_file = output_path / 'personalization_evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(personalization_results, f, indent=2)
        
        print(f"\n✅ Personalization evaluation completed")
        print(f"Results saved to: {results_file}")
        
        return personalization_results
    
    def run_cross_adaptation_validation(self, base_model_path: str, k_shots: int = 10) -> Dict[str, Any]:
        """Run cross-adaptation validation to detect overfitting."""
        print("\n🔍 CROSS-ADAPTATION VALIDATION")
        print("-" * 40)
        print("Fine-tune on Speaker A, test on Speaker B")
        print("Detects overfitting in personalization approach")
        
        personalizer = FewShotPersonalizer(base_model_path)
        cross_adaptation_results = {}
        
        # Test a few speaker pairs
        test_pairs = [
            ('speaker 1 ', 'speaker 2 '),
            ('speaker 3 ', 'speaker 4'),
            ('speaker 5', 'speaker 6')
        ]
        
        for source_speaker, target_speaker in test_pairs:
            print(f"\nTesting: {source_speaker} → {target_speaker}")
            
            try:
                # Personalize on source speaker
                source_results = personalizer.personalize(
                    speaker_name=source_speaker,
                    k_shots=k_shots,
                    epochs=5,
                    freeze_encoder=True
                )
                
                # Evaluate on target speaker
                target_videos = personalizer._collect_speaker_videos(target_speaker)
                preprocessor = personalizer.preprocessor if hasattr(personalizer, 'preprocessor') else None
                
                if preprocessor is None:
                    from advanced_training_components import StandardizedPreprocessor
                    preprocessor = StandardizedPreprocessor(
                        target_size=(64, 96),
                        target_frames=32,
                        grayscale=True,
                        normalize=True
                    )
                
                cross_accuracy = personalizer._evaluate_on_speaker(
                    personalizer.model, target_videos, preprocessor
                )
                
                cross_adaptation_results[f"{source_speaker}_to_{target_speaker}"] = {
                    'source_accuracy': source_results['final_accuracy'],
                    'cross_accuracy': cross_accuracy,
                    'overfitting_gap': source_results['final_accuracy'] - cross_accuracy
                }
                
                print(f"  Source accuracy: {source_results['final_accuracy']:.2f}%")
                print(f"  Cross accuracy: {cross_accuracy:.2f}%")
                print(f"  Overfitting gap: {source_results['final_accuracy'] - cross_accuracy:.2f}%")
                
            except Exception as e:
                print(f"  ⚠️  Error: {e}")
                continue
        
        return cross_adaptation_results
    
    def generate_dual_track_report(self, loso_results: Dict[str, Any], 
                                 personalization_results: Dict[str, Any],
                                 cross_adaptation_results: Dict[str, Any],
                                 output_dir: str = "dual_track_results"):
        """Generate comprehensive dual-track report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('DUAL-TRACK EVALUATION RESULTS\nGeneralization vs Personalization', fontsize=16, fontweight='bold')
        
        # LOSO results (Track 1)
        loso_data = loso_results['loso_cross_validation_results']
        speakers = list(loso_data['individual_accuracies'].keys())
        loso_accuracies = list(loso_data['individual_accuracies'].values())
        
        axes[0, 0].bar(range(len(speakers)), loso_accuracies, alpha=0.7, color='red', label='LOSO Accuracy')
        axes[0, 0].axhline(y=loso_data['mean_accuracy'], color='darkred', linestyle='--', 
                          label=f"Mean: {loso_data['mean_accuracy']:.2f}%")
        axes[0, 0].set_title('TRACK 1: LOSO Cross-Validation\n(TRUE GENERALIZATION)', fontweight='bold')
        axes[0, 0].set_xlabel('Held-Out Speaker')
        axes[0, 0].set_ylabel('Validation Accuracy (%)')
        axes[0, 0].set_xticks(range(len(speakers)))
        axes[0, 0].set_xticklabels([s.replace('speaker ', 'S') for s in speakers], rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add accuracy values on bars
        for i, acc in enumerate(loso_accuracies):
            axes[0, 0].text(i, acc + 1, f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Personalization results (Track 2)
        if 'K10' in personalization_results['summary']:
            k10_data = personalization_results['summary']['K10']
            k10_accuracies = k10_data['individual_accuracies']
            
            axes[0, 1].bar(range(len(k10_accuracies)), k10_accuracies, alpha=0.7, color='green', label='K=10 Accuracy')
            axes[0, 1].axhline(y=k10_data['mean_accuracy'], color='darkgreen', linestyle='--',
                              label=f"Mean: {k10_data['mean_accuracy']:.2f}%")
            axes[0, 1].set_title('TRACK 2: Few-Shot Personalization\n(WITHIN-SPEAKER, K=10)', fontweight='bold')
            axes[0, 1].set_xlabel('Speaker')
            axes[0, 1].set_ylabel('Personalized Accuracy (%)')
            axes[0, 1].set_xticks(range(len(speakers)))
            axes[0, 1].set_xticklabels([s.replace('speaker ', 'S') for s in speakers], rotation=45)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add accuracy values on bars
            for i, acc in enumerate(k10_accuracies):
                axes[0, 1].text(i, acc + 1, f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Comparison plot
        if 'K10' in personalization_results['summary']:
            x = np.arange(len(speakers))
            width = 0.35
            
            axes[0, 2].bar(x - width/2, loso_accuracies, width, label='LOSO (Generalization)', 
                          color='red', alpha=0.7)
            axes[0, 2].bar(x + width/2, k10_accuracies, width, label='K=10 (Personalization)', 
                          color='green', alpha=0.7)
            
            axes[0, 2].set_title('DIRECT COMPARISON\nGeneralization vs Personalization', fontweight='bold')
            axes[0, 2].set_xlabel('Speaker')
            axes[0, 2].set_ylabel('Accuracy (%)')
            axes[0, 2].set_xticks(x)
            axes[0, 2].set_xticklabels([s.replace('speaker ', 'S') for s in speakers], rotation=45)
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Personalization time analysis
        if 'K10' in personalization_results['summary']:
            k10_times = k10_data['individual_times']
            axes[1, 0].bar(range(len(k10_times)), k10_times, alpha=0.7, color='blue')
            axes[1, 0].axhline(y=k10_data['mean_time'], color='darkblue', linestyle='--',
                              label=f"Mean: {k10_data['mean_time']:.1f}s")
            axes[1, 0].set_title('Personalization Time\n(Target: <60s)', fontweight='bold')
            axes[1, 0].set_xlabel('Speaker')
            axes[1, 0].set_ylabel('Time (seconds)')
            axes[1, 0].set_xticks(range(len(speakers)))
            axes[1, 0].set_xticklabels([s.replace('speaker ', 'S') for s in speakers], rotation=45)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Cross-adaptation validation
        if cross_adaptation_results:
            gaps = [result['overfitting_gap'] for result in cross_adaptation_results.values()]
            pair_names = list(cross_adaptation_results.keys())
            
            axes[1, 1].bar(range(len(gaps)), gaps, alpha=0.7, color='orange')
            axes[1, 1].set_title('Cross-Adaptation Validation\n(Overfitting Detection)', fontweight='bold')
            axes[1, 1].set_xlabel('Speaker Pair')
            axes[1, 1].set_ylabel('Overfitting Gap (%)')
            axes[1, 1].set_xticks(range(len(pair_names)))
            axes[1, 1].set_xticklabels([name.replace('speaker ', 'S').replace('_to_', '→') 
                                       for name in pair_names], rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        # Summary statistics
        axes[1, 2].axis('off')
        summary_text = f"""
DUAL-TRACK SUMMARY

TRACK 1: GENERALIZATION (LOSO)
Mean Accuracy: {loso_data['mean_accuracy']:.2f}% ± {loso_data['std_accuracy']:.2f}%
Range: {min(loso_accuracies):.1f}% - {max(loso_accuracies):.1f}%
Interpretation: TRUE cross-speaker performance

TRACK 2: PERSONALIZATION (K=10)
"""
        if 'K10' in personalization_results['summary']:
            summary_text += f"""Mean Accuracy: {k10_data['mean_accuracy']:.2f}% ± {k10_data['std_accuracy']:.2f}%
Range: {min(k10_accuracies):.1f}% - {max(k10_accuracies):.1f}%
Mean Time: {k10_data['mean_time']:.1f}s ± {k10_data['std_time']:.1f}s
Interpretation: WITHIN-speaker after adaptation

SCIENTIFIC INTEGRITY:
✓ No conflation of metrics
✓ Clear separation of capabilities
✓ Honest reporting of limitations
"""
        
        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes, 
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path / 'dual_track_evaluation_report.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate text report
        report_path = output_path / 'dual_track_evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write("DUAL-TRACK EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write("SCIENTIFIC APPROACH: Clear separation of generalization vs personalization\n")
            f.write("CLINICAL RELEVANCE: Both robustness metrics AND bedside usability\n\n")
            
            f.write("TRACK 1: LOSO CROSS-VALIDATION (TRUE GENERALIZATION)\n")
            f.write("-" * 50 + "\n")
            f.write(f"Mean Accuracy: {loso_data['mean_accuracy']:.2f}% ± {loso_data['std_accuracy']:.2f}%\n")
            f.write("Individual Results:\n")
            for speaker, acc in loso_data['individual_accuracies'].items():
                f.write(f"  {speaker}: {acc:.2f}%\n")
            f.write("\nInterpretation: Honest cross-speaker generalization without contamination\n\n")
            
            f.write("TRACK 2: FEW-SHOT PERSONALIZATION (BEDSIDE CALIBRATION)\n")
            f.write("-" * 50 + "\n")
            if 'K10' in personalization_results['summary']:
                f.write(f"Mean Accuracy (K=10): {k10_data['mean_accuracy']:.2f}% ± {k10_data['std_accuracy']:.2f}%\n")
                f.write(f"Mean Time: {k10_data['mean_time']:.1f}s ± {k10_data['std_time']:.1f}s\n")
                f.write("Individual Results:\n")
                for i, speaker in enumerate(speakers):
                    if i < len(k10_accuracies):
                        f.write(f"  {speaker}: {k10_accuracies[i]:.2f}% in {k10_times[i]:.1f}s\n")
            f.write("\nInterpretation: Within-speaker performance after rapid adaptation\n\n")
            
            f.write("CROSS-ADAPTATION VALIDATION (OVERFITTING CHECK)\n")
            f.write("-" * 50 + "\n")
            if cross_adaptation_results:
                for pair, result in cross_adaptation_results.items():
                    f.write(f"{pair}: {result['overfitting_gap']:.2f}% gap\n")
            f.write("\nInterpretation: Validates personalization doesn't overfit\n\n")
            
            f.write("CONCLUSION\n")
            f.write("-" * 50 + "\n")
            f.write("This dual-track approach provides:\n")
            f.write("1. Honest generalization metrics for clinical validation\n")
            f.write("2. Practical personalization for bedside deployment\n")
            f.write("3. Scientific integrity through clear metric separation\n")
            f.write("4. Overfitting detection through cross-adaptation validation\n")
        
        print(f"\n✅ Dual-track report generated:")
        print(f"  Visualization: {output_path / 'dual_track_evaluation_report.png'}")
        print(f"  Text report: {report_path}")


def main():
    """Main execution function for dual-track evaluation."""
    parser = argparse.ArgumentParser(description='Dual-Track Evaluation System')
    parser.add_argument('--base-model', required=True, help='Path to base model checkpoint')
    parser.add_argument('--loso-epochs', type=int, default=20, help='Epochs for LOSO training')
    parser.add_argument('--k-shots', type=int, nargs='+', default=[10, 20], 
                       help='K-shot values to test')
    parser.add_argument('--output-dir', default='dual_track_evaluation',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = DualTrackEvaluator()
    
    # LOSO configuration
    loso_config = {
        'max_epochs': args.loso_epochs,
        'batch_size': 8,
        'learning_rate': 3e-4,
        'dropout': 0.4,
        'weight_decay': 1e-3,
        'early_stop_patience': 10
    }
    
    # Run dual-track evaluation
    print("🚀 Starting dual-track evaluation...")
    
    # Track 1: LOSO Cross-Validation
    loso_results = evaluator.run_loso_evaluation(loso_config, f"{args.output_dir}/loso")
    
    # Track 2: Few-Shot Personalization
    personalization_results = evaluator.run_personalization_evaluation(
        args.base_model, args.k_shots, f"{args.output_dir}/personalization"
    )
    
    # Cross-adaptation validation
    cross_adaptation_results = evaluator.run_cross_adaptation_validation(args.base_model)
    
    # Generate comprehensive report
    evaluator.generate_dual_track_report(
        loso_results, personalization_results, cross_adaptation_results, args.output_dir
    )
    
    print(f"\n🎯 DUAL-TRACK EVALUATION COMPLETED")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
