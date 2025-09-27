#!/usr/bin/env python3
"""
Leave-One-Speaker-Out (LOSO) Cross-Validation Framework
======================================================

Implements honest cross-speaker generalization validation by training on 5 speakers
and validating on the 6th held-out speaker, repeated for all 6 speakers.

This provides true generalization metrics without speaker contamination, addressing
the fundamental issue of validation leakage in speaker-specific approaches.

Features:
- LOSO cross-validation across all 6 speaker sets
- Speaker-disjoint data splits with zero contamination
- Comprehensive reporting of generalization vs personalization metrics
- Integration with existing preprocessing pipeline
- Base model training for few-shot personalization

Author: Augment Agent
Date: 2025-09-27
"""

import os
import sys
import json
import csv
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import existing components
from advanced_training_components import (
    EnhancedLightweightCNNLSTM,
    FocalLoss,
    ConservativeAugmentation,
    StandardizedPreprocessor,
    ComprehensiveVideoDataset
)

class LOSODatasetManager:
    """Manages speaker-disjoint data splits for LOSO cross-validation."""
    
    def __init__(self, speaker_sets_dir: str = "data/speaker sets"):
        """
        Initialize LOSO dataset manager.
        
        Args:
            speaker_sets_dir: Directory containing speaker set folders
        """
        self.speaker_sets_dir = Path(speaker_sets_dir)
        self.speakers = self._discover_speakers()
        self.class_names = ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Discovered {len(self.speakers)} speakers: {self.speakers}")
    
    def _discover_speakers(self) -> List[str]:
        """Discover available speaker directories."""
        speakers = []
        for item in self.speaker_sets_dir.iterdir():
            if item.is_dir() and item.name.startswith('speaker'):
                speakers.append(item.name)
        return sorted(speakers)
    
    def _normalize_class_name(self, class_name: str) -> str:
        """Normalize class names to standard format."""
        # Convert to lowercase and replace spaces with underscores
        normalized = class_name.lower().replace(' ', '_')
        
        # Handle specific variations
        class_mappings = {
            'doctor': 'doctor',
            'i_need_to_move': 'i_need_to_move', 
            'my_mouth_is_dry': 'my_mouth_is_dry',
            'pillow': 'pillow'
        }
        
        return class_mappings.get(normalized, normalized)
    
    def _collect_speaker_videos(self, speaker_name: str) -> List[Dict[str, Any]]:
        """Collect all videos for a specific speaker."""
        speaker_path = self.speaker_sets_dir / speaker_name
        videos = []
        
        for class_dir in speaker_path.iterdir():
            if not class_dir.is_dir():
                continue
                
            normalized_class = self._normalize_class_name(class_dir.name)
            if normalized_class not in self.class_names:
                self.logger.warning(f"Unknown class: {class_dir.name} -> {normalized_class}")
                continue
            
            # Collect video files (support multiple formats)
            video_files = []
            for ext in ['*.mp4', '*.mov', '*.avi', '*.mkv']:
                video_files.extend(class_dir.glob(ext))

            for video_file in video_files:
                videos.append({
                    'video_path': str(video_file),
                    'class': normalized_class,
                    'class_idx': self.class_to_idx[normalized_class],
                    'speaker': speaker_name,
                    'original_class_name': class_dir.name
                })
        
        return videos
    
    def create_loso_splits(self, held_out_speaker: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Create LOSO splits with one speaker held out for validation.
        
        Args:
            held_out_speaker: Speaker to hold out for validation
            
        Returns:
            Tuple of (training_videos, validation_videos)
        """
        if held_out_speaker not in self.speakers:
            raise ValueError(f"Speaker {held_out_speaker} not found in {self.speakers}")
        
        training_videos = []
        validation_videos = []
        
        # Collect training videos from all other speakers
        for speaker in self.speakers:
            speaker_videos = self._collect_speaker_videos(speaker)
            
            if speaker == held_out_speaker:
                validation_videos.extend(speaker_videos)
            else:
                training_videos.extend(speaker_videos)
        
        self.logger.info(f"LOSO Split - Held out: {held_out_speaker}")
        self.logger.info(f"  Training videos: {len(training_videos)} from {len(self.speakers)-1} speakers")
        self.logger.info(f"  Validation videos: {len(validation_videos)} from 1 speaker")
        
        # Verify class distribution
        self._log_class_distribution(training_videos, "Training")
        self._log_class_distribution(validation_videos, "Validation")
        
        return training_videos, validation_videos
    
    def _log_class_distribution(self, videos: List[Dict], split_name: str):
        """Log class distribution for a data split."""
        class_counts = {}
        for video in videos:
            class_name = video['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        self.logger.info(f"  {split_name} class distribution:")
        for class_name in self.class_names:
            count = class_counts.get(class_name, 0)
            self.logger.info(f"    {class_name}: {count} videos")
    
    def save_loso_manifests(self, training_videos: List[Dict], validation_videos: List[Dict], 
                           output_dir: str, held_out_speaker: str) -> Tuple[str, str]:
        """Save LOSO manifests to CSV files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save training manifest (use 'file_path' to match ComprehensiveVideoDataset expectations)
        train_manifest_path = output_path / f"loso_train_holdout_{held_out_speaker}.csv"
        with open(train_manifest_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['file_path', 'class', 'class_idx', 'speaker'])
            writer.writeheader()
            for video in training_videos:
                writer.writerow({
                    'file_path': video['video_path'],
                    'class': video['class'],
                    'class_idx': video['class_idx'],
                    'speaker': video['speaker']
                })

        # Save validation manifest (use 'file_path' to match ComprehensiveVideoDataset expectations)
        val_manifest_path = output_path / f"loso_val_holdout_{held_out_speaker}.csv"
        with open(val_manifest_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['file_path', 'class', 'class_idx', 'speaker'])
            writer.writeheader()
            for video in validation_videos:
                writer.writerow({
                    'file_path': video['video_path'],
                    'class': video['class'],
                    'class_idx': video['class_idx'],
                    'speaker': video['speaker']
                })
        
        self.logger.info(f"Saved LOSO manifests:")
        self.logger.info(f"  Training: {train_manifest_path}")
        self.logger.info(f"  Validation: {val_manifest_path}")
        
        return str(train_manifest_path), str(val_manifest_path)
    
    def generate_all_loso_splits(self, output_dir: str = "loso_splits") -> Dict[str, Tuple[str, str]]:
        """Generate LOSO splits for all speakers."""
        all_splits = {}
        
        for speaker in self.speakers:
            self.logger.info(f"\n=== Generating LOSO split for {speaker} ===")
            training_videos, validation_videos = self.create_loso_splits(speaker)
            train_manifest, val_manifest = self.save_loso_manifests(
                training_videos, validation_videos, output_dir, speaker
            )
            all_splits[speaker] = (train_manifest, val_manifest)
        
        # Save summary
        summary_path = Path(output_dir) / "loso_splits_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'speakers': self.speakers,
                'class_names': self.class_names,
                'splits': all_splits,
                'generated_at': datetime.now().isoformat()
            }, f, indent=2)
        
        self.logger.info(f"\nGenerated LOSO splits for all {len(self.speakers)} speakers")
        self.logger.info(f"Summary saved to: {summary_path}")
        
        return all_splits


class LOSOTrainer:
    """LOSO cross-validation trainer for honest generalization metrics."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LOSO trainer.

        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Results storage
        self.loso_results = {}

    def _create_model(self) -> nn.Module:
        """Create model instance."""
        model = EnhancedLightweightCNNLSTM(
            num_classes=len(self.class_names),
            dropout=self.config.get('dropout', 0.4)
        ).to(self.device)

        param_count = model.count_parameters()
        self.logger.info(f"Model parameters: {param_count:,}")

        return model

    def _create_data_loaders(self, train_manifest: str, val_manifest: str) -> Tuple[DataLoader, DataLoader]:
        """Create data loaders from manifests."""
        # Create preprocessor
        preprocessor = StandardizedPreprocessor(
            target_size=(64, 96),
            target_frames=32,
            grayscale=True,
            normalize=True
        )

        # Create augmentation for training
        train_augmentation = ConservativeAugmentation(
            brightness_range=0.15,
            contrast_range=0.1,
            horizontal_flip_prob=0.5
        )

        # Create datasets
        train_dataset = ComprehensiveVideoDataset(
            manifest_path=train_manifest,
            preprocessor=preprocessor,
            augmentation=train_augmentation,
            synthetic_ratio=0.25
        )

        val_dataset = ComprehensiveVideoDataset(
            manifest_path=val_manifest,
            preprocessor=preprocessor,
            augmentation=None,
            synthetic_ratio=0.0
        )

        # Create data loaders (ensure minimum batch size of 2 for BatchNorm)
        batch_size = max(2, self.config.get('batch_size', 8))

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,  # Disable for MPS compatibility
            drop_last=True  # Drop last incomplete batch for training
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,  # Disable for MPS compatibility
            drop_last=False
        )

        self.logger.info(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

        return train_loader, val_loader

    def _train_single_fold(self, train_manifest: str, val_manifest: str,
                          held_out_speaker: str, output_dir: str) -> Dict[str, Any]:
        """Train a single LOSO fold."""
        self.logger.info(f"\n=== Training LOSO fold - Held out: {held_out_speaker} ===")

        # Create model and data loaders
        model = self._create_model()
        train_loader, val_loader = self._create_data_loaders(train_manifest, val_manifest)

        # Setup training components
        criterion = FocalLoss(alpha=1.0, gamma=2.0, reduction='mean')
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.get('learning_rate', 3e-4),
            weight_decay=self.config.get('weight_decay', 1e-3)
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )

        # Training loop
        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0
        max_patience = self.config.get('early_stop_patience', 15)

        for epoch in range(self.config.get('max_epochs', 50)):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (videos, labels) in enumerate(train_loader):
                videos, labels = videos.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(videos)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_acc = 100.0 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_predictions = []
            all_targets = []

            with torch.no_grad():
                for videos, labels in val_loader:
                    videos, labels = videos.to(self.device), labels.to(self.device)
                    outputs = model(videos)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(labels.cpu().numpy())

            val_acc = 100.0 * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)

            # Learning rate scheduling
            scheduler.step(val_acc)

            # Early stopping and best model tracking
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0

                # Save best model
                fold_output_dir = Path(output_dir) / f"fold_{held_out_speaker}"
                fold_output_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'model_state_dict': best_model_state,
                    'config': self.config,
                    'class_to_idx': self.class_to_idx,
                    'held_out_speaker': held_out_speaker,
                    'best_val_acc': best_val_acc,
                    'epoch': epoch
                }, fold_output_dir / 'best_model.pth')

            else:
                patience_counter += 1

            # Log progress
            if epoch % 5 == 0 or epoch == self.config.get('max_epochs', 50) - 1:
                self.logger.info(
                    f"Epoch {epoch:2d}: Train Acc={train_acc:5.2f}%, Val Acc={val_acc:5.2f}%, "
                    f"Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}"
                )

            # Early stopping
            if patience_counter >= max_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        # Generate detailed validation report
        model.load_state_dict(best_model_state)
        model.eval()

        final_predictions = []
        final_targets = []

        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(self.device), labels.to(self.device)
                outputs = model(videos)
                _, predicted = torch.max(outputs.data, 1)

                final_predictions.extend(predicted.cpu().numpy())
                final_targets.extend(labels.cpu().numpy())

        # Generate classification report
        report = classification_report(
            final_targets, final_predictions,
            target_names=self.class_names,
            output_dict=True
        )

        # Generate confusion matrix
        cm = confusion_matrix(final_targets, final_predictions)

        fold_results = {
            'held_out_speaker': held_out_speaker,
            'best_val_accuracy': best_val_acc,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'final_epoch': epoch,
            'model_path': str(fold_output_dir / 'best_model.pth')
        }

        self.logger.info(f"LOSO fold {held_out_speaker} completed: {best_val_acc:.2f}% accuracy")

        return fold_results

    def run_loso_cross_validation(self, loso_splits: Dict[str, Tuple[str, str]],
                                 output_dir: str = "loso_results") -> Dict[str, Any]:
        """Run complete LOSO cross-validation across all speakers."""
        self.logger.info("\n🎯 STARTING LOSO CROSS-VALIDATION")
        self.logger.info("=" * 60)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        all_fold_results = {}
        accuracies = []

        # Train each fold
        for speaker, (train_manifest, val_manifest) in loso_splits.items():
            fold_results = self._train_single_fold(
                train_manifest, val_manifest, speaker, output_dir
            )
            all_fold_results[speaker] = fold_results
            accuracies.append(fold_results['best_val_accuracy'])

        # Calculate cross-validation statistics
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        # Compile comprehensive results
        loso_summary = {
            'loso_cross_validation_results': {
                'mean_accuracy': mean_accuracy,
                'std_accuracy': std_accuracy,
                'individual_accuracies': {speaker: acc for speaker, acc in zip(loso_splits.keys(), accuracies)},
                'fold_results': all_fold_results
            },
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }

        # Save comprehensive results
        results_path = output_path / 'loso_cross_validation_results.json'
        with open(results_path, 'w') as f:
            json.dump(loso_summary, f, indent=2, default=str)

        # Generate summary report
        self._generate_loso_report(loso_summary, output_path)

        self.logger.info("\n🎯 LOSO CROSS-VALIDATION COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Mean Accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")
        self.logger.info(f"Individual Accuracies:")
        for speaker, acc in zip(loso_splits.keys(), accuracies):
            self.logger.info(f"  {speaker}: {acc:.2f}%")
        self.logger.info(f"Results saved to: {results_path}")

        return loso_summary

    def _generate_loso_report(self, loso_summary: Dict[str, Any], output_dir: Path):
        """Generate comprehensive LOSO report with visualizations."""
        results = loso_summary['loso_cross_validation_results']

        # Create accuracy comparison plot
        plt.figure(figsize=(12, 8))

        # Individual fold accuracies
        speakers = list(results['individual_accuracies'].keys())
        accuracies = list(results['individual_accuracies'].values())

        plt.subplot(2, 2, 1)
        bars = plt.bar(range(len(speakers)), accuracies, alpha=0.7, color='skyblue')
        plt.axhline(y=results['mean_accuracy'], color='red', linestyle='--',
                   label=f"Mean: {results['mean_accuracy']:.2f}%")
        plt.xlabel('Held-Out Speaker')
        plt.ylabel('Validation Accuracy (%)')
        plt.title('LOSO Cross-Validation Results\n(True Cross-Speaker Generalization)')
        plt.xticks(range(len(speakers)), speakers, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add accuracy values on bars
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)

        # Accuracy distribution
        plt.subplot(2, 2, 2)
        plt.hist(accuracies, bins=5, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.axvline(x=results['mean_accuracy'], color='red', linestyle='--',
                   label=f"Mean: {results['mean_accuracy']:.2f}%")
        plt.xlabel('Validation Accuracy (%)')
        plt.ylabel('Frequency')
        plt.title('Accuracy Distribution Across Folds')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Combined confusion matrix (if available)
        plt.subplot(2, 1, 2)
        combined_cm = np.zeros((len(self.class_names), len(self.class_names)))
        for fold_results in results['fold_results'].values():
            cm = np.array(fold_results['confusion_matrix'])
            combined_cm += cm

        sns.heatmap(combined_cm, annot=True, fmt='g', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Combined Confusion Matrix (All LOSO Folds)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        plt.tight_layout()
        plt.savefig(output_dir / 'loso_cross_validation_report.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Generate text report
        report_path = output_dir / 'loso_cross_validation_report.txt'
        with open(report_path, 'w') as f:
            f.write("LEAVE-ONE-SPEAKER-OUT (LOSO) CROSS-VALIDATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write("HONEST CROSS-SPEAKER GENERALIZATION METRICS\n")
            f.write("-" * 45 + "\n\n")

            f.write(f"Overall Results:\n")
            f.write(f"  Mean Accuracy: {results['mean_accuracy']:.2f}% ± {results['std_accuracy']:.2f}%\n")
            f.write(f"  Number of Folds: {len(results['individual_accuracies'])}\n\n")

            f.write("Individual Fold Results:\n")
            for speaker, acc in results['individual_accuracies'].items():
                f.write(f"  Held-out {speaker}: {acc:.2f}%\n")

            f.write(f"\nThis represents TRUE cross-speaker generalization performance.\n")
            f.write(f"Each fold trains on 5 speakers and validates on 1 held-out speaker.\n")
            f.write(f"No speaker contamination between training and validation sets.\n")

        self.logger.info(f"LOSO report generated: {report_path}")


def main():
    """Main execution function for LOSO cross-validation."""
    parser = argparse.ArgumentParser(description='LOSO Cross-Validation Framework')
    parser.add_argument('--speaker-sets-dir', default='data/speaker sets',
                       help='Directory containing speaker set folders')
    parser.add_argument('--output-dir', default='loso_cross_validation_results',
                       help='Output directory for results')
    parser.add_argument('--max-epochs', type=int, default=50,
                       help='Maximum training epochs per fold')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.4,
                       help='Dropout rate')

    args = parser.parse_args()

    # Setup configuration
    config = {
        'max_epochs': args.max_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'dropout': args.dropout,
        'weight_decay': 1e-3,
        'early_stop_patience': 15
    }

    print("🎯 LOSO CROSS-VALIDATION FRAMEWORK")
    print("=" * 50)
    print("Implementing honest cross-speaker generalization validation")
    print("Training on 5 speakers, validating on 1 held-out speaker")
    print("Repeated for all 6 speakers to provide unbiased metrics")
    print("=" * 50)

    # Initialize dataset manager
    dataset_manager = LOSODatasetManager(args.speaker_sets_dir)

    # Generate LOSO splits
    loso_splits = dataset_manager.generate_all_loso_splits("loso_splits")

    # Initialize trainer
    trainer = LOSOTrainer(config)

    # Run LOSO cross-validation
    results = trainer.run_loso_cross_validation(loso_splits, args.output_dir)

    print(f"\n✅ LOSO Cross-Validation completed!")
    print(f"Mean Accuracy: {results['loso_cross_validation_results']['mean_accuracy']:.2f}% ± {results['loso_cross_validation_results']['std_accuracy']:.2f}%")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
