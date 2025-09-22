#!/usr/bin/env python3
"""
4-Class Cross-Demographic Lip-Reading Training Pipeline
Based on comprehensive dataset analysis and validated binary architecture
"""

import os
import csv
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import time
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

class FourClassCrossDemographicTrainer:
    def __init__(self):
        self.analysis_dir = Path("4class_analysis_results")
        self.output_dir = Path("4class_training_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration based on validated binary experiments
        self.batch_size = 6
        self.max_epochs = 60
        self.initial_lr = 0.005
        self.device = torch.device('cpu')
        
        # Success criteria adjusted for 4-class complexity
        self.target_train_acc = 90.0  # Proven achievable from binary experiments
        self.target_val_acc = 70.0    # Realistic for 4-class vs 62.5% binary
        
        # Recommended 4-class configuration from analysis
        self.selected_classes = ['my_mouth_is_dry', 'i_need_to_move', 'doctor', 'pillow']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.selected_classes)}
        
        # Training demographics (combined for maximum data)
        self.training_demographics = {'65plus_female_caucasian', '18to39_male_not_specified'}
        
        # Validation demographic (best candidate from analysis)
        self.validation_demographic = '18to39_female_caucasian'  # 59 videos across 7 classes
        
        print("🚀 4-CLASS CROSS-DEMOGRAPHIC TRAINING PIPELINE")
        print("=" * 80)
        print("🎯 PRIMARY GOAL: Scale from binary (62.5%) to 4-class cross-demographic training")
        print(f"📊 Selected Classes: {', '.join(self.selected_classes)}")
        print(f"🎯 Targets: {self.target_train_acc}% training, {self.target_val_acc}% cross-demographic validation")
        print(f"👥 Training Demographics: {', '.join(self.training_demographics)}")
        print(f"👤 Validation Demographic: {self.validation_demographic}")
        
    def create_4class_manifests(self):
        """Create 4-class training and validation manifests with strict demographic separation."""
        print("\n📋 CREATING 4-CLASS MANIFESTS WITH DEMOGRAPHIC SEPARATION")
        print("=" * 70)
        
        # Load the comprehensive video inventory
        inventory_path = self.analysis_dir / "full_video_inventory.csv"
        if not inventory_path.exists():
            raise FileNotFoundError(f"Video inventory not found: {inventory_path}")
        
        video_df = pd.read_csv(inventory_path)
        print(f"📊 Loaded {len(video_df)} videos from inventory")
        
        # Filter for selected classes
        class_filtered_df = video_df[video_df['class'].isin(self.selected_classes)]
        print(f"📊 Filtered to {len(class_filtered_df)} videos for 4 selected classes")
        
        # Create training manifest (combined demographics)
        training_videos = class_filtered_df[
            class_filtered_df['demographic_group'].isin(self.training_demographics)
        ]
        
        # Create validation manifest (single demographic)
        validation_videos = class_filtered_df[
            class_filtered_df['demographic_group'] == self.validation_demographic
        ]
        
        print(f"\n📊 DATASET DISTRIBUTION:")
        print(f"   Training Videos: {len(training_videos)}")
        print(f"   Validation Videos: {len(validation_videos)}")
        
        # Display class distribution
        print(f"\n📊 TRAINING CLASS DISTRIBUTION:")
        train_class_counts = training_videos['class'].value_counts()
        for class_name in self.selected_classes:
            count = train_class_counts.get(class_name, 0)
            print(f"   {class_name}: {count} videos")
        
        print(f"\n📊 VALIDATION CLASS DISTRIBUTION:")
        val_class_counts = validation_videos['class'].value_counts()
        for class_name in self.selected_classes:
            count = val_class_counts.get(class_name, 0)
            print(f"   {class_name}: {count} videos")
        
        # Verify demographic separation
        train_demographics = set(training_videos['demographic_group'].unique())
        val_demographics = set(validation_videos['demographic_group'].unique())
        overlap = train_demographics.intersection(val_demographics)
        
        if overlap:
            raise ValueError(f"❌ Demographic overlap detected: {overlap}")
        
        print(f"\n✅ ZERO DEMOGRAPHIC OVERLAP CONFIRMED:")
        print(f"   Training Demographics: {train_demographics}")
        print(f"   Validation Demographics: {val_demographics}")
        
        # Save manifests
        train_manifest_path = self.output_dir / "4class_train_manifest.csv"
        val_manifest_path = self.output_dir / "4class_validation_manifest.csv"
        
        training_videos.to_csv(train_manifest_path, index=False)
        validation_videos.to_csv(val_manifest_path, index=False)
        
        print(f"\n📄 Manifests saved:")
        print(f"   Training: {train_manifest_path}")
        print(f"   Validation: {val_manifest_path}")
        
        return train_manifest_path, val_manifest_path
    
    def load_4class_datasets(self):
        """Load 4-class datasets with enhanced augmentation."""
        print("\n📋 LOADING 4-CLASS DATASETS")
        print("=" * 50)
        
        # Create manifests
        train_manifest, val_manifest = self.create_4class_manifests()
        
        # Load datasets with proven augmentation from binary experiments
        self.train_dataset = FourClassLipReadingDataset(
            train_manifest, self.class_to_idx, augment=True, is_training=True
        )
        self.val_dataset = FourClassLipReadingDataset(
            val_manifest, self.class_to_idx, augment=False, is_training=False
        )
        
        print(f"📊 Training: {len(self.train_dataset)} videos (with 3x augmentation)")
        print(f"   Demographics: {self.train_dataset.get_demographics()}")
        print(f"   Classes: {self.train_dataset.get_class_distribution()}")
        
        print(f"📊 Validation: {len(self.val_dataset)} videos")
        print(f"   Demographics: {self.val_dataset.get_demographics()}")
        print(f"   Classes: {self.val_dataset.get_class_distribution()}")
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, 
            num_workers=0, drop_last=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        
        print(f"✅ 4-Class cross-demographic setup complete")
        
    def setup_4class_training(self):
        """Setup 4-class model and training components."""
        print("\n🏗️  SETTING UP 4-CLASS TRAINING SYSTEM")
        print("=" * 60)
        
        # Use proven architecture from binary experiments (only change output layer)
        self.model = FourClassBinaryModel().to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"📊 4-Class model parameters: {total_params:,}")
        
        # Identical optimizer configuration from binary experiments
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.initial_lr, 
            weight_decay=0.002,
            betas=(0.9, 0.999)
        )
        
        # Identical scheduler from binary experiments
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=15, T_mult=2, eta_min=0.0001
        )
        
        # Identical loss function from binary experiments
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Enhanced tracking for 4-class analysis
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.per_class_accuracies = []
        self.confusion_matrices = []
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        print(f"✅ 4-Class training setup complete:")
        print(f"   Architecture: Proven 2.98M parameter model (binary → 4-class)")
        print(f"   Optimizer: AdamW (lr={self.initial_lr}, weight_decay=0.002)")
        print(f"   Scheduler: CosineAnnealingWarmRestarts (T_0=15)")
        print(f"   Loss: CrossEntropyLoss with label smoothing (0.1)")
        
    def train_epoch_4class(self, epoch):
        """Training epoch for 4-class model."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (videos, labels) in enumerate(self.train_loader):
            videos, labels = videos.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(videos)
            loss = self.criterion(outputs, labels)
            
            # L2 regularization (same as binary experiments)
            l2_reg = sum(p.pow(2.0).sum() for p in self.model.parameters())
            loss = loss + 0.00005 * l2_reg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Progress logging
            if batch_idx % 8 == 0:  # Less frequent for 4-class
                acc = 100.0 * correct / total
                print(f"   Batch {batch_idx+1:2d}/{len(self.train_loader):2d} | "
                      f"Loss: {loss.item():.4f} | Acc: {acc:.1f}%")
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc
    
    def validate_epoch_4class(self):
        """Enhanced validation for 4-class with detailed metrics."""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_confidences = []
        total_loss = 0.0
        
        with torch.no_grad():
            for videos, labels in self.val_loader:
                videos, labels = videos.to(self.device), labels.to(self.device)
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                # Get predictions and confidence scores
                probabilities = F.softmax(outputs, dim=1)
                confidences, predicted = torch.max(probabilities, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
        
        # Overall accuracy
        correct = sum(p == l for p, l in zip(all_predictions, all_labels))
        overall_accuracy = 100.0 * correct / len(all_labels)
        
        # Per-class accuracy
        per_class_acc = {}
        for class_idx, class_name in enumerate(self.selected_classes):
            class_indices = [i for i, l in enumerate(all_labels) if l == class_idx]
            if class_indices:
                class_correct = sum(all_predictions[i] == all_labels[i] for i in class_indices)
                class_accuracy = 100.0 * class_correct / len(class_indices)
                per_class_acc[class_name] = class_accuracy
            else:
                per_class_acc[class_name] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Confidence statistics
        confidence_stats = {
            'mean_confidence': np.mean(all_confidences),
            'correct_confidence': np.mean([all_confidences[i] for i in range(len(all_confidences)) 
                                         if all_predictions[i] == all_labels[i]]),
            'incorrect_confidence': np.mean([all_confidences[i] for i in range(len(all_confidences)) 
                                           if all_predictions[i] != all_labels[i]]) if any(all_predictions[i] != all_labels[i] for i in range(len(all_predictions))) else 0.0
        }
        
        epoch_loss = total_loss / len(self.val_loader)
        
        return epoch_loss, overall_accuracy, per_class_acc, cm, confidence_stats

    def train_4class_model(self):
        """Execute 4-class training with comprehensive analysis."""
        print("\n🎯 STARTING 4-CLASS CROSS-DEMOGRAPHIC TRAINING")
        print("=" * 80)
        print(f"🚀 Goal: >{self.target_val_acc}% cross-demographic validation accuracy")
        print(f"📊 Scaling from binary (62.5%) to 4-class complexity")

        start_time = time.time()
        patience = 35  # Increased patience for 4-class complexity
        epochs_without_improvement = 0

        for epoch in range(1, self.max_epochs + 1):
            print(f"\n📅 Epoch {epoch:2d}/{self.max_epochs}")
            print("-" * 70)

            # Training
            train_loss, train_acc = self.train_epoch_4class(epoch)

            # Validation with detailed metrics
            val_loss, val_acc, per_class_acc, cm, conf_stats = self.validate_epoch_4class()

            # Update tracking
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.per_class_accuracies.append(per_class_acc)
            self.confusion_matrices.append(cm)

            # Check for improvement
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                epochs_without_improvement = 0

                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': self.best_val_acc,
                    'per_class_accuracies': per_class_acc,
                    'confusion_matrix': cm
                }, self.output_dir / 'best_4class_model.pth')

                print(f"   🎉 NEW BEST VALIDATION: {val_acc:.1f}%")
            else:
                epochs_without_improvement += 1

            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # Enhanced epoch summary
            print(f"\n📊 Epoch {epoch} 4-Class Summary:")
            print(f"   Train: {train_acc:.1f}% | Overall Val: {val_acc:.1f}%")
            print(f"   Best: {self.best_val_acc:.1f}% @ Epoch {self.best_epoch}")
            print(f"   Per-Class Validation:")
            for class_name, acc in per_class_acc.items():
                print(f"     {class_name}: {acc:.1f}%")
            print(f"   Confidence: Mean={conf_stats['mean_confidence']:.3f}, "
                  f"Correct={conf_stats['correct_confidence']:.3f}")
            print(f"   Learning Rate: {current_lr:.2e}")
            print(f"   Time: {time.time() - start_time:.1f}s")

            # Check success criteria
            min_class_acc = min(per_class_acc.values()) if per_class_acc else 0
            if (train_acc >= self.target_train_acc and
                val_acc >= self.target_val_acc and
                min_class_acc >= 40.0):  # No class below 40%

                print(f"\n🎉 4-CLASS SUCCESS CRITERIA ACHIEVED!")
                print(f"   ✅ Training: {train_acc:.1f}% ≥ {self.target_train_acc}%")
                print(f"   ✅ Cross-demographic validation: {val_acc:.1f}% ≥ {self.target_val_acc}%")
                print(f"   ✅ Minimum class accuracy: {min_class_acc:.1f}% ≥ 40%")
                print(f"   ✅ All classes above random chance (25%)")

                success = True
                break

            # Early stopping (adjusted for 4-class)
            if epochs_without_improvement >= patience and self.best_val_acc < 55.0:
                print(f"\n⏹️  Early stopping: Plateaued below 55% for {patience} epochs")
                success = False
                break
        else:
            # Check final success
            final_min_class_acc = min(self.per_class_accuracies[-1].values()) if self.per_class_accuracies else 0
            success = (max(self.train_accuracies) >= self.target_train_acc and
                      self.best_val_acc >= self.target_val_acc and
                      final_min_class_acc >= 40.0)

        # Generate comprehensive final report
        total_time = time.time() - start_time
        self.generate_4class_report(total_time, success)
        self.plot_4class_results()
        self.create_confusion_matrix_plot()

        return success

    def generate_4class_report(self, training_time, success):
        """Generate comprehensive 4-class training report."""
        final_train_acc = self.train_accuracies[-1] if self.train_accuracies else 0
        final_val_acc = self.val_accuracies[-1] if self.val_accuracies else 0

        # Get best epoch metrics
        best_per_class = self.per_class_accuracies[self.best_epoch - 1] if self.per_class_accuracies else {}
        best_cm = self.confusion_matrices[self.best_epoch - 1] if self.confusion_matrices else None

        # Calculate stability metrics
        last_10_val_accs = self.val_accuracies[-10:] if len(self.val_accuracies) >= 10 else self.val_accuracies
        val_stability = np.std(last_10_val_accs) if last_10_val_accs else 0

        print(f"\n🎯 4-CLASS CROSS-DEMOGRAPHIC TRAINING COMPLETED")
        print("=" * 80)
        print(f"📊 4-Class Results:")
        print(f"   Final Training: {final_train_acc:.1f}%")
        print(f"   Final Validation: {final_val_acc:.1f}%")
        print(f"   Best Validation: {self.best_val_acc:.1f}% @ Epoch {self.best_epoch}")
        print(f"   Validation Stability (σ): {val_stability:.2f}%")
        print(f"   Training Time: {training_time:.1f}s")
        print(f"   Total Epochs: {len(self.train_accuracies)}")

        print(f"\n📊 Best Epoch Per-Class Performance:")
        if best_per_class:
            for class_name, acc in best_per_class.items():
                print(f"   {class_name}: {acc:.1f}%")

            min_class_acc = min(best_per_class.values())
            max_class_acc = max(best_per_class.values())
            class_consistency = min_class_acc / max_class_acc if max_class_acc > 0 else 0
            print(f"   Class Consistency: {class_consistency:.3f} ({min_class_acc:.1f}% - {max_class_acc:.1f}%)")

        # Scaling analysis
        binary_baseline = 62.5  # From previous binary experiments
        scaling_factor = self.best_val_acc / binary_baseline if binary_baseline > 0 else 0
        print(f"\n📊 Scaling Analysis:")
        print(f"   Binary Baseline: {binary_baseline}%")
        print(f"   4-Class Performance: {self.best_val_acc:.1f}%")
        print(f"   Scaling Factor: {scaling_factor:.3f}")
        print(f"   Performance Change: {self.best_val_acc - binary_baseline:+.1f}%")

        # Success analysis
        if success:
            print(f"\n✅ 4-CLASS CROSS-DEMOGRAPHIC SUCCESS!")
            print(f"🎉 Achieved >{self.target_val_acc}% validation accuracy with 4-class complexity")
            print(f"🚀 Validated scaling from binary to multi-class cross-demographic training")

            if self.best_val_acc >= 75:
                print(f"   🌟 Exceptional 4-class performance: {self.best_val_acc:.1f}%")
            elif self.best_val_acc >= 70:
                print(f"   ✅ Target achieved: {self.best_val_acc:.1f}%")

            if val_stability < 5.0:
                print(f"   📈 Stable learning: σ={val_stability:.2f}%")

            if class_consistency >= 0.7:
                print(f"   📊 Good class balance: {class_consistency:.3f}")

        else:
            print(f"\n⚠️  4-Class target not fully achieved")
            print(f"💡 Best validation: {self.best_val_acc:.1f}%")

            if self.best_val_acc >= 60:
                print(f"   🔥 Strong progress: Significant improvement over random (25%)")
            elif self.best_val_acc >= 50:
                print(f"   📈 Good progress: Clear learning demonstrated")

        # Save detailed report
        report_path = self.output_dir / '4class_training_report.txt'
        with open(report_path, 'w') as f:
            f.write("4-CLASS CROSS-DEMOGRAPHIC TRAINING REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"PRIMARY GOAL: >{self.target_val_acc}% cross-demographic validation accuracy\n")
            f.write(f"CLASSES: {', '.join(self.selected_classes)}\n")
            f.write(f"SUCCESS: {'YES' if success else 'NO'}\n\n")
            f.write(f"FINAL RESULTS:\n")
            f.write(f"Best validation accuracy: {self.best_val_acc:.1f}%\n")
            f.write(f"Validation stability: {val_stability:.2f}%\n")
            f.write(f"Training time: {training_time:.1f}s\n")
            f.write(f"Total epochs: {len(self.train_accuracies)}\n\n")

            f.write("PER-CLASS PERFORMANCE (Best Epoch):\n")
            if best_per_class:
                for class_name, acc in best_per_class.items():
                    f.write(f"{class_name}: {acc:.1f}%\n")

            f.write(f"\nSCALING ANALYSIS:\n")
            f.write(f"Binary baseline: {binary_baseline}%\n")
            f.write(f"4-Class performance: {self.best_val_acc:.1f}%\n")
            f.write(f"Scaling factor: {scaling_factor:.3f}\n")

        print(f"📄 4-Class training report saved: {report_path}")
        return success

    def plot_4class_results(self):
        """Plot comprehensive 4-class training results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        epochs = range(1, len(self.train_losses) + 1)

        # Loss curves
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss', alpha=0.8)
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss', alpha=0.8)
        ax1.set_title('4-Class Training: Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy curves
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy', alpha=0.8)
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy', alpha=0.8)
        ax2.axhline(y=self.target_train_acc, color='b', linestyle='--', alpha=0.7,
                   label=f'Target Train ({self.target_train_acc}%)')
        ax2.axhline(y=self.target_val_acc, color='r', linestyle='--', alpha=0.7,
                   label=f'Target Val ({self.target_val_acc}%)')
        ax2.axhline(y=25, color='gray', linestyle=':', alpha=0.7, label='Random (25%)')
        ax2.set_title('4-Class Training: Accuracy Curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Per-class accuracy evolution
        if self.per_class_accuracies:
            for class_name in self.selected_classes:
                class_accs = [epoch_accs.get(class_name, 0) for epoch_accs in self.per_class_accuracies]
                ax3.plot(epochs, class_accs, label=class_name, alpha=0.8)
        ax3.axhline(y=40, color='gray', linestyle='--', alpha=0.7, label='Min Target (40%)')
        ax3.axhline(y=25, color='gray', linestyle=':', alpha=0.7, label='Random (25%)')
        ax3.set_title('4-Class Training: Per-Class Accuracy Evolution')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Validation stability analysis
        if len(self.val_accuracies) >= 10:
            window_size = 5
            rolling_std = []
            for i in range(window_size, len(self.val_accuracies) + 1):
                window_vals = self.val_accuracies[i-window_size:i]
                rolling_std.append(np.std(window_vals))

            rolling_epochs = range(window_size, len(self.val_accuracies) + 1)
            ax4.plot(rolling_epochs, rolling_std, 'g-', label='Rolling Std (5 epochs)', alpha=0.8)
            ax4.axhline(y=8, color='r', linestyle='--', alpha=0.7, label='Stability Target (8%)')
            ax4.set_title('4-Class Training: Validation Stability')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Standard Deviation (%)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / '4class_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 4-Class training curves saved: {self.output_dir / '4class_training_curves.png'}")

    def create_confusion_matrix_plot(self):
        """Create confusion matrix visualization for best epoch."""
        if not self.confusion_matrices:
            return

        best_cm = self.confusion_matrices[self.best_epoch - 1]

        plt.figure(figsize=(10, 8))
        sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.selected_classes,
                   yticklabels=self.selected_classes)
        plt.title(f'4-Class Confusion Matrix (Best Epoch {self.best_epoch})')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.tight_layout()
        plt.savefig(self.output_dir / '4class_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 4-Class confusion matrix saved: {self.output_dir / '4class_confusion_matrix.png'}")

    def run_4class_pipeline(self):
        """Execute complete 4-class training pipeline."""
        try:
            self.load_4class_datasets()
            self.setup_4class_training()
            success = self.train_4class_model()
            return success
        except Exception as e:
            print(f"\n❌ 4-CLASS TRAINING FAILED: {e}")
            raise

class FourClassLipReadingDataset(Dataset):
    """4-Class dataset with proven augmentation from binary experiments."""

    def __init__(self, manifest_path, class_to_idx, augment=False, is_training=False):
        self.class_to_idx = class_to_idx
        self.augment = augment
        self.is_training = is_training
        self.videos = []

        # Load from CSV manifest
        with open(manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['class'] in class_to_idx:
                    self.videos.append({
                        'path': row['video_path'],
                        'class': row['class'],
                        'class_idx': class_to_idx[row['class']],
                        'demographic_group': row['demographic_group']
                    })

    def __len__(self):
        # 3x augmentation for training (proven from binary experiments)
        return len(self.videos) * 3 if self.augment else len(self.videos)

    def __getitem__(self, idx):
        # Handle augmented indices
        video_idx = idx % len(self.videos)
        augment_type = idx // len(self.videos) if self.augment else 0

        video_info = self.videos[video_idx]
        frames = self._load_video_enhanced(video_info['path'])

        # Apply proven augmentation from binary experiments
        if self.augment and augment_type > 0:
            frames = self._apply_augmentation(frames, augment_type)

        # Identical preprocessing from binary experiments
        frames_tensor = torch.from_numpy(frames).float() / 255.0
        frames_tensor = frames_tensor.unsqueeze(0)  # Add channel

        return frames_tensor, video_info['class_idx']

    def _load_video_enhanced(self, video_path):
        """Identical video loading from binary experiments."""
        cap = cv2.VideoCapture(video_path)
        frames = []

        # Get total frame count for better sampling
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frames = 24  # Proven from binary experiments

        if total_frames > target_frames:
            frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        else:
            frame_indices = list(range(total_frames))

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(gray_frame, (64, 48))  # Proven resolution
            frames.append(resized_frame)

        cap.release()

        # Pad if needed
        while len(frames) < target_frames:
            frames.append(frames[-1] if frames else np.zeros((48, 64)))

        return np.array(frames[:target_frames])  # Shape: (24, 48, 64)

    def _apply_augmentation(self, frames, augment_type):
        """Identical augmentation from binary experiments."""
        augmented_frames = frames.copy()

        if augment_type == 1:
            # Brightness and contrast variation
            brightness_factor = np.random.uniform(0.85, 1.15)  # ±15%
            contrast_factor = np.random.uniform(0.9, 1.1)     # ±10%

            augmented_frames = augmented_frames.astype(np.float32)
            augmented_frames = augmented_frames * contrast_factor + (brightness_factor - 1) * 128
            augmented_frames = np.clip(augmented_frames, 0, 255).astype(np.uint8)

        elif augment_type == 2:
            # Horizontal flipping + slight brightness
            augmented_frames = np.flip(augmented_frames, axis=2)  # Flip width dimension
            brightness_factor = np.random.uniform(0.9, 1.1)
            augmented_frames = augmented_frames.astype(np.float32)
            augmented_frames = augmented_frames * brightness_factor
            augmented_frames = np.clip(augmented_frames, 0, 255).astype(np.uint8)

        return augmented_frames

    def get_demographics(self):
        demographics = defaultdict(int)
        for video in self.videos:
            demographics[video['demographic_group']] += 1
        return dict(demographics)

    def get_class_distribution(self):
        classes = defaultdict(int)
        for video in self.videos:
            classes[video['class']] += 1
        return dict(classes)

class FourClassBinaryModel(nn.Module):
    """4-Class model using proven binary architecture (only output layer changed)."""

    def __init__(self):
        super(FourClassBinaryModel, self).__init__()

        # IDENTICAL architecture from proven binary experiments
        self.conv3d1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=1)
        self.bn3d1 = nn.BatchNorm3d(32)
        self.pool3d1 = nn.MaxPool3d(kernel_size=(1, 2, 2))  # Spatial only

        self.conv3d2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1)
        self.bn3d2 = nn.BatchNorm3d(64)
        self.pool3d2 = nn.MaxPool3d(kernel_size=(2, 2, 2))  # Temporal + spatial

        self.conv3d3 = nn.Conv3d(64, 96, kernel_size=(3, 3, 3), padding=1)
        self.bn3d3 = nn.BatchNorm3d(96)
        self.pool3d3 = nn.MaxPool3d(kernel_size=(2, 2, 2))  # Temporal + spatial

        self.conv3d4 = nn.Conv3d(96, 128, kernel_size=(3, 3, 3), padding=1)
        self.bn3d4 = nn.BatchNorm3d(128)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((3, 3, 4))  # Adaptive pooling

        # Feature size: 128 * 3 * 3 * 4 = 4,608 (IDENTICAL to binary)
        self.feature_size = 128 * 3 * 3 * 4

        # IDENTICAL fully connected layers from binary experiments
        self.dropout1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)

        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)

        self.dropout3 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 32)

        # ONLY CHANGE: Output layer for 4 classes instead of 2
        self.fc_out = nn.Linear(32, 4)  # Changed from 2 to 4

        print(f"🏗️  4-Class Model (Based on Proven Binary Architecture):")
        print(f"   - Input: (B, 1, 24, 48, 64)")
        print(f"   - Features: {self.feature_size:,}")
        print(f"   - Architecture: IDENTICAL to binary (97.9% training accuracy)")
        print(f"   - Only Change: Output layer 2→4 classes")

    def forward(self, x):
        # IDENTICAL forward pass from proven binary experiments
        x = F.relu(self.bn3d1(self.conv3d1(x)))
        x = self.pool3d1(x)

        x = F.relu(self.bn3d2(self.conv3d2(x)))
        x = self.pool3d2(x)

        x = F.relu(self.bn3d3(self.conv3d3(x)))
        x = self.pool3d3(x)

        x = F.relu(self.bn3d4(self.conv3d4(x)))
        x = self.adaptive_pool(x)

        # Flatten and classify (IDENTICAL except final layer)
        x = x.view(x.size(0), -1)

        x = self.dropout1(x)
        x = F.relu(self.bn_fc1(self.fc1(x)))

        x = self.dropout2(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))

        x = self.dropout3(x)
        x = F.relu(self.fc3(x))
        x = self.fc_out(x)  # 4 classes instead of 2

        return x

def main():
    """Execute 4-class cross-demographic training pipeline."""
    print("🚀 STARTING 4-CLASS CROSS-DEMOGRAPHIC TRAINING PIPELINE")
    print("🎯 PRIMARY GOAL: Scale from binary (62.5%) to 4-class cross-demographic training")
    print("💡 Using proven 2.98M parameter architecture with validated training configuration")
    print("📊 Classes: my_mouth_is_dry, i_need_to_move, doctor, pillow")
    print("👥 Training: 65+ female Caucasian + 18-39 male not_specified")
    print("👤 Validation: 18-39 female Caucasian")

    trainer = FourClassCrossDemographicTrainer()
    success = trainer.run_4class_pipeline()

    if success:
        print("\n🎉 4-CLASS CROSS-DEMOGRAPHIC SUCCESS!")
        print(f"✅ Achieved >70% validation accuracy with 4-class complexity")
        print("🚀 Successfully scaled from binary to multi-class cross-demographic training")
        print("📊 Comprehensive analysis completed - ready for strategic assessment")
        print("\n⚠️  MANDATORY STOP: Awaiting user evaluation before any 7-class scaling")
    else:
        print("\n💡 4-Class training completed with valuable insights")
        print("🔍 Comprehensive analysis available for strategic decision making")
        print("📊 Performance data collected for scaling feasibility assessment")

if __name__ == "__main__":
    main()
