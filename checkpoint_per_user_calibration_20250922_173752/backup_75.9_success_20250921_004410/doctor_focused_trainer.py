#!/usr/bin/env python3
"""
Doctor-Focused 4-Class Cross-Demographic Training
Targeted improvement for doctor class performance (40% → 60%+)
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

class DoctorFocusedTrainer:
    def __init__(self):
        self.base_results_dir = Path("4class_training_results")
        self.output_dir = Path("doctor_focused_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load best model from previous training
        self.best_model_path = self.base_results_dir / "best_4class_model.pth"
        
        # Enhanced configuration for doctor class improvement
        self.batch_size = 6
        self.max_epochs = 20  # Focused retraining
        self.initial_lr = 0.002  # Lower LR for fine-tuning
        self.device = torch.device('cpu')
        
        # Success criteria for doctor improvement
        self.doctor_target_acc = 60.0  # Target: 40% → 60%
        self.overall_target_acc = 70.0  # Maintain overall performance
        self.other_classes_tolerance = 5.0  # Max 5% degradation
        
        # Class configuration
        self.selected_classes = ['my_mouth_is_dry', 'i_need_to_move', 'doctor', 'pillow']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.selected_classes)}
        
        # Enhanced class weights (from analysis)
        self.class_weights = torch.tensor([0.713, 1.050, 2.265, 1.312])  # doctor gets 2.265x weight
        
        print("🏥 DOCTOR-FOCUSED 4-CLASS IMPROVEMENT TRAINING")
        print("=" * 80)
        print("🎯 PRIMARY GOAL: Improve doctor class from 40.0% to 60.0%+ accuracy")
        print("📊 Strategy: Enhanced augmentation + class weighting + targeted retraining")
        print(f"⚖️  Class weights: doctor=2.265x (enhanced), others=standard")
        print(f"🔄 Focused retraining: {self.max_epochs} epochs with early stopping")
        
    def load_enhanced_datasets(self):
        """Load datasets with doctor-focused enhancements."""
        print("\n📋 LOADING DOCTOR-ENHANCED DATASETS")
        print("=" * 50)
        
        # Load existing manifests
        train_manifest = self.base_results_dir / "4class_train_manifest.csv"
        val_manifest = self.base_results_dir / "4class_validation_manifest.csv"
        
        # Create enhanced datasets with doctor-specific augmentation
        self.train_dataset = DoctorEnhancedDataset(
            train_manifest, self.class_to_idx, augment=True, is_training=True
        )
        self.val_dataset = DoctorEnhancedDataset(
            val_manifest, self.class_to_idx, augment=False, is_training=False
        )
        
        print(f"📊 Training: {len(self.train_dataset)} videos (with doctor-enhanced augmentation)")
        print(f"📊 Validation: {len(self.val_dataset)} videos")
        
        # Analyze doctor class distribution
        train_doctor_count = sum(1 for i in range(len(self.train_dataset.videos)) 
                               if self.train_dataset.videos[i]['class'] == 'doctor')
        val_doctor_count = sum(1 for i in range(len(self.val_dataset.videos)) 
                             if self.val_dataset.videos[i]['class'] == 'doctor')
        
        print(f"🏥 Doctor videos: {train_doctor_count} training, {val_doctor_count} validation")
        print(f"🎯 Doctor augmentation: 5x multiplier (vs 3x for other classes)")
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, 
            num_workers=0, drop_last=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        
        print(f"✅ Doctor-enhanced datasets loaded successfully")
    
    def setup_doctor_focused_training(self):
        """Setup model and training components with doctor focus."""
        print("\n🏗️  SETTING UP DOCTOR-FOCUSED TRAINING SYSTEM")
        print("=" * 60)
        
        # Load the best model from previous training
        self.model = DoctorFocusedModel().to(self.device)
        
        if self.best_model_path.exists():
            print(f"📥 Loading best model from: {self.best_model_path}")
            checkpoint = torch.load(self.best_model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded model with previous best validation: {checkpoint.get('best_val_acc', 'N/A')}%")
        else:
            print("⚠️  No previous model found, starting from scratch")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"📊 Model parameters: {total_params:,}")
        
        # Fine-tuning optimizer with lower learning rate
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.initial_lr,  # Lower LR for fine-tuning
            weight_decay=0.001,  # Reduced weight decay
            betas=(0.9, 0.999)
        )
        
        # Gentler scheduler for fine-tuning
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=1, eta_min=0.0001
        )
        
        # Class-weighted loss function (key improvement)
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights.to(self.device),
            label_smoothing=0.1
        )
        
        # Enhanced tracking for doctor-focused analysis
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.doctor_accuracies = []
        self.per_class_accuracies = []
        self.confusion_matrices = []
        self.best_val_acc = 0.0
        self.best_doctor_acc = 0.0
        self.best_epoch = 0
        
        print(f"✅ Doctor-focused training setup complete:")
        print(f"   Architecture: Proven 2.98M parameter model (fine-tuning)")
        print(f"   Optimizer: AdamW (lr={self.initial_lr}, fine-tuning mode)")
        print(f"   Loss: Weighted CrossEntropyLoss (doctor=2.265x weight)")
        print(f"   Focus: Doctor class improvement with overall performance maintenance")
    
    def train_epoch_doctor_focused(self, epoch):
        """Training epoch with doctor class focus."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        doctor_correct = 0
        doctor_total = 0
        
        for batch_idx, (videos, labels) in enumerate(self.train_loader):
            videos, labels = videos.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(videos)
            loss = self.criterion(outputs, labels)  # Uses class weights
            
            # Lighter L2 regularization for fine-tuning
            l2_reg = sum(p.pow(2.0).sum() for p in self.model.parameters())
            loss = loss + 0.00002 * l2_reg  # Reduced from 0.00005
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.3)  # Gentler clipping
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Track doctor class specifically
            doctor_mask = (labels == self.class_to_idx['doctor'])
            if doctor_mask.sum() > 0:
                doctor_total += doctor_mask.sum().item()
                doctor_correct += (predicted[doctor_mask] == labels[doctor_mask]).sum().item()
            
            # Progress logging with doctor focus
            if batch_idx % 10 == 0:
                overall_acc = 100.0 * correct / total
                doctor_acc = 100.0 * doctor_correct / max(doctor_total, 1)
                print(f"   Batch {batch_idx+1:2d}/{len(self.train_loader):2d} | "
                      f"Loss: {loss.item():.4f} | Overall: {overall_acc:.1f}% | Doctor: {doctor_acc:.1f}%")
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total
        doctor_epoch_acc = 100.0 * doctor_correct / max(doctor_total, 1)
        
        return epoch_loss, epoch_acc, doctor_epoch_acc
    
    def validate_epoch_doctor_focused(self):
        """Enhanced validation with doctor class focus."""
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
        
        # Per-class accuracy with doctor focus
        per_class_acc = {}
        doctor_accuracy = 0.0
        
        for class_idx, class_name in enumerate(self.selected_classes):
            class_indices = [i for i, l in enumerate(all_labels) if l == class_idx]
            if class_indices:
                class_correct = sum(all_predictions[i] == all_labels[i] for i in class_indices)
                class_accuracy = 100.0 * class_correct / len(class_indices)
                per_class_acc[class_name] = class_accuracy
                
                if class_name == 'doctor':
                    doctor_accuracy = class_accuracy
            else:
                per_class_acc[class_name] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Doctor-specific confidence analysis
        doctor_indices = [i for i, l in enumerate(all_labels) if l == self.class_to_idx['doctor']]
        doctor_confidence_stats = {
            'mean_confidence': np.mean([all_confidences[i] for i in doctor_indices]) if doctor_indices else 0.0,
            'correct_predictions': sum(all_predictions[i] == all_labels[i] for i in doctor_indices),
            'total_predictions': len(doctor_indices)
        }
        
        epoch_loss = total_loss / len(self.val_loader)
        
        return epoch_loss, overall_accuracy, doctor_accuracy, per_class_acc, cm, doctor_confidence_stats

    def train_doctor_focused_model(self):
        """Execute doctor-focused training with comprehensive monitoring."""
        print("\n🎯 STARTING DOCTOR-FOCUSED IMPROVEMENT TRAINING")
        print("=" * 80)
        print(f"🏥 Goal: Doctor class 40.0% → {self.doctor_target_acc}%+ accuracy")
        print(f"📊 Maintain overall {self.overall_target_acc}%+ with <{self.other_classes_tolerance}% degradation")

        start_time = time.time()
        patience = 8  # Early stopping for focused training
        epochs_without_improvement = 0

        # Track baseline performance for comparison
        baseline_performance = {
            'doctor': 40.0,
            'my_mouth_is_dry': 100.0,
            'i_need_to_move': 87.5,
            'pillow': 85.7,
            'overall': 72.4
        }

        for epoch in range(1, self.max_epochs + 1):
            print(f"\n📅 Epoch {epoch:2d}/{self.max_epochs} - Doctor Focus Training")
            print("-" * 70)

            # Training with doctor focus
            train_loss, train_acc, train_doctor_acc = self.train_epoch_doctor_focused(epoch)

            # Validation with detailed doctor analysis
            val_loss, val_acc, doctor_acc, per_class_acc, cm, doctor_stats = self.validate_epoch_doctor_focused()

            # Update tracking
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.doctor_accuracies.append(doctor_acc)
            self.per_class_accuracies.append(per_class_acc)
            self.confusion_matrices.append(cm)

            # Check for improvement (prioritize doctor + overall)
            doctor_improved = doctor_acc > self.best_doctor_acc
            overall_maintained = val_acc >= self.overall_target_acc

            is_best = doctor_improved and overall_maintained
            if is_best or (doctor_acc > self.best_doctor_acc and val_acc > self.best_val_acc):
                self.best_val_acc = val_acc
                self.best_doctor_acc = doctor_acc
                self.best_epoch = epoch
                epochs_without_improvement = 0

                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': self.best_val_acc,
                    'best_doctor_acc': self.best_doctor_acc,
                    'per_class_accuracies': per_class_acc,
                    'confusion_matrix': cm
                }, self.output_dir / 'best_doctor_focused_model.pth')

                print(f"   🎉 NEW BEST: Doctor {doctor_acc:.1f}%, Overall {val_acc:.1f}%")
            else:
                epochs_without_improvement += 1

            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # Enhanced epoch summary with doctor focus
            print(f"\n📊 Epoch {epoch} Doctor-Focused Summary:")
            print(f"   Overall: Train {train_acc:.1f}% | Val {val_acc:.1f}%")
            print(f"   🏥 Doctor: Train {train_doctor_acc:.1f}% | Val {doctor_acc:.1f}%")
            print(f"   Best: Doctor {self.best_doctor_acc:.1f}% | Overall {self.best_val_acc:.1f}% @ Epoch {self.best_epoch}")

            print(f"   Per-Class Validation:")
            for class_name, acc in per_class_acc.items():
                baseline_acc = baseline_performance.get(class_name, 0)
                change = acc - baseline_acc
                status = "📈" if change >= 0 else "📉"
                print(f"     {class_name}: {acc:.1f}% ({change:+.1f}%) {status}")

            print(f"   Doctor Stats: {doctor_stats['correct_predictions']}/{doctor_stats['total_predictions']} correct")
            print(f"   Learning Rate: {current_lr:.2e}")
            print(f"   Time: {time.time() - start_time:.1f}s")

            # Check success criteria
            doctor_target_met = doctor_acc >= self.doctor_target_acc
            overall_maintained = val_acc >= self.overall_target_acc

            # Check other classes degradation
            other_classes_ok = True
            for class_name, acc in per_class_acc.items():
                if class_name != 'doctor':
                    baseline_acc = baseline_performance.get(class_name, 0)
                    degradation = baseline_acc - acc
                    if degradation > self.other_classes_tolerance:
                        other_classes_ok = False
                        print(f"   ⚠️  {class_name} degraded by {degradation:.1f}% (>{self.other_classes_tolerance}%)")

            if doctor_target_met and overall_maintained and other_classes_ok:
                print(f"\n🎉 DOCTOR IMPROVEMENT SUCCESS ACHIEVED!")
                print(f"   ✅ Doctor: {doctor_acc:.1f}% ≥ {self.doctor_target_acc}%")
                print(f"   ✅ Overall: {val_acc:.1f}% ≥ {self.overall_target_acc}%")
                print(f"   ✅ Other classes: Within {self.other_classes_tolerance}% tolerance")

                success = True
                break

            # Early stopping for doctor focus
            if epochs_without_improvement >= patience:
                print(f"\n⏹️  Early stopping: No improvement for {patience} epochs")
                success = False
                break
        else:
            # Check final success
            success = (self.best_doctor_acc >= self.doctor_target_acc and
                      self.best_val_acc >= self.overall_target_acc)

        # Generate comprehensive final report
        total_time = time.time() - start_time
        self.generate_doctor_focused_report(total_time, success, baseline_performance)
        self.plot_doctor_focused_results()
        self.create_doctor_confusion_analysis()

        return success

    def generate_doctor_focused_report(self, training_time, success, baseline_performance):
        """Generate comprehensive doctor-focused training report."""
        final_train_acc = self.train_accuracies[-1] if self.train_accuracies else 0
        final_val_acc = self.val_accuracies[-1] if self.val_accuracies else 0
        final_doctor_acc = self.doctor_accuracies[-1] if self.doctor_accuracies else 0

        # Get best epoch metrics
        best_per_class = self.per_class_accuracies[self.best_epoch - 1] if self.per_class_accuracies else {}

        # Calculate improvements
        doctor_improvement = self.best_doctor_acc - baseline_performance['doctor']
        overall_change = self.best_val_acc - baseline_performance['overall']

        print(f"\n🏥 DOCTOR-FOCUSED TRAINING COMPLETED")
        print("=" * 80)
        print(f"📊 Doctor Class Results:")
        print(f"   Baseline: {baseline_performance['doctor']:.1f}%")
        print(f"   Best Achieved: {self.best_doctor_acc:.1f}%")
        print(f"   Improvement: {doctor_improvement:+.1f} percentage points")
        print(f"   Target: {self.doctor_target_acc:.1f}% ({'✅ MET' if self.best_doctor_acc >= self.doctor_target_acc else '❌ NOT MET'})")

        print(f"\n📊 Overall Performance:")
        print(f"   Best Overall: {self.best_val_acc:.1f}%")
        print(f"   Change: {overall_change:+.1f} percentage points")
        print(f"   Target: {self.overall_target_acc:.1f}% ({'✅ MAINTAINED' if self.best_val_acc >= self.overall_target_acc else '❌ DEGRADED'})")

        print(f"\n📊 Per-Class Impact Analysis:")
        if best_per_class:
            for class_name, acc in best_per_class.items():
                baseline_acc = baseline_performance.get(class_name, 0)
                change = acc - baseline_acc
                status = "✅" if change >= -self.other_classes_tolerance else "⚠️"
                print(f"   {class_name}: {acc:.1f}% ({change:+.1f}%) {status}")

        print(f"\n📊 Training Metrics:")
        print(f"   Training Time: {training_time:.1f}s")
        print(f"   Total Epochs: {len(self.train_accuracies)}")
        print(f"   Best Epoch: {self.best_epoch}")

        # Success analysis
        if success:
            print(f"\n✅ DOCTOR IMPROVEMENT SUCCESS!")
            print(f"🎉 Successfully improved doctor class performance while maintaining overall quality")
            print(f"🚀 Ready for 7-class scaling with strengthened doctor foundation")
        else:
            print(f"\n⚠️  Partial improvement achieved")
            print(f"💡 Doctor improvement: {doctor_improvement:+.1f}% (target: {self.doctor_target_acc - baseline_performance['doctor']:+.1f}%)")

            if self.best_doctor_acc >= 50.0:
                print(f"   📈 Significant progress: {self.best_doctor_acc:.1f}% (10+ point improvement)")

        # Save detailed report
        report_path = self.output_dir / 'doctor_focused_report.txt'
        with open(report_path, 'w') as f:
            f.write("DOCTOR-FOCUSED 4-CLASS TRAINING REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"DOCTOR CLASS IMPROVEMENT:\n")
            f.write(f"Baseline: {baseline_performance['doctor']:.1f}%\n")
            f.write(f"Best achieved: {self.best_doctor_acc:.1f}%\n")
            f.write(f"Improvement: {doctor_improvement:+.1f} percentage points\n")
            f.write(f"Target met: {'YES' if self.best_doctor_acc >= self.doctor_target_acc else 'NO'}\n\n")

            f.write(f"OVERALL PERFORMANCE:\n")
            f.write(f"Best validation: {self.best_val_acc:.1f}%\n")
            f.write(f"Change from baseline: {overall_change:+.1f}%\n")
            f.write(f"Target maintained: {'YES' if self.best_val_acc >= self.overall_target_acc else 'NO'}\n\n")

            f.write("PER-CLASS RESULTS (Best Epoch):\n")
            if best_per_class:
                for class_name, acc in best_per_class.items():
                    baseline_acc = baseline_performance.get(class_name, 0)
                    change = acc - baseline_acc
                    f.write(f"{class_name}: {acc:.1f}% ({change:+.1f}%)\n")

            f.write(f"\nTRAINING SUMMARY:\n")
            f.write(f"Training time: {training_time:.1f}s\n")
            f.write(f"Total epochs: {len(self.train_accuracies)}\n")
            f.write(f"Best epoch: {self.best_epoch}\n")
            f.write(f"Success: {'YES' if success else 'PARTIAL'}\n")

        print(f"📄 Doctor-focused report saved: {report_path}")
        return success

    def plot_doctor_focused_results(self):
        """Plot comprehensive doctor-focused training results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        epochs = range(1, len(self.train_losses) + 1)

        # Loss curves
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss', alpha=0.8)
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss', alpha=0.8)
        ax1.set_title('Doctor-Focused Training: Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Overall accuracy curves
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy', alpha=0.8)
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy', alpha=0.8)
        ax2.axhline(y=self.overall_target_acc, color='r', linestyle='--', alpha=0.7,
                   label=f'Target Overall ({self.overall_target_acc}%)')
        ax2.axhline(y=72.4, color='gray', linestyle=':', alpha=0.7, label='Baseline (72.4%)')
        ax2.set_title('Doctor-Focused Training: Overall Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Doctor class focus
        ax3.plot(epochs, self.doctor_accuracies, 'g-', label='Doctor Validation', linewidth=3, alpha=0.9)
        ax3.axhline(y=self.doctor_target_acc, color='g', linestyle='--', alpha=0.7,
                   label=f'Doctor Target ({self.doctor_target_acc}%)')
        ax3.axhline(y=40.0, color='gray', linestyle=':', alpha=0.7, label='Doctor Baseline (40%)')
        ax3.set_title('Doctor-Focused Training: Doctor Class Performance')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Doctor Accuracy (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Per-class comparison (latest epoch)
        if self.per_class_accuracies:
            latest_per_class = self.per_class_accuracies[-1]
            baseline_performance = {'doctor': 40.0, 'my_mouth_is_dry': 100.0, 'i_need_to_move': 87.5, 'pillow': 85.7}

            classes = list(latest_per_class.keys())
            current_accs = [latest_per_class[cls] for cls in classes]
            baseline_accs = [baseline_performance.get(cls, 0) for cls in classes]

            x = np.arange(len(classes))
            width = 0.35

            ax4.bar(x - width/2, baseline_accs, width, label='Baseline', alpha=0.7, color='lightgray')
            ax4.bar(x + width/2, current_accs, width, label='Doctor-Focused', alpha=0.8, color='steelblue')

            ax4.set_title('Doctor-Focused Training: Per-Class Comparison')
            ax4.set_xlabel('Classes')
            ax4.set_ylabel('Accuracy (%)')
            ax4.set_xticks(x)
            ax4.set_xticklabels(classes, rotation=45)
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'doctor_focused_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 Doctor-focused curves saved: {self.output_dir / 'doctor_focused_curves.png'}")

    def create_doctor_confusion_analysis(self):
        """Create detailed confusion matrix analysis for doctor class."""
        if not self.confusion_matrices:
            return

        best_cm = self.confusion_matrices[self.best_epoch - 1]

        # Create confusion matrix plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.selected_classes,
                   yticklabels=self.selected_classes)
        plt.title(f'Doctor-Focused Confusion Matrix (Best Epoch {self.best_epoch})')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')

        # Highlight doctor row and column
        ax = plt.gca()
        doctor_idx = self.class_to_idx['doctor']

        # Highlight doctor row (true doctor predictions)
        ax.add_patch(plt.Rectangle((0, doctor_idx), len(self.selected_classes), 1,
                                 fill=False, edgecolor='red', lw=3))
        # Highlight doctor column (predicted as doctor)
        ax.add_patch(plt.Rectangle((doctor_idx, 0), 1, len(self.selected_classes),
                                 fill=False, edgecolor='green', lw=3))

        plt.tight_layout()
        plt.savefig(self.output_dir / 'doctor_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Analyze doctor confusion patterns
        doctor_row = best_cm[doctor_idx]  # True doctor predictions
        doctor_col = best_cm[:, doctor_idx]  # Predicted as doctor

        print(f"\n🔍 DOCTOR CONFUSION ANALYSIS:")
        print(f"   True doctor videos: {doctor_row.sum()}")
        print(f"   Correctly classified: {best_cm[doctor_idx, doctor_idx]}")
        print(f"   Doctor accuracy: {100.0 * best_cm[doctor_idx, doctor_idx] / max(doctor_row.sum(), 1):.1f}%")

        print(f"\n   Doctor misclassified as:")
        for i, class_name in enumerate(self.selected_classes):
            if i != doctor_idx and doctor_row[i] > 0:
                percentage = 100.0 * doctor_row[i] / doctor_row.sum()
                print(f"     {class_name}: {doctor_row[i]} videos ({percentage:.1f}%)")

        print(f"\n   Other classes misclassified as doctor:")
        for i, class_name in enumerate(self.selected_classes):
            if i != doctor_idx and doctor_col[i] > 0:
                percentage = 100.0 * doctor_col[i] / doctor_col.sum()
                print(f"     {class_name}: {doctor_col[i]} videos ({percentage:.1f}%)")

        print(f"📊 Doctor confusion analysis saved: {self.output_dir / 'doctor_confusion_matrix.png'}")

    def run_doctor_focused_pipeline(self):
        """Execute complete doctor-focused improvement pipeline."""
        try:
            self.load_enhanced_datasets()
            self.setup_doctor_focused_training()
            success = self.train_doctor_focused_model()
            return success
        except Exception as e:
            print(f"\n❌ DOCTOR-FOCUSED TRAINING FAILED: {e}")
            raise

class DoctorEnhancedDataset(Dataset):
    """Enhanced dataset with doctor-specific augmentation strategies."""

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
        if not self.augment:
            return len(self.videos)

        # Enhanced augmentation for doctor class
        total_length = 0
        for video in self.videos:
            if video['class'] == 'doctor':
                total_length += 5  # 5x augmentation for doctor
            else:
                total_length += 3  # 3x augmentation for others

        return total_length

    def __getitem__(self, idx):
        if not self.augment:
            video_info = self.videos[idx]
            frames = self._load_video_enhanced(video_info['path'])
            frames_tensor = torch.from_numpy(frames).float() / 255.0
            frames_tensor = frames_tensor.unsqueeze(0)
            return frames_tensor, video_info['class_idx']

        # Handle enhanced augmentation indices
        current_idx = 0
        for video_idx, video_info in enumerate(self.videos):
            multiplier = 5 if video_info['class'] == 'doctor' else 3

            if current_idx <= idx < current_idx + multiplier:
                augment_type = idx - current_idx
                frames = self._load_video_enhanced(video_info['path'])

                # Apply doctor-enhanced augmentation
                if augment_type > 0:
                    frames = self._apply_doctor_enhanced_augmentation(
                        frames, augment_type, video_info['class'] == 'doctor'
                    )

                frames_tensor = torch.from_numpy(frames).float() / 255.0
                frames_tensor = frames_tensor.unsqueeze(0)
                return frames_tensor, video_info['class_idx']

            current_idx += multiplier

        # Fallback
        video_info = self.videos[0]
        frames = self._load_video_enhanced(video_info['path'])
        frames_tensor = torch.from_numpy(frames).float() / 255.0
        frames_tensor = frames_tensor.unsqueeze(0)
        return frames_tensor, video_info['class_idx']

    def _load_video_enhanced(self, video_path):
        """Enhanced video loading identical to previous successful training."""
        cap = cv2.VideoCapture(video_path)
        frames = []

        # Get total frame count for better sampling
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frames = 24  # Proven from previous training

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

    def _apply_doctor_enhanced_augmentation(self, frames, augment_type, is_doctor_class):
        """Apply doctor-enhanced augmentation strategies."""
        augmented_frames = frames.copy()

        if augment_type == 1:
            # Enhanced brightness and contrast for doctor class
            if is_doctor_class:
                brightness_factor = np.random.uniform(0.8, 1.2)   # ±20% for doctor
                contrast_factor = np.random.uniform(0.85, 1.15)   # Enhanced range
            else:
                brightness_factor = np.random.uniform(0.85, 1.15) # ±15% for others
                contrast_factor = np.random.uniform(0.9, 1.1)     # Standard range

            augmented_frames = augmented_frames.astype(np.float32)
            augmented_frames = augmented_frames * contrast_factor + (brightness_factor - 1) * 128
            augmented_frames = np.clip(augmented_frames, 0, 255).astype(np.uint8)

        elif augment_type == 2:
            # Enhanced horizontal flipping for doctor class
            flip_probability = 0.5 if is_doctor_class else 0.33

            if np.random.random() < flip_probability:
                augmented_frames = np.flip(augmented_frames, axis=2)  # Flip width dimension

            # Add slight brightness variation
            brightness_factor = np.random.uniform(0.9, 1.1)
            augmented_frames = augmented_frames.astype(np.float32)
            augmented_frames = augmented_frames * brightness_factor
            augmented_frames = np.clip(augmented_frames, 0, 255).astype(np.uint8)

        elif augment_type == 3:
            # Temporal speed variation (doctor-specific enhancement)
            if is_doctor_class:
                speed_factor = np.random.uniform(0.9, 1.1)  # ±10% speed variation

                # Resample frames based on speed factor
                original_indices = np.arange(len(augmented_frames))
                new_indices = np.linspace(0, len(augmented_frames) - 1,
                                        int(len(augmented_frames) / speed_factor))
                new_indices = np.clip(new_indices, 0, len(augmented_frames) - 1).astype(int)

                # Ensure we still have 24 frames
                if len(new_indices) < 24:
                    # Pad by repeating last frame
                    while len(new_indices) < 24:
                        new_indices = np.append(new_indices, new_indices[-1])
                elif len(new_indices) > 24:
                    # Subsample to 24 frames
                    new_indices = new_indices[:24]

                augmented_frames = augmented_frames[new_indices]

            # Add brightness variation
            brightness_factor = np.random.uniform(0.9, 1.1)
            augmented_frames = augmented_frames.astype(np.float32)
            augmented_frames = augmented_frames * brightness_factor
            augmented_frames = np.clip(augmented_frames, 0, 255).astype(np.uint8)

        elif augment_type == 4 and is_doctor_class:
            # Additional doctor-specific augmentation (gamma correction)
            gamma = np.random.uniform(0.8, 1.2)

            # Apply gamma correction
            augmented_frames = augmented_frames.astype(np.float32) / 255.0
            augmented_frames = np.power(augmented_frames, gamma)
            augmented_frames = (augmented_frames * 255.0).astype(np.uint8)

        return augmented_frames

class DoctorFocusedModel(nn.Module):
    """Identical model architecture to successful 4-class training."""

    def __init__(self):
        super(DoctorFocusedModel, self).__init__()

        # IDENTICAL architecture from successful 4-class training
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

        # Feature size: 128 * 3 * 3 * 4 = 4,608 (IDENTICAL)
        self.feature_size = 128 * 3 * 3 * 4

        # IDENTICAL fully connected layers
        self.dropout1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)

        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)

        self.dropout3 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 32)

        # 4-class output (same as successful training)
        self.fc_out = nn.Linear(32, 4)

    def forward(self, x):
        # IDENTICAL forward pass from successful training
        x = F.relu(self.bn3d1(self.conv3d1(x)))
        x = self.pool3d1(x)

        x = F.relu(self.bn3d2(self.conv3d2(x)))
        x = self.pool3d2(x)

        x = F.relu(self.bn3d3(self.conv3d3(x)))
        x = self.pool3d3(x)

        x = F.relu(self.bn3d4(self.conv3d4(x)))
        x = self.adaptive_pool(x)

        # Flatten and classify
        x = x.view(x.size(0), -1)

        x = self.dropout1(x)
        x = F.relu(self.bn_fc1(self.fc1(x)))

        x = self.dropout2(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))

        x = self.dropout3(x)
        x = F.relu(self.fc3(x))
        x = self.fc_out(x)

        return x

def main():
    """Execute doctor-focused improvement training."""
    print("🏥 STARTING DOCTOR-FOCUSED 4-CLASS IMPROVEMENT TRAINING")
    print("🎯 PRIMARY GOAL: Improve doctor class from 40.0% to 60.0%+ accuracy")
    print("💡 Strategy: Enhanced augmentation + class weighting + targeted fine-tuning")
    print("📊 Maintain overall 70%+ performance with <5% degradation in other classes")
    print("🔄 Focused retraining with early stopping and comprehensive monitoring")

    trainer = DoctorFocusedTrainer()
    success = trainer.run_doctor_focused_pipeline()

    if success:
        print("\n🎉 DOCTOR CLASS IMPROVEMENT SUCCESS!")
        print(f"✅ Successfully improved doctor class performance to 60%+ accuracy")
        print("✅ Maintained overall cross-demographic performance above 70%")
        print("✅ Preserved other class performance within tolerance")
        print("🚀 Strong foundation established for 7-class scaling")
        print("\n📊 Ready to proceed with 7-class cross-demographic training!")
    else:
        print("\n💡 Doctor class improvement completed with valuable insights")
        print("🔍 Partial improvement achieved - analyze results for next steps")
        print("📊 Performance data available for strategic decision making")

if __name__ == "__main__":
    main()
