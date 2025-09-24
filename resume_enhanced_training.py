#!/usr/bin/env python3
"""
Resume Training from 75.9% Checkpoint with Balanced Sampling
Target: 82% Cross-Demographic Validation Accuracy

This script loads the successful 75.9% validation accuracy checkpoint from epoch 141
and resumes training with balanced dataset configuration to target 82% validation accuracy.
"""

import os
import csv
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from pathlib import Path
import time
import random
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
from datetime import datetime

class ResumedBalancedTrainer:
    def __init__(self):
        # Checkpoint configuration
        self.checkpoint_dir = Path("backup_75.9_success_20250921_004410")
        self.checkpoint_path = self.checkpoint_dir / "best_4class_model.pth"
        self.train_manifest = self.checkpoint_dir / "4class_train_manifest.csv"
        self.val_manifest = self.checkpoint_dir / "4class_validation_manifest.csv"
        
        # Output directory for resumed training
        self.output_dir = Path("resumed_balanced_training_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training configuration - targeting 82% validation accuracy
        self.batch_size = 8  # Increased from 6 for better gradient estimates
        self.max_epochs = 60  # Extended training for 82% target
        self.initial_lr = 0.0005  # Reduced from 0.002 for fine-tuning
        self.device = torch.device('cpu')
        
        # Target configuration
        self.target_val_acc = 82.0  # Primary goal
        self.early_stopping_patience = 20  # Increased patience
        self.min_improvement = 0.1  # Minimum improvement threshold
        
        # Class configuration
        self.selected_classes = ['my_mouth_is_dry', 'i_need_to_move', 'doctor', 'pillow']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.selected_classes)}
        
        # Balanced sampling configuration
        self.use_weighted_sampler = True
        self.balance_strategy = "weighted_random"  # or "duplicate_balancing"
        
        print("🎯 RESUMING TRAINING FROM 75.9% CHECKPOINT")
        print("=" * 80)
        print(f"📊 Target: {self.target_val_acc}% cross-demographic validation accuracy")
        print(f"🔄 Strategy: Balanced sampling + extended training")
        print(f"📈 Max epochs: {self.max_epochs} with early stopping patience: {self.early_stopping_patience}")
        print(f"⚖️  Balance strategy: {self.balance_strategy}")
        
    def load_checkpoint_and_datasets(self):
        """Load the 75.9% checkpoint and prepare balanced datasets."""
        print("\n📋 LOADING CHECKPOINT AND BALANCED DATASETS")
        print("=" * 60)
        
        # Verify checkpoint exists
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Load checkpoint
        print(f"📥 Loading checkpoint: {self.checkpoint_path}")
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        print(f"✅ Loaded checkpoint with validation accuracy: {self.checkpoint.get('best_val_acc', 'N/A')}%")
        
        # Create balanced datasets
        self.train_dataset = BalancedLipReadingDataset(
            self.train_manifest, self.class_to_idx, augment=True, is_training=True
        )
        self.val_dataset = BalancedLipReadingDataset(
            self.val_manifest, self.class_to_idx, augment=False, is_training=False
        )
        
        print(f"📊 Training: {len(self.train_dataset)} videos")
        print(f"📊 Validation: {len(self.val_dataset)} videos")
        
        # Analyze class distribution
        self.analyze_class_distribution()
        
        # Create balanced data loaders
        self.create_balanced_data_loaders()
        
        print(f"✅ Checkpoint and balanced datasets loaded successfully")
    
    def analyze_class_distribution(self):
        """Analyze and report class distribution for balanced sampling."""
        print("\n📊 ANALYZING CLASS DISTRIBUTION FOR BALANCED SAMPLING")
        print("=" * 60)
        
        # Count classes in training set
        train_class_counts = Counter()
        for video in self.train_dataset.videos:
            train_class_counts[video['class']] += 1
        
        # Count classes in validation set
        val_class_counts = Counter()
        for video in self.val_dataset.videos:
            val_class_counts[video['class']] += 1
        
        print("Training set distribution:")
        total_train = sum(train_class_counts.values())
        for class_name in self.selected_classes:
            count = train_class_counts[class_name]
            percentage = (count / total_train) * 100
            print(f"  {class_name}: {count} videos ({percentage:.1f}%)")
        
        print("\nValidation set distribution:")
        total_val = sum(val_class_counts.values())
        for class_name in self.selected_classes:
            count = val_class_counts[class_name]
            percentage = (count / total_val) * 100
            print(f"  {class_name}: {count} videos ({percentage:.1f}%)")
        
        # Calculate class weights for balanced sampling
        self.class_counts = [train_class_counts[cls] for cls in self.selected_classes]
        max_count = max(self.class_counts)
        self.class_weights_balanced = [max_count / count for count in self.class_counts]
        
        print(f"\nBalanced sampling weights:")
        for i, cls in enumerate(self.selected_classes):
            print(f"  {cls}: {self.class_weights_balanced[i]:.3f}")
        
        # Store for analysis
        self.train_class_counts = train_class_counts
        self.val_class_counts = val_class_counts
    
    def create_balanced_data_loaders(self):
        """Create data loaders with balanced sampling."""
        print(f"\n🔄 CREATING BALANCED DATA LOADERS ({self.balance_strategy})")
        print("=" * 60)
        
        if self.balance_strategy == "weighted_random" and self.use_weighted_sampler:
            # Create sample weights for WeightedRandomSampler
            sample_weights = []
            for video in self.train_dataset.videos:
                class_idx = self.class_to_idx[video['class']]
                sample_weights.append(self.class_weights_balanced[class_idx])
            
            # Create WeightedRandomSampler
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            
            self.train_loader = DataLoader(
                self.train_dataset, 
                batch_size=self.batch_size, 
                sampler=sampler,
                num_workers=0, 
                drop_last=True
            )
            
            print(f"✅ Created WeightedRandomSampler with {len(sample_weights)} samples")
            
        else:
            # Standard balanced loader with shuffle
            self.train_loader = DataLoader(
                self.train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True,
                num_workers=0, 
                drop_last=True
            )
            
            print(f"✅ Created standard balanced loader with shuffle")
        
        # Validation loader (no balancing needed)
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=0
        )
        
        print(f"📊 Training batches: {len(self.train_loader)}")
        print(f"📊 Validation batches: {len(self.val_loader)}")
    
    def setup_resumed_training(self):
        """Setup model and training components from checkpoint."""
        print("\n🏗️  SETTING UP RESUMED TRAINING SYSTEM")
        print("=" * 60)
        
        # Initialize model with same architecture
        self.model = ResumedLipReadingModel().to(self.device)
        
        # Load model state from checkpoint
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        print(f"✅ Loaded model state from checkpoint")
        
        # Get model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Setup optimizer with reduced learning rate for fine-tuning
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.initial_lr,  # Reduced LR for fine-tuning
            weight_decay=1e-4,   # L2 regularization
            betas=(0.9, 0.999)
        )
        
        # Load optimizer state if available
        if 'optimizer_state_dict' in self.checkpoint:
            try:
                self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
                # Update learning rate to new value
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.initial_lr
                print(f"✅ Loaded optimizer state and updated LR to {self.initial_lr}")
            except Exception as e:
                print(f"⚠️  Could not load optimizer state: {e}")
                print("   Continuing with fresh optimizer")
        
        # Learning rate scheduler for extended training
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=15, T_mult=1, eta_min=1e-6
        )
        
        # Balanced loss function with class weights
        class_weights_tensor = torch.tensor(self.class_weights_balanced, dtype=torch.float32).to(self.device)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights_tensor,
            label_smoothing=0.1
        )
        
        # Initialize tracking variables
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.per_class_accuracies = []
        self.confusion_matrices = []
        
        # Resume from checkpoint values
        self.best_val_acc = self.checkpoint.get('best_val_acc', 0.0)
        self.start_epoch = self.checkpoint.get('epoch', 0) + 1
        self.epochs_without_improvement = 0
        
        print(f"✅ Resumed training setup complete:")
        print(f"   Starting epoch: {self.start_epoch}")
        print(f"   Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"   Architecture: 2.98M parameter CNN-3D model")
        print(f"   Optimizer: AdamW (lr={self.initial_lr}, fine-tuning mode)")
        print(f"   Loss: CrossEntropyLoss with balanced class weights")
        print(f"   Scheduler: CosineAnnealingWarmRestarts")

    def train_epoch(self, epoch):
        """Train for one epoch with balanced sampling."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        epoch_start_time = time.time()

        for batch_idx, (videos, labels) in enumerate(self.train_loader):
            videos, labels = videos.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(videos)
            loss = self.criterion(outputs, labels)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Per-class statistics
            for i in range(len(labels)):
                class_name = self.selected_classes[labels[i]]
                class_total[class_name] += 1
                if predicted[i] == labels[i]:
                    class_correct[class_name] += 1

            # Progress reporting
            if batch_idx % 10 == 0:
                batch_acc = 100. * correct / total
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}: '
                      f'Loss: {loss.item():.4f}, Acc: {batch_acc:.2f}%')

        epoch_time = time.time() - epoch_start_time
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        # Per-class accuracies
        class_accs = {}
        for class_name in self.selected_classes:
            if class_total[class_name] > 0:
                class_accs[class_name] = 100. * class_correct[class_name] / class_total[class_name]
            else:
                class_accs[class_name] = 0.0

        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)

        print(f'\nEpoch {epoch} Training Results:')
        print(f'  Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        print(f'  Time: {epoch_time:.1f}s')
        print(f'  Per-class accuracies:')
        for class_name, acc in class_accs.items():
            print(f'    {class_name}: {acc:.1f}%')

        return epoch_loss, epoch_acc, class_accs

    def validate_epoch(self, epoch):
        """Validate for one epoch."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        with torch.no_grad():
            for videos, labels in self.val_loader:
                videos, labels = videos.to(self.device), labels.to(self.device)
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Store for confusion matrix
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Per-class statistics
                for i in range(len(labels)):
                    class_name = self.selected_classes[labels[i]]
                    class_total[class_name] += 1
                    if predicted[i] == labels[i]:
                        class_correct[class_name] += 1

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total

        # Per-class accuracies
        class_accs = {}
        for class_name in self.selected_classes:
            if class_total[class_name] > 0:
                class_accs[class_name] = 100. * class_correct[class_name] / class_total[class_name]
            else:
                class_accs[class_name] = 0.0

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)

        self.val_losses.append(epoch_loss)
        self.val_accuracies.append(epoch_acc)
        self.per_class_accuracies.append(class_accs)
        self.confusion_matrices.append(cm)

        print(f'\nEpoch {epoch} Validation Results:')
        print(f'  Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        print(f'  Per-class accuracies:')
        for class_name, acc in class_accs.items():
            print(f'    {class_name}: {acc:.1f}%')

        return epoch_loss, epoch_acc, class_accs, cm

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'val_acc': val_acc,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'per_class_accuracies': self.per_class_accuracies,
            'class_to_idx': self.class_to_idx,
            'selected_classes': self.selected_classes
        }

        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"resumed_checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.output_dir / "resumed_best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model: {best_path}")

        print(f"💾 Saved checkpoint: {checkpoint_path}")

    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        epochs = range(1, len(self.train_losses) + 1)

        # Loss curves
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy curves
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        ax2.axhline(y=self.target_val_acc, color='g', linestyle='--', label=f'Target ({self.target_val_acc}%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)

        # Per-class validation accuracies
        if self.per_class_accuracies:
            for class_name in self.selected_classes:
                class_accs = [epoch_accs[class_name] for epoch_accs in self.per_class_accuracies]
                ax3.plot(epochs, class_accs, label=class_name)
            ax3.set_title('Per-Class Validation Accuracy')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Accuracy (%)')
            ax3.legend()
            ax3.grid(True)

        # Learning rate
        if hasattr(self.scheduler, 'get_last_lr'):
            lrs = [self.scheduler.get_last_lr()[0] for _ in epochs]
            ax4.plot(epochs, lrs, 'g-')
            ax4.set_title('Learning Rate Schedule')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.set_yscale('log')
            ax4.grid(True)

        plt.tight_layout()
        curves_path = self.output_dir / "resumed_training_curves.png"
        plt.savefig(curves_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 Saved training curves: {curves_path}")

    def run_resumed_training(self):
        """Execute the complete resumed training pipeline."""
        print("\n🚀 STARTING RESUMED BALANCED TRAINING PIPELINE")
        print("=" * 80)

        try:
            # Load checkpoint and datasets
            self.load_checkpoint_and_datasets()

            # Setup training
            self.setup_resumed_training()

            print(f"\n🎯 TRAINING TARGET: {self.target_val_acc}% validation accuracy")
            print(f"📈 Starting from: {self.best_val_acc:.2f}% (epoch {self.start_epoch-1})")
            print(f"🔄 Max epochs: {self.max_epochs}, Early stopping: {self.early_stopping_patience}")

            # Training loop
            for epoch in range(self.start_epoch, self.max_epochs + 1):
                print(f"\n{'='*60}")
                print(f"EPOCH {epoch}/{self.max_epochs}")
                print(f"{'='*60}")

                # Train epoch
                train_loss, train_acc, train_class_accs = self.train_epoch(epoch)

                # Validate epoch
                val_loss, val_acc, val_class_accs, cm = self.validate_epoch(epoch)

                # Update scheduler
                self.scheduler.step()

                # Check for improvement
                is_best = val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc
                    self.epochs_without_improvement = 0
                    print(f"🎉 NEW BEST VALIDATION ACCURACY: {val_acc:.2f}%")

                    # Check if target reached
                    if val_acc >= self.target_val_acc:
                        print(f"🎯 TARGET ACHIEVED! Validation accuracy: {val_acc:.2f}% >= {self.target_val_acc}%")
                        self.save_checkpoint(epoch, val_acc, is_best=True)
                        self.plot_training_curves()
                        self.save_final_results(epoch, val_acc, success=True)
                        return True
                else:
                    self.epochs_without_improvement += 1
                    print(f"📉 No improvement for {self.epochs_without_improvement} epochs")

                # Save checkpoint
                self.save_checkpoint(epoch, val_acc, is_best=is_best)

                # Early stopping check
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f"\n⏹️  EARLY STOPPING: No improvement for {self.early_stopping_patience} epochs")
                    break

                # Progress summary
                print(f"\n📊 EPOCH {epoch} SUMMARY:")
                print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
                print(f"   Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
                print(f"   Best:  {self.best_val_acc:.2f}% (target: {self.target_val_acc}%)")
                print(f"   LR:    {self.optimizer.param_groups[0]['lr']:.6f}")

            # Training completed
            self.plot_training_curves()
            success = self.best_val_acc >= self.target_val_acc
            self.save_final_results(epoch, self.best_val_acc, success=success)

            return success

        except Exception as e:
            print(f"\n❌ TRAINING ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_final_results(self, final_epoch, final_acc, success=False):
        """Save final training results and analysis."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'success': bool(success),
            'target_accuracy': float(self.target_val_acc),
            'best_accuracy': float(self.best_val_acc),
            'final_accuracy': float(final_acc),
            'final_epoch': int(final_epoch),
            'total_epochs_trained': int(final_epoch - self.start_epoch + 1),
            'improvement_from_checkpoint': float(self.best_val_acc - self.checkpoint.get('best_val_acc', 0.0)),
            'training_config': {
                'batch_size': int(self.batch_size),
                'initial_lr': float(self.initial_lr),
                'balance_strategy': str(self.balance_strategy),
                'early_stopping_patience': int(self.early_stopping_patience)
            },
            'class_distribution': {
                'train': {k: int(v) for k, v in self.train_class_counts.items()},
                'validation': {k: int(v) for k, v in self.val_class_counts.items()}
            }
        }

        # Save results
        results_path = self.output_dir / "resumed_training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Create summary report
        report_path = self.output_dir / "resumed_training_report.txt"
        with open(report_path, 'w') as f:
            f.write("RESUMED BALANCED TRAINING REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"TARGET: {self.target_val_acc}% cross-demographic validation accuracy\n")
            f.write(f"SUCCESS: {'YES' if success else 'NO'}\n\n")
            f.write(f"FINAL RESULTS:\n")
            f.write(f"Best validation accuracy: {self.best_val_acc:.2f}%\n")
            f.write(f"Improvement from checkpoint: +{results['improvement_from_checkpoint']:.2f}%\n")
            f.write(f"Final epoch: {final_epoch}\n")
            f.write(f"Total epochs trained: {results['total_epochs_trained']}\n\n")

            if self.per_class_accuracies:
                f.write("PER-CLASS PERFORMANCE (Best Epoch):\n")
                best_epoch_idx = self.val_accuracies.index(max(self.val_accuracies))
                best_class_accs = self.per_class_accuracies[best_epoch_idx]
                for class_name, acc in best_class_accs.items():
                    f.write(f"{class_name}: {acc:.1f}%\n")

        print(f"💾 Saved final results: {results_path}")
        print(f"📄 Saved training report: {report_path}")


class BalancedLipReadingDataset(Dataset):
    """Dataset class for balanced lip-reading with enhanced augmentation."""

    def __init__(self, manifest_path, class_to_idx, augment=False, is_training=False):
        self.manifest_path = Path(manifest_path)
        self.class_to_idx = class_to_idx
        self.augment = augment
        self.is_training = is_training
        self.videos = []

        # Load video information from manifest
        with open(self.manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['class'] in self.class_to_idx:
                    self.videos.append({
                        'path': row['video_path'],
                        'class': row['class'],
                        'demographic_group': row.get('demographic_group', 'unknown')
                    })

        print(f"📊 Loaded {len(self.videos)} videos from {manifest_path}")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_info = self.videos[idx]
        video_path = video_info['path']
        class_name = video_info['class']

        # Load video frames
        frames = self.load_video_frames(video_path)

        # Apply augmentation if enabled
        if self.augment and self.is_training:
            frames = self.apply_augmentation(frames, class_name)

        # Convert to tensor
        frames_tensor = torch.FloatTensor(frames).unsqueeze(0)  # Add channel dimension
        label = self.class_to_idx[class_name]

        return frames_tensor, label

    def load_video_frames(self, video_path):
        """Load and preprocess video frames."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale and normalize
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)

        cap.release()

        # Ensure we have exactly 32 frames
        frames = np.array(frames)
        if len(frames) > 32:
            # Take center 32 frames
            start_idx = (len(frames) - 32) // 2
            frames = frames[start_idx:start_idx + 32]
        elif len(frames) < 32:
            # Pad with last frame
            last_frame = frames[-1] if len(frames) > 0 else np.zeros((64, 96), dtype=np.float32)
            while len(frames) < 32:
                frames = np.append(frames, [last_frame], axis=0)

        return frames

    def apply_augmentation(self, frames, class_name):
        """Apply balanced augmentation to frames."""
        # Enhanced augmentation for balanced training
        augmented_frames = frames.copy()

        # Random horizontal flip (50% chance)
        if random.random() < 0.5:
            augmented_frames = np.flip(augmented_frames, axis=2)

        # Brightness adjustment (±15%)
        brightness_factor = random.uniform(0.85, 1.15)
        augmented_frames = np.clip(augmented_frames * brightness_factor, 0, 1)

        # Contrast adjustment (0.9-1.1x)
        contrast_factor = random.uniform(0.9, 1.1)
        augmented_frames = np.clip((augmented_frames - 0.5) * contrast_factor + 0.5, 0, 1)

        # Gamma correction (0.95-1.05x)
        gamma = random.uniform(0.95, 1.05)
        augmented_frames = np.power(augmented_frames, gamma)

        # Small amount of Gaussian noise
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.01, augmented_frames.shape)
            augmented_frames = np.clip(augmented_frames + noise, 0, 1)

        return augmented_frames


class ResumedLipReadingModel(nn.Module):
    """Identical model architecture to the 75.9% checkpoint."""

    def __init__(self):
        super(ResumedLipReadingModel, self).__init__()

        # IDENTICAL architecture from successful 75.9% training
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
    """Execute resumed balanced training pipeline."""
    print("🎯 STARTING RESUMED BALANCED TRAINING FROM 75.9% CHECKPOINT")
    print("🚀 TARGET: 82% Cross-Demographic Validation Accuracy")
    print("⚖️  STRATEGY: Balanced sampling + extended fine-tuning")

    trainer = ResumedBalancedTrainer()
    success = trainer.run_resumed_training()

    if success:
        print("\n🎉 TRAINING SUCCESS!")
        print(f"✅ Successfully achieved {trainer.target_val_acc}% validation accuracy target")
        print(f"📈 Best accuracy: {trainer.best_val_acc:.2f}%")
        print("🚀 Model ready for production deployment")
    else:
        print("\n💡 Training completed with valuable progress")
        print(f"📊 Best accuracy achieved: {trainer.best_val_acc:.2f}%")
        print(f"🎯 Target was: {trainer.target_val_acc}%")
        print("🔍 Check results for further optimization strategies")

if __name__ == "__main__":
    main()
