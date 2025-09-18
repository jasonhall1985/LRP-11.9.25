#!/usr/bin/env python3
"""
Comprehensive Lip-Reading Classifier
====================================
Train a CNN-LSTM model using strict demographic dataset splits.

Target: >80% validation accuracy across 7 word phrase classes
Architecture: CNN-LSTM with proper regularization
Input: 96×64 landscape videos, 32 frames, grayscale

Author: Augment Agent
Date: 2025-09-18
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class VideoDataset(Dataset):
    """PyTorch Dataset for lip-reading videos."""

    def __init__(self, video_paths, labels, target_frames=32, target_height=64, target_width=96):
        self.video_paths = video_paths
        self.labels = labels
        self.target_frames = target_frames
        self.target_height = target_height
        self.target_width = target_width

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Load and preprocess video
        video_data = self.preprocess_video(video_path)

        if video_data is None:
            # Return zeros if video loading fails
            video_data = torch.zeros(self.target_frames, self.target_height, self.target_width)

        return video_data, label

    def preprocess_video(self, video_path):
        """Preprocess video to target format."""
        try:
            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                return None

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to grayscale
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Resize to target dimensions
                frame = cv2.resize(frame, (self.target_width, self.target_height))

                # Normalize to [0, 1]
                frame = frame.astype(np.float32) / 255.0

                frames.append(frame)

            cap.release()

            if len(frames) == 0:
                return None

            # Handle temporal sampling to get exactly target_frames
            frames = np.array(frames)

            if len(frames) >= self.target_frames:
                # Sample frames evenly
                indices = np.linspace(0, len(frames) - 1, self.target_frames, dtype=int)
                frames = frames[indices]
            else:
                # Pad with last frame if too few frames
                padding_needed = self.target_frames - len(frames)
                last_frame = frames[-1] if len(frames) > 0 else np.zeros((self.target_height, self.target_width))
                padding = np.tile(last_frame[np.newaxis, :, :], (padding_needed, 1, 1))
                frames = np.concatenate([frames, padding], axis=0)

            # Convert to PyTorch tensor: (frames, height, width)
            frames = torch.FloatTensor(frames)

            return frames

        except Exception as e:
            print(f"❌ Error processing video {video_path}: {str(e)}")
            return None


class CNN_LSTM_Model(nn.Module):
    """CNN-LSTM model for lip-reading classification."""

    def __init__(self, num_classes=7, input_channels=1, hidden_size=256):
        super(CNN_LSTM_Model, self).__init__()

        # 3D CNN layers for spatial-temporal feature extraction
        self.conv3d1 = nn.Conv3d(input_channels, 32, kernel_size=(3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.dropout1 = nn.Dropout3d(0.25)

        self.conv3d2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d((2, 2, 2))
        self.dropout2 = nn.Dropout3d(0.25)

        self.conv3d3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d((2, 2, 2))
        self.dropout3 = nn.Dropout3d(0.3)

        # Calculate the size after conv layers
        # Input: (32, 64, 96) -> after pooling: (8, 8, 12)
        self.feature_size = 128 * 8 * 12  # 128 channels * 8 height * 12 width

        # LSTM layers
        self.lstm1 = nn.LSTM(self.feature_size, hidden_size, batch_first=True, dropout=0.3)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size//2, batch_first=True, dropout=0.3)

        # Dense layers
        self.fc1 = nn.Linear(hidden_size//2, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout_fc1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 128)
        self.dropout_fc2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input shape: (batch_size, frames, height, width)
        # Reshape for 3D conv: (batch_size, channels, frames, height, width)
        x = x.unsqueeze(1)  # Add channel dimension

        # 3D CNN feature extraction
        x = F.relu(self.bn1(self.conv3d1(x)))
        x = self.dropout1(self.pool1(x))

        x = F.relu(self.bn2(self.conv3d2(x)))
        x = self.dropout2(self.pool2(x))

        x = F.relu(self.bn3(self.conv3d3(x)))
        x = self.dropout3(self.pool3(x))

        # Reshape for LSTM: (batch_size, time_steps, features)
        batch_size, channels, frames, height, width = x.size()
        x = x.permute(0, 2, 1, 3, 4)  # (batch_size, frames, channels, height, width)
        x = x.contiguous().view(batch_size, frames, -1)  # Flatten spatial dimensions

        # LSTM layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        # Take the last output
        x = x[:, -1, :]  # (batch_size, hidden_size//2)

        # Dense layers
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout_fc2(x)

        x = self.fc3(x)

        return x


class LipReadingClassifier:
    """Comprehensive lip-reading classifier with CNN-LSTM architecture."""

    def __init__(self, model_dir="lip_reading_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Video preprocessing parameters
        self.target_frames = 32
        self.target_height = 64
        self.target_width = 96
        self.channels = 1  # Grayscale

        # Class mapping
        self.classes = ['doctor', 'glasses', 'help', 'phone', 'pillow', 'my_mouth_is_dry', 'i_need_to_move']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")

        # Model and training attributes
        self.model = None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.class_weights = None

        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        print("🎯 Comprehensive Lip-Reading Classifier Initialized")
        print(f"📁 Model directory: {self.model_dir}")
        print(f"🎬 Target video format: {self.target_frames} frames × {self.target_height}×{self.target_width} grayscale")
        print(f"📊 Classes: {len(self.classes)} word phrases")
    
    def load_dataset_split(self, manifest_path, split_name):
        """Load dataset split and return video paths and labels."""
        print(f"\n📁 Loading {split_name} dataset from: {manifest_path}")

        if not Path(manifest_path).exists():
            print(f"❌ Manifest file not found: {manifest_path}")
            return None, None

        df = pd.read_csv(manifest_path)
        print(f"📄 Found {len(df)} videos in {split_name} split")

        # Filter for known classes only
        df = df[df['class'].isin(self.classes)]
        print(f"📊 Filtered to {len(df)} videos with known classes")

        if len(df) == 0:
            print(f"❌ No valid videos found in {split_name} split")
            return None, None

        # Show class distribution
        class_counts = df['class'].value_counts()
        print(f"📈 Class distribution in {split_name}:")
        for class_name, count in class_counts.items():
            print(f"   - {class_name}: {count} videos")

        # Extract video paths and labels
        video_paths = []
        labels = []

        for idx, row in df.iterrows():
            video_path = Path(row['full_path'])
            class_name = row['class']

            if video_path.exists():
                video_paths.append(str(video_path))
                labels.append(self.class_to_idx[class_name])
            else:
                print(f"⚠️  File not found: {video_path}")

        print(f"✅ Successfully found: {len(video_paths)} videos")

        return video_paths, labels
    
    def create_data_loaders(self, batch_size=8):
        """Create PyTorch data loaders for all splits."""
        print("🎯 CREATING DATA LOADERS")
        print("=" * 60)

        # Load training set
        train_paths, train_labels = self.load_dataset_split(
            "strict_demographic_splits/strict_train_manifest.csv", "training"
        )

        # Load validation set
        val_paths, val_labels = self.load_dataset_split(
            "strict_demographic_splits/strict_validation_manifest.csv", "validation"
        )

        # Load test set
        test_paths, test_labels = self.load_dataset_split(
            "strict_demographic_splits/strict_test_manifest.csv", "test"
        )

        if train_paths is None or val_paths is None or test_paths is None:
            print("❌ Failed to load one or more dataset splits")
            return False

        # Create datasets
        train_dataset = VideoDataset(train_paths, train_labels,
                                   self.target_frames, self.target_height, self.target_width)
        val_dataset = VideoDataset(val_paths, val_labels,
                                 self.target_frames, self.target_height, self.target_width)
        test_dataset = VideoDataset(test_paths, test_labels,
                                  self.target_frames, self.target_height, self.target_width)

        # Compute class weights for balanced training
        self.compute_class_weights(train_labels)

        # Create weighted sampler for training
        sample_weights = [self.class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                     sampler=sampler, num_workers=2, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                   shuffle=False, num_workers=2, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=2, pin_memory=True)

        print(f"\n📊 DATA LOADER SUMMARY:")
        print(f"   Training batches: {len(self.train_loader)}")
        print(f"   Validation batches: {len(self.val_loader)}")
        print(f"   Test batches: {len(self.test_loader)}")
        print(f"   Batch size: {batch_size}")

        return True
    
    def compute_class_weights(self, train_labels):
        """Compute class weights for balanced training."""
        print(f"\n⚖️  COMPUTING CLASS WEIGHTS")
        print("-" * 40)

        # Get unique classes in training set
        unique_classes = np.unique(train_labels)

        # Compute class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=train_labels
        )

        # Create class weight dictionary
        self.class_weights = {int(cls): weight for cls, weight in zip(unique_classes, class_weights)}

        print("Class weights for balanced training:")
        for cls_idx, weight in self.class_weights.items():
            class_name = self.idx_to_class[cls_idx]
            print(f"   {class_name}: {weight:.3f}")

        return self.class_weights

    def build_model(self):
        """Build CNN-LSTM model."""
        print(f"\n🏗️  BUILDING CNN-LSTM MODEL")
        print("-" * 40)

        self.model = CNN_LSTM_Model(num_classes=len(self.classes)).to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print("✅ CNN-LSTM model built successfully")
        print(f"📊 Total parameters: {total_params:,}")
        print(f"📊 Trainable parameters: {trainable_params:,}")

        return self.model
    
    def train_model(self, epochs=100, batch_size=8, learning_rate=0.001):
        """Train the lip-reading model."""
        print(f"\n🎯 TRAINING LIP-READING MODEL")
        print("=" * 60)
        print(f"🎯 Target: >80% validation accuracy")
        print(f"⚖️  Using class-balanced training")
        print(f"📊 Epochs: {epochs}, Batch size: {batch_size}")

        # Setup optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)

        # Training variables
        best_val_acc = 0.0
        patience_counter = 0
        patience = 15

        print(f"🚀 Starting training on {self.device}")

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (videos, labels) in enumerate(self.train_loader):
                videos, labels = videos.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(videos)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(self.train_loader)}, '
                          f'Loss: {loss.item():.4f}')

            train_acc = train_correct / train_total
            train_loss = train_loss / len(self.train_loader)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for videos, labels in self.val_loader:
                    videos, labels = videos.to(self.device), labels.to(self.device)
                    outputs = self.model(videos)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_acc = val_correct / val_total
            val_loss = val_loss / len(self.val_loader)

            # Update learning rate
            scheduler.step(val_acc)

            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%)')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%)')

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), self.model_dir / 'best_model.pth')
                print(f'  🎯 New best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)')
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break

            print()

        print("✅ Training completed!")
        print(f"🎯 Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")

        if best_val_acc >= 0.80:
            print("🎉 TARGET ACHIEVED: >80% validation accuracy!")
        else:
            print(f"⚠️  Target not reached. Best: {best_val_acc*100:.2f}% (Target: 80%)")

        return self.history
    
    def plot_training_history(self):
        """Plot training history."""
        if not self.history['train_acc']:
            print("❌ No training history available")
            return

        print(f"\n📊 PLOTTING TRAINING HISTORY")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        epochs = range(1, len(self.history['train_acc']) + 1)

        # Plot accuracy
        ax1.plot(epochs, self.history['train_acc'], label='Training Accuracy', linewidth=2)
        ax1.plot(epochs, self.history['val_acc'], label='Validation Accuracy', linewidth=2)
        ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Target (80%)')
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot loss
        ax2.plot(epochs, self.history['train_loss'], label='Training Loss', linewidth=2)
        ax2.plot(epochs, self.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = self.model_dir / 'training_history.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training history saved: {plot_path}")

        plt.show()
    
    def evaluate_model(self):
        """Evaluate model on test set."""
        print(f"\n🔍 EVALUATING MODEL ON TEST SET")
        print("=" * 60)

        if self.model is None:
            print("❌ No model available for evaluation")
            return None

        # Load best model
        best_model_path = self.model_dir / 'best_model.pth'
        if best_model_path.exists():
            print(f"📁 Loading best model: {best_model_path}")
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        # Evaluate on test set
        self.model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_labels = []

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for videos, labels in self.test_loader:
                videos, labels = videos.to(self.device), labels.to(self.device)
                outputs = self.model(videos)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_accuracy = test_correct / test_total
        test_loss = test_loss / len(self.test_loader)

        print(f"📊 Test Results:")
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

        # Classification report
        print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
        print("-" * 60)

        # Get class names for available classes in test set
        test_classes = np.unique(all_labels)
        test_class_names = [self.idx_to_class[idx] for idx in test_classes]

        report = classification_report(
            all_labels, all_predictions,
            target_names=test_class_names,
            labels=test_classes,
            digits=4
        )
        print(report)

        # Confusion matrix
        self.plot_confusion_matrix(all_labels, all_predictions, test_classes, test_class_names)

        # Save evaluation results
        eval_results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'classification_report': report
        }

        return eval_results
    
    def plot_confusion_matrix(self, y_true, y_pred, class_indices, class_names):
        """Plot confusion matrix."""
        print(f"\n📊 GENERATING CONFUSION MATRIX")
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=class_indices)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save confusion matrix
        cm_path = self.model_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved: {cm_path}")
        
        plt.show()
    
    def save_training_summary(self, eval_results):
        """Save comprehensive training summary."""
        print(f"\n📄 SAVING TRAINING SUMMARY")

        summary_path = self.model_dir / 'training_summary.txt'

        with open(summary_path, 'w') as f:
            f.write("COMPREHENSIVE LIP-READING CLASSIFIER TRAINING SUMMARY\n")
            f.write("=" * 60 + "\n\n")

            # Model architecture
            f.write("MODEL ARCHITECTURE:\n")
            f.write("-" * 30 + "\n")
            f.write("CNN-LSTM Architecture for Lip-Reading (PyTorch)\n")
            f.write(f"Input Shape: ({self.target_frames}, {self.target_height}, {self.target_width})\n")
            f.write(f"Output Classes: {len(self.classes)}\n")

            if self.model:
                total_params = sum(p.numel() for p in self.model.parameters())
                f.write(f"Total Parameters: {total_params:,}\n\n")

            # Dataset information
            f.write("DATASET INFORMATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Training Batches: {len(self.train_loader) if self.train_loader else 'N/A'}\n")
            f.write(f"Validation Batches: {len(self.val_loader) if self.val_loader else 'N/A'}\n")
            f.write(f"Test Batches: {len(self.test_loader) if self.test_loader else 'N/A'}\n\n")

            # Training results
            if self.history['val_acc']:
                best_val_acc = max(self.history['val_acc'])
                f.write("TRAINING RESULTS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)\n")
                f.write(f"Target Achievement: {'✅ YES' if best_val_acc >= 0.80 else '❌ NO'} (Target: 80%)\n")
                f.write(f"Total Epochs: {len(self.history['val_acc'])}\n\n")

            # Test results
            if eval_results:
                f.write("TEST SET EVALUATION:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Test Accuracy: {eval_results['test_accuracy']:.4f} ({eval_results['test_accuracy']*100:.2f}%)\n")
                f.write(f"Test Loss: {eval_results['test_loss']:.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(eval_results['classification_report'])

            # Class weights
            if self.class_weights:
                f.write("\nCLASS WEIGHTS:\n")
                f.write("-" * 30 + "\n")
                for cls_idx, weight in self.class_weights.items():
                    class_name = self.idx_to_class[cls_idx]
                    f.write(f"{class_name}: {weight:.3f}\n")

        print(f"✅ Training summary saved: {summary_path}")

    def run_complete_training(self):
        """Run the complete training pipeline."""
        print("🎯 COMPREHENSIVE LIP-READING CLASSIFIER TRAINING PIPELINE")
        print("=" * 80)

        # Step 1: Create data loaders
        if not self.create_data_loaders(batch_size=8):
            print("❌ Failed to create data loaders")
            return False

        # Step 2: Build model
        self.build_model()

        # Step 3: Train model
        self.train_model(epochs=100, batch_size=8)

        # Step 4: Plot training history
        self.plot_training_history()

        # Step 5: Evaluate model
        eval_results = self.evaluate_model()

        # Step 6: Save training summary
        self.save_training_summary(eval_results)

        print(f"\n🎯 TRAINING PIPELINE COMPLETE!")
        print("=" * 80)
        print(f"📁 All outputs saved to: {self.model_dir}")

        return True

def main():
    """Main execution function."""
    # Initialize classifier
    classifier = LipReadingClassifier()
    
    # Run complete training pipeline
    success = classifier.run_complete_training()
    
    if success:
        print("✅ Lip-reading classifier training completed successfully!")
    else:
        print("❌ Training failed. Please check the logs.")
    
    return classifier

if __name__ == "__main__":
    main()
