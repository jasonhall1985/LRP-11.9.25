#!/usr/bin/env python3
"""
🎯 SPEAKER-DISJOINT TRAINING PIPELINE
====================================

Specialized training pipeline for achieving 82% cross-demographic validation accuracy
using speaker-disjoint splits to address checkpoint 165 catastrophic failure.

Key Features:
- Speaker-disjoint training (Speaker 1) and validation (Speaker 2)
- Command-line argument support for flexible configuration
- Lightweight 3D CNN-LSTM architecture (721K parameters)
- Cross-demographic validation targeting 82% accuracy
- Strong regularization to prevent overfitting
"""

import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

class LightweightCNNLSTM(nn.Module):
    """Lightweight 3D CNN-LSTM architecture for lip-reading (721K parameters)"""
    
    def __init__(self, num_classes=4, dropout=0.4):
        super(LightweightCNNLSTM, self).__init__()
        
        # Lightweight 3D CNN feature extractor
        self.conv3d1 = nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=1)
        self.bn3d1 = nn.BatchNorm3d(16)
        self.pool3d1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        
        self.conv3d2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1)
        self.bn3d2 = nn.BatchNorm3d(32)
        self.pool3d2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv3d3 = nn.Conv3d(32, 48, kernel_size=(3, 3, 3), padding=1)
        self.bn3d3 = nn.BatchNorm3d(48)
        self.pool3d3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 6))

        # LSTM for temporal modeling
        self.lstm_input_size = 48 * 4 * 6  # 1152 features per timestep
        self.lstm_hidden_size = 128
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        
        # Classifier head with configurable dropout
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.lstm_hidden_size, 64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        
        self.dropout2 = nn.Dropout(dropout * 0.75)  # Slightly less dropout for final layer
        self.fc_out = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # CNN feature extraction
        # Input: (batch, 1, 32, 64, 96)
        x = torch.relu(self.bn3d1(self.conv3d1(x)))
        x = self.pool3d1(x)
        
        x = torch.relu(self.bn3d2(self.conv3d2(x)))
        x = self.pool3d2(x)
        
        x = torch.relu(self.bn3d3(self.conv3d3(x)))
        x = self.pool3d3(x)
        
        x = self.adaptive_pool(x)
        
        # Reshape for LSTM: (batch, timesteps, features)
        batch_size = x.size(0)
        timesteps = x.size(2)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.contiguous().view(batch_size, timesteps, -1)
        
        # LSTM temporal modeling
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last timestep output
        x = lstm_out[:, -1, :]
        
        # Classification head
        x = self.dropout1(x)
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        
        x = self.dropout2(x)
        x = self.fc_out(x)
        
        return x

class PreprocessedVideoDataset(Dataset):
    """Dataset for loading preprocessed .npy video files"""
    
    def __init__(self, manifest_path, class_to_idx=None):
        self.manifest_df = pd.read_csv(manifest_path)
        
        # Create class mapping if not provided
        if class_to_idx is None:
            unique_classes = sorted(self.manifest_df['class'].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        else:
            self.class_to_idx = class_to_idx
        
        print(f"📊 Dataset loaded: {len(self.manifest_df)} samples")
        print(f"📊 Classes: {list(self.class_to_idx.keys())}")
        
        # Print class distribution
        class_counts = self.manifest_df['class'].value_counts()
        for cls, count in class_counts.items():
            print(f"   {cls}: {count} samples")
    
    def __len__(self):
        return len(self.manifest_df)
    
    def __getitem__(self, idx):
        row = self.manifest_df.iloc[idx]
        
        # Load preprocessed video data
        video_data = np.load(row['file_path'])
        
        # Convert to tensor and add channel dimension: (1, 32, 64, 96)
        video_tensor = torch.FloatTensor(video_data).unsqueeze(0)
        
        # Get label
        label = self.class_to_idx[row['class']]
        
        return video_tensor, label

class SpeakerDisjointTrainer:
    """Speaker-disjoint training pipeline for cross-demographic validation"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cpu')  # Use CPU for compatibility
        
        # Class mapping (fixed for 4-class problem)
        self.class_to_idx = {
            'doctor': 0,
            'i_need_to_move': 1,
            'my_mouth_is_dry': 2,
            'pillow': 3
        }
        
        # Initialize model
        self.model = LightweightCNNLSTM(
            num_classes=len(self.class_to_idx),
            dropout=args.dropout
        )
        
        print(f"🏗️  Model initialized: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        # Training state
        self.best_val_acc = 0.0
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
    def create_data_loaders(self):
        """Create training and validation data loaders"""
        print("📂 Creating data loaders...")
        
        # Create datasets
        train_dataset = PreprocessedVideoDataset(self.args.train_manifest, self.class_to_idx)
        val_dataset = PreprocessedVideoDataset(self.args.val_manifest, self.class_to_idx)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0  # Use 0 for compatibility
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        print(f"✅ Training batches: {len(self.train_loader)}")
        print(f"✅ Validation batches: {len(self.val_loader)}")
    
    def train_epoch(self, optimizer, criterion, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (videos, labels) in enumerate(self.train_loader):
            videos, labels = videos.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"   Batch {batch_idx}/{len(self.train_loader)}: Loss {loss.item():.4f}")
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, criterion):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for videos, labels in self.val_loader:
                videos, labels = videos.to(self.device), labels.to(self.device)
                
                outputs = self.model(videos)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc, all_predictions, all_labels
    
    def train(self):
        """Execute full training pipeline"""
        print("🚀 Starting Speaker-Disjoint Training")
        print("=" * 50)
        
        # Create data loaders
        self.create_data_loaders()
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        # Training loop
        epochs_without_improvement = 0
        
        for epoch in range(self.args.epochs):
            print(f"\n📅 Epoch {epoch+1}/{self.args.epochs}")
            print("-" * 30)
            
            # Train
            train_loss, train_acc = self.train_epoch(optimizer, criterion, epoch)
            
            # Validate
            val_loss, val_acc, val_predictions, val_labels = self.validate_epoch(criterion)
            
            # Update learning rate
            scheduler.step(val_acc)
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            print(f"📊 Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"📊 Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                epochs_without_improvement = 0
                
                # Save model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': self.best_val_acc,
                    'class_to_idx': self.class_to_idx,
                    'train_losses': self.train_losses,
                    'train_accuracies': self.train_accuracies,
                    'val_losses': self.val_losses,
                    'val_accuracies': self.val_accuracies
                }, 'best_model.pth')
                
                print(f"✅ New best model saved! Validation accuracy: {val_acc:.2f}%")
                
                # Check if target reached
                if val_acc >= self.args.target_accuracy:
                    print(f"🎉 TARGET REACHED! Validation accuracy {val_acc:.2f}% >= {self.args.target_accuracy}%")
                    break
            else:
                epochs_without_improvement += 1
                print(f"⏳ No improvement for {epochs_without_improvement} epochs")
            
            # Early stopping
            if epochs_without_improvement >= self.args.early_stop_patience:
                print(f"🛑 Early stopping after {epochs_without_improvement} epochs without improvement")
                break
        
        print(f"\n🏁 Training completed!")
        print(f"🏆 Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # Save training history
        history = {
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'class_to_idx': self.class_to_idx,
            'args': vars(self.args)
        }
        
        with open('training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        return self.best_val_acc >= self.args.target_accuracy

def main():
    parser = argparse.ArgumentParser(description='Speaker-Disjoint Training Pipeline')
    
    # Data arguments
    parser.add_argument('--train-manifest', required=True, help='Training manifest CSV file')
    parser.add_argument('--val-manifest', required=True, help='Validation manifest CSV file')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=80, help='Maximum epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--early-stop-patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--target-accuracy', type=float, default=82.0, help='Target validation accuracy')
    
    # Architecture arguments
    parser.add_argument('--architecture', default='lightweight', help='Model architecture')
    
    # Output arguments
    parser.add_argument('--output-dir', default='.', help='Output directory')
    
    # Flags
    parser.add_argument('--cross-demographic-validation', action='store_true', 
                       help='Enable cross-demographic validation mode')
    parser.add_argument('--no-synthetic-augmentation', action='store_true',
                       help='Disable synthetic augmentation')
    
    args = parser.parse_args()
    
    print("🎯 SPEAKER-DISJOINT TRAINING PIPELINE")
    print("=" * 50)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Configuration:")
    print(f"   Training manifest: {args.train_manifest}")
    print(f"   Validation manifest: {args.val_manifest}")
    print(f"   Target accuracy: {args.target_accuracy}%")
    print(f"   Max epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Weight decay: {args.weight_decay}")
    print(f"   Dropout: {args.dropout}")
    print(f"   Early stopping patience: {args.early_stop_patience}")
    print("")
    
    # Initialize trainer
    trainer = SpeakerDisjointTrainer(args)
    
    # Execute training
    success = trainer.train()
    
    if success:
        print(f"\n🎉 SUCCESS: Target accuracy {args.target_accuracy}% achieved!")
        print(f"🏆 Final validation accuracy: {trainer.best_val_acc:.2f}%")
    else:
        print(f"\n⚠️  Training completed but target accuracy {args.target_accuracy}% not reached")
        print(f"🏆 Best validation accuracy: {trainer.best_val_acc:.2f}%")
    
    print(f"\n✅ Training pipeline completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
