#!/usr/bin/env python3
"""
Balanced 49-Per-Class Model Training
Train 4-class lip-reading model with perfectly balanced dataset (49 videos per class)
Use identical CNN-LSTM architecture from previous balanced model training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import numpy as np
import os
from datetime import datetime
import random
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class LipReadingDataset(Dataset):
    """
    Dataset class for lip-reading videos with data augmentation
    """
    
    def __init__(self, manifest_path, augment=False):
        self.df = pd.read_csv(manifest_path)
        self.augment = augment
        
        # Create class to index mapping
        self.classes = sorted(self.df['class'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        print(f"Dataset loaded: {len(self.df)} videos")
        print(f"Classes: {self.classes}")
        print(f"Class distribution:")
        for cls in self.classes:
            count = len(self.df[self.df['class'] == cls])
            print(f"  {cls}: {count} videos")
    
    def __len__(self):
        return len(self.df)
    
    def load_video(self, video_path):
        """Load and preprocess video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        
        cap.release()
        
        # Ensure exactly 32 frames
        if len(frames) >= 32:
            start_idx = (len(frames) - 32) // 2
            frames = frames[start_idx:start_idx + 32]
        else:
            # Pad with repeated frames
            while len(frames) < 32:
                frames.extend(frames[:min(len(frames), 32 - len(frames))])
            frames = frames[:32]
        
        return np.array(frames)
    
    def augment_video(self, frames):
        """Apply data augmentation"""
        if not self.augment:
            return frames
        
        # Brightness adjustment (±10-15%)
        brightness_factor = np.random.uniform(0.85, 1.15)
        frames = np.clip(frames * brightness_factor, 0, 1)
        
        # Contrast adjustment (0.9-1.1x)
        contrast_factor = np.random.uniform(0.9, 1.1)
        frames = np.clip((frames - 0.5) * contrast_factor + 0.5, 0, 1)
        
        # Horizontal flipping (50% chance)
        if np.random.random() > 0.5:
            frames = np.flip(frames, axis=2).copy()  # Make a copy to avoid negative strides
        
        return frames
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row['video_path']
        class_name = row['class']
        class_idx = self.class_to_idx[class_name]
        
        # Load video
        frames = self.load_video(video_path)
        
        # Apply augmentation
        frames = self.augment_video(frames)
        
        # Convert to tensor: (T, H, W) -> (1, T, H, W)
        frames_tensor = torch.FloatTensor(frames).unsqueeze(0)
        
        return frames_tensor, class_idx

class CNN_LSTM_Model(nn.Module):
    """
    CNN-LSTM model for lip-reading (identical to previous balanced model)
    """
    
    def __init__(self, num_classes=4):
        super(CNN_LSTM_Model, self).__init__()
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        # Calculate CNN output size
        # Input: (1, 64, 96) -> After 3 maxpools: (128, 8, 12)
        self.cnn_output_size = 128 * 8 * 12
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, 1, seq_len, height, width)
        batch_size, channels, seq_len, height, width = x.size()
        
        # Reshape for CNN: (batch_size * seq_len, channels, height, width)
        x = x.view(batch_size * seq_len, channels, height, width)
        
        # CNN feature extraction
        x = self.cnn(x)
        
        # Flatten CNN output
        x = x.view(batch_size * seq_len, -1)
        
        # Reshape for LSTM: (batch_size, seq_len, features)
        x = x.view(batch_size, seq_len, -1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Use last LSTM output
        x = lstm_out[:, -1, :]
        
        # Classification
        x = self.classifier(x)
        
        return x

def train_model():
    """
    Train the balanced 49-per-class lip-reading model
    """
    print("🎯 BALANCED 49-PER-CLASS MODEL TRAINING")
    print("=" * 60)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    print("\n📊 Loading balanced datasets...")
    train_dataset = LipReadingDataset('balanced_49_training_results/balanced_196_train_manifest.csv', augment=True)
    val_dataset = LipReadingDataset('balanced_49_training_results/balanced_196_validation_manifest.csv', augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Initialize model
    model = CNN_LSTM_Model(num_classes=4).to(device)
    
    # Loss and optimizer (identical to previous balanced model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    # Training parameters
    num_epochs = 100
    best_val_acc = 0.0
    patience = 20
    patience_counter = 0
    
    # Training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        train_acc = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100.0 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save history
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] "
                  f"Train Loss: {avg_train_loss:.4f} Train Acc: {train_acc:.2f}% "
                  f"Val Loss: {avg_val_loss:.4f} Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'train_accuracy': train_acc,
                'class_to_idx': train_dataset.class_to_idx,
                'classes': train_dataset.classes
            }
            torch.save(checkpoint, 'balanced_49_training_results/balanced_49each_model.pth')
            print(f"✅ New best model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    print(f"\n🎯 Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    return best_val_acc

if __name__ == "__main__":
    best_acc = train_model()
    print(f"\n✅ TRAINING COMPLETE: Balanced 49-per-class model trained with {best_acc:.2f}% validation accuracy")
    print("🎯 Ready for comprehensive evaluation and baseline comparison")
