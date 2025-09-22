#!/usr/bin/env python3
"""
Retrain a balanced 4-class model without doctor bias.
Uses equal class weights and balanced sampling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from collections import Counter
import time

# Import existing components
from load_75_9_checkpoint import DoctorFocusedModel

# VideoDataset class definition (copied from comprehensive_lip_reading_classifier)
import cv2
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, manifest_df, transform=None):
        self.manifest = manifest_df.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(manifest_df['class'].unique()))}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        video_path = row['video_path']
        class_name = row['class']

        # Load video frames
        frames = self.load_video_frames(video_path)

        # Convert to tensor
        video_tensor = torch.FloatTensor(frames).unsqueeze(0)  # Add channel dimension

        # Get class index
        class_idx = self.class_to_idx[class_name]

        return video_tensor, class_idx

    def load_video_frames(self, video_path):
        """Load and preprocess video frames."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale and normalize
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = gray.astype(np.float32) / 255.0
            frames.append(gray)

        cap.release()

        # Sample 32 frames
        if len(frames) >= 32:
            indices = np.linspace(0, len(frames) - 1, 32, dtype=int)
            frames = [frames[i] for i in indices]
        else:
            # Pad with last frame if too short
            while len(frames) < 32:
                frames.append(frames[-1] if frames else np.zeros((64, 96), dtype=np.float32))

        return np.array(frames[:32])

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BalancedTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.class_to_idx = None
        self.idx_to_class = None
        
    def load_data(self):
        """Load training and validation data with balanced sampling."""
        logger.info("🔄 Loading balanced training data...")
        
        # Load manifests
        train_manifest = pd.read_csv('backup_75.9_success_20250921_004410/demographic_train_manifest.csv')
        val_manifest = pd.read_csv('backup_75.9_success_20250921_004410/demographic_validation_manifest.csv')
        
        logger.info(f"📊 Training videos: {len(train_manifest)}")
        logger.info(f"📊 Validation videos: {len(val_manifest)}")
        
        # Analyze class distribution
        train_classes = train_manifest['class'].value_counts()
        val_classes = val_manifest['class'].value_counts()
        
        logger.info("📊 Training class distribution:")
        for class_name, count in train_classes.items():
            logger.info(f"   {class_name}: {count} videos")
            
        logger.info("📊 Validation class distribution:")
        for class_name, count in val_classes.items():
            logger.info(f"   {class_name}: {count} videos")
        
        # Create class mappings
        all_classes = sorted(train_manifest['class'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        logger.info(f"🎯 Classes: {all_classes}")
        
        # Create datasets
        train_paths = train_manifest['video_path'].tolist()
        train_labels = [self.class_to_idx[cls] for cls in train_manifest['class']]
        val_paths = val_manifest['video_path'].tolist()
        val_labels = [self.class_to_idx[cls] for cls in val_manifest['class']]

        train_dataset = VideoDataset(train_paths, train_labels)
        val_dataset = VideoDataset(val_paths, val_labels)
        
        # Calculate balanced sampling weights
        class_counts = Counter([train_dataset[i][1] for i in range(len(train_dataset))])
        total_samples = len(train_dataset)
        
        # Equal weight for all classes
        class_weights = {cls_idx: total_samples / (len(class_counts) * count) 
                        for cls_idx, count in class_counts.items()}
        
        sample_weights = [class_weights[train_dataset[i][1]] for i in range(len(train_dataset))]
        
        logger.info("⚖️ Balanced sampling weights:")
        for cls_idx, weight in class_weights.items():
            class_name = self.idx_to_class[cls_idx]
            logger.info(f"   {class_name}: {weight:.3f}")
        
        # Create balanced sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=8, 
            sampler=sampler,
            num_workers=2
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=8, 
            shuffle=False,
            num_workers=2
        )
        
        logger.info("✅ Data loaders created with balanced sampling")
        
    def create_model(self):
        """Create a fresh model without bias."""
        logger.info("🔄 Creating balanced model...")
        
        num_classes = len(self.class_to_idx)
        self.model = DoctorFocusedModel(num_classes=num_classes)
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"📊 Model parameters: {total_params:,}")
        
        return self.model
    
    def train_balanced_model(self, epochs=25):
        """Train model with balanced approach."""
        logger.info("🚀 Starting balanced training...")
        
        # Equal class weights for loss function
        num_classes = len(self.class_to_idx)
        class_weights = torch.ones(num_classes).to(self.device)  # Equal weights
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_acc = 0.0
        best_model_state = None
        
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
                    logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            train_acc = 100.0 * train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            class_correct = {i: 0 for i in range(num_classes)}
            class_total = {i: 0 for i in range(num_classes)}
            
            with torch.no_grad():
                for videos, labels in self.val_loader:
                    videos, labels = videos.to(self.device), labels.to(self.device)
                    outputs = self.model(videos)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    # Per-class accuracy
                    for i in range(labels.size(0)):
                        label = labels[i].item()
                        class_total[label] += 1
                        if predicted[i] == labels[i]:
                            class_correct[label] += 1
            
            val_acc = 100.0 * val_correct / val_total
            
            # Per-class accuracies
            logger.info(f"\nEpoch {epoch+1}/{epochs} Results:")
            logger.info(f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
            logger.info("Per-class validation accuracy:")
            
            for cls_idx in range(num_classes):
                class_name = self.idx_to_class[cls_idx]
                if class_total[cls_idx] > 0:
                    cls_acc = 100.0 * class_correct[cls_idx] / class_total[cls_idx]
                    logger.info(f"   {class_name}: {cls_acc:.1f}% ({class_correct[cls_idx]}/{class_total[cls_idx]})")
                else:
                    logger.info(f"   {class_name}: No samples")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                logger.info(f"🎯 New best validation accuracy: {best_val_acc:.2f}%")
            
            scheduler.step(val_loss)
            
        # Save best model
        if best_model_state:
            save_path = "balanced_4class_model.pth"
            torch.save({
                'model_state_dict': best_model_state,
                'class_to_idx': self.class_to_idx,
                'idx_to_class': self.idx_to_class,
                'best_val_acc': best_val_acc,
                'model_config': {
                    'num_classes': num_classes,
                    'architecture': 'DoctorFocusedModel'
                }
            }, save_path)
            
            logger.info(f"✅ Best model saved: {save_path}")
            logger.info(f"✅ Best validation accuracy: {best_val_acc:.2f}%")
            
        return best_val_acc

def main():
    print("🔄 BALANCED 4-CLASS MODEL TRAINING")
    print("=" * 50)
    
    trainer = BalancedTrainer()
    
    # Load data
    trainer.load_data()
    
    # Create model
    trainer.create_model()
    
    # Train model
    best_acc = trainer.train_balanced_model(epochs=30)
    
    print(f"\n✅ Training completed!")
    print(f"✅ Best validation accuracy: {best_acc:.2f}%")
    print(f"✅ Model saved as: balanced_4class_model.pth")

if __name__ == "__main__":
    main()
