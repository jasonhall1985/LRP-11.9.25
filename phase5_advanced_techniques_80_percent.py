#!/usr/bin/env python3
"""
Phase 5: Advanced Techniques for 80% Accuracy
Transfer learning, multi-scale processing, ensemble methods
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2
import json
import time
from datetime import datetime
from pathlib import Path
import logging
import psutil
from sklearn.metrics import accuracy_score, f1_score, classification_report
import math
from collections import defaultdict

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MultiScaleLipDataset(Dataset):
    """Multi-scale dataset for advanced training."""
    
    def __init__(self, video_paths, labels, split='train', scales=[64, 112, 160]):
        self.video_paths = video_paths
        self.labels = labels
        self.split = split
        self.scales = scales
        self.augment = split == 'train'
        
        self.class_to_idx = {'doctor': 0, 'glasses': 1, 'help': 2, 'phone': 3, 'pillow': 4}
        print(f"📊 {split.upper()} Multi-Scale Dataset: {len(self.video_paths)} videos")
        
    def __len__(self):
        return len(self.video_paths)
    
    def load_video_multiscale(self, video_path):
        """Load video at multiple scales."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                frames.append(frame)
            
            cap.release()
            
            # Temporal sampling to 32 frames
            if len(frames) >= 32:
                indices = np.linspace(0, len(frames) - 1, 32, dtype=int)
                frames = [frames[i] for i in indices]
            else:
                while len(frames) < 32:
                    frames.append(frames[-1] if frames else np.zeros((480, 640), dtype=np.uint8))
            
            frames = np.array(frames[:32])
            
            # Create multi-scale versions
            multiscale_frames = {}
            for scale in self.scales:
                scaled_frames = []
                for frame in frames:
                    # Resize to 224x224 first, then center crop to target scale
                    frame_224 = cv2.resize(frame, (224, 224))
                    if scale == 224:
                        scaled_frame = frame_224
                    else:
                        # Center crop to target scale
                        h, w = frame_224.shape
                        start_h, start_w = (h - scale) // 2, (w - scale) // 2
                        scaled_frame = frame_224[start_h:start_h + scale, start_w:start_w + scale]
                    scaled_frames.append(scaled_frame)
                
                multiscale_frames[scale] = np.array(scaled_frames)
            
            return multiscale_frames
            
        except Exception as e:
            print(f"❌ Error loading video {video_path}: {str(e)}")
            return {scale: np.zeros((32, scale, scale), dtype=np.uint8) for scale in self.scales}
    
    def apply_advanced_augmentations(self, multiscale_frames):
        """Apply advanced augmentations to all scales."""
        if not self.augment:
            return multiscale_frames
        
        augmented = {}
        
        for scale, frames in multiscale_frames.items():
            # Copy frames for augmentation
            aug_frames = frames.copy()
            
            # Temporal augmentations
            if random.random() < 0.3:
                # Temporal jitter
                jitter = random.randint(-2, 2)
                if jitter > 0:
                    aug_frames = aug_frames[jitter:]
                    aug_frames = np.pad(aug_frames, ((0, jitter), (0, 0), (0, 0)), mode='edge')
                elif jitter < 0:
                    aug_frames = aug_frames[:jitter]
                    aug_frames = np.pad(aug_frames, ((-jitter, 0), (0, 0), (0, 0)), mode='edge')
            
            # Spatial augmentations
            if random.random() < 0.4:
                # Translation
                max_shift = max(2, scale // 40)  # Scale-aware translation
                dx = random.randint(-max_shift, max_shift)
                dy = random.randint(-max_shift, max_shift)
                if dx != 0 or dy != 0:
                    aug_frames = np.roll(aug_frames, (dy, dx), axis=(1, 2))
            
            if random.random() < 0.3:
                # Scale variation
                scale_factor = random.uniform(0.95, 1.05)
                h, w = aug_frames.shape[1], aug_frames.shape[2]
                new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                
                for i in range(len(aug_frames)):
                    frame = cv2.resize(aug_frames[i], (new_w, new_h))
                    if scale_factor > 1.0:  # Crop center
                        start_h, start_w = (new_h - h) // 2, (new_w - w) // 2
                        frame = frame[start_h:start_h + h, start_w:start_w + w]
                    else:  # Pad
                        pad_h, pad_w = (h - new_h) // 2, (w - new_w) // 2
                        frame = np.pad(frame, ((pad_h, h - new_h - pad_h), (pad_w, w - new_w - pad_w)), mode='edge')
                    aug_frames[i] = frame
            
            # Appearance augmentations
            if random.random() < 0.4:
                # Advanced brightness/contrast
                alpha = random.uniform(0.9, 1.1)  # Contrast
                beta = random.uniform(-10, 10)    # Brightness
                aug_frames = np.clip(aug_frames * alpha + beta, 0, 255).astype(np.uint8)
            
            if random.random() < 0.2:
                # Gaussian noise
                noise = np.random.normal(0, 2, aug_frames.shape).astype(np.int16)
                aug_frames = np.clip(aug_frames.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Advanced cutout (scale-aware)
            if random.random() < 0.2:
                patch_size = max(4, scale // 16)  # Scale-aware patch size
                h, w = aug_frames.shape[1], aug_frames.shape[2]
                
                # Avoid center region (lip area)
                center_h, center_w = h // 2, w // 2
                margin = max(10, scale // 8)
                
                # Choose random location outside center
                if random.random() < 0.5:  # Top or bottom
                    y = random.randint(0, center_h - margin - patch_size) if random.random() < 0.5 else random.randint(center_h + margin, h - patch_size)
                    x = random.randint(0, w - patch_size)
                else:  # Left or right
                    y = random.randint(0, h - patch_size)
                    x = random.randint(0, center_w - margin - patch_size) if random.random() < 0.5 else random.randint(center_w + margin, w - patch_size)
                
                aug_frames[:, y:y+patch_size, x:x+patch_size] = 0
            
            augmented[scale] = aug_frames
        
        return augmented
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load multi-scale video
        multiscale_frames = self.load_video_multiscale(video_path)
        
        # Apply advanced augmentations
        multiscale_frames = self.apply_advanced_augmentations(multiscale_frames)
        
        # Normalize and convert to tensors
        multiscale_tensors = {}
        for scale, frames in multiscale_frames.items():
            # Normalize to [0, 1]
            frames = frames.astype(np.float32) / 255.0
            
            # Convert to tensor: (C, T, H, W) format
            frames_tensor = torch.from_numpy(frames).unsqueeze(0)
            multiscale_tensors[scale] = frames_tensor
        
        return multiscale_tensors, label

class MultiScaleModel(nn.Module):
    """Multi-scale model with feature fusion."""
    
    def __init__(self, num_classes=5, scales=[64, 112, 160]):
        super(MultiScaleModel, self).__init__()
        self.scales = scales
        
        # Create separate backbones for each scale
        self.backbones = nn.ModuleDict()
        for scale in scales:
            self.backbones[str(scale)] = self.create_backbone()
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(128 * len(scales), 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Enhanced head with attention
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.1)
        self.layer_norm = nn.LayerNorm(256)
        self.classifier = nn.Linear(256, num_classes)
        
    def create_backbone(self):
        """Create 3D CNN backbone."""
        return nn.Sequential(
            # First conv block
            nn.Conv3d(1, 32, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            # Second conv block
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            # Third conv block
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
    
    def forward(self, multiscale_inputs):
        # Extract features from each scale
        scale_features = []
        
        for scale in self.scales:
            x = multiscale_inputs[scale]
            features = self.backbones[str(scale)](x)  # (batch, 128, 1, 1, 1)
            features = features.flatten(1)  # (batch, 128)
            scale_features.append(features)
        
        # Concatenate multi-scale features
        fused_features = torch.cat(scale_features, dim=1)  # (batch, 128 * num_scales)
        
        # Feature fusion
        fused_features = self.fusion(fused_features)  # (batch, 256)
        
        # Self-attention
        # Reshape for attention: (seq_len, batch, embed_dim)
        attn_input = fused_features.unsqueeze(0)  # (1, batch, 256)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        attn_output = attn_output.squeeze(0)  # (batch, 256)
        
        # Residual connection and layer norm
        output = self.layer_norm(attn_output + fused_features)
        
        # Final classification
        output = self.classifier(output)
        
        return output

class EnsembleModel(nn.Module):
    """Ensemble of multiple models for maximum accuracy."""
    
    def __init__(self, models, weights=None):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights if weights is not None else [1.0] * len(models)
        
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Weighted average
        weighted_output = sum(w * out for w, out in zip(self.weights, outputs))
        return weighted_output / sum(self.weights)

class AdvancedTrainer:
    """Advanced trainer with transfer learning and ensemble methods."""
    
    def __init__(self, model, train_loader, val_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Advanced loss with focal loss for hard examples
        self.criterion = self.create_focal_loss(alpha=1.0, gamma=2.0)
        
        # Advanced optimizer with different learning rates
        self.optimizer = self.create_advanced_optimizer()
        
        # Advanced scheduler
        self.scheduler = self.create_advanced_scheduler()
        
        # Training state
        self.epoch = 0
        self.best_val_f1 = 0.0
        self.train_losses = []
        self.val_accuracies = []
        
        # Setup logging
        self.setup_logging()
        
    def create_focal_loss(self, alpha=1.0, gamma=2.0):
        """Create focal loss for handling hard examples."""
        class FocalLoss(nn.Module):
            def __init__(self, alpha=1.0, gamma=2.0):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
                
            def forward(self, inputs, targets):
                ce_loss = F.cross_entropy(inputs, targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                return focal_loss.mean()
        
        return FocalLoss(alpha, gamma)
    
    def create_advanced_optimizer(self):
        """Create advanced optimizer with parameter groups."""
        # Separate parameters by type
        backbone_params = []
        fusion_params = []
        attention_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if 'backbones' in name:
                backbone_params.append(param)
            elif 'fusion' in name:
                fusion_params.append(param)
            elif 'attention' in name or 'layer_norm' in name:
                attention_params.append(param)
            else:
                classifier_params.append(param)
        
        return optim.AdamW([
            {'params': backbone_params, 'lr': 1e-4, 'weight_decay': 1e-4},
            {'params': fusion_params, 'lr': 3e-4, 'weight_decay': 1e-4},
            {'params': attention_params, 'lr': 5e-4, 'weight_decay': 1e-5},
            {'params': classifier_params, 'lr': 8e-4, 'weight_decay': 1e-4}
        ])
    
    def create_advanced_scheduler(self):
        """Create advanced learning rate scheduler."""
        return optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[1e-4, 3e-4, 5e-4, 8e-4],
            epochs=15,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
    
    def setup_logging(self):
        """Setup logging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = f"advanced_training_{timestamp}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.experiment_dir}/training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🚀 Advanced training started: {self.experiment_dir}")
    
    def train_epoch(self):
        """Train for one epoch with advanced techniques."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Move multi-scale data to device
            multiscale_data = {}
            for scale, scale_data in data.items():
                multiscale_data[scale] = scale_data.to(self.device)
            target = target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(multiscale_data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Advanced gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy())
            
            if batch_idx % 10 == 0:
                memory_mb = get_memory_usage()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(f"Epoch {self.epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                               f"Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%, "
                               f"LR: {current_lr:.2e}, Memory: {memory_mb:.1f}MB")
        
        avg_loss = total_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        train_f1 = f1_score(all_targets, all_preds, average='macro') * 100
        
        self.train_losses.append(avg_loss)
        return avg_loss, train_acc, train_f1
    
    def validate(self):
        """Validate with advanced TTA."""
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                # Move multi-scale data to device
                multiscale_data = {}
                for scale, scale_data in data.items():
                    multiscale_data[scale] = scale_data.to(self.device)
                target = target.to(self.device)
                
                # Advanced TTA with multiple predictions
                outputs = []
                
                # Original prediction
                outputs.append(self.model(multiscale_data))
                
                # Temporal variations
                for scale in multiscale_data.keys():
                    original_data = multiscale_data[scale]
                    if original_data.size(2) >= 32:
                        # Multiple temporal crops
                        for start_idx in [0, 2, 4]:
                            if start_idx + 28 <= original_data.size(2):
                                temp_data = multiscale_data.copy()
                                crop_data = original_data[:, :, start_idx:start_idx+28, :, :]
                                crop_data = F.pad(crop_data, (0, 0, 0, 0, 0, 4))
                                temp_data[scale] = crop_data
                                outputs.append(self.model(temp_data))
                
                # Average all predictions
                output = torch.stack(outputs).mean(0)
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        val_acc = 100. * correct / total
        val_f1 = f1_score(all_targets, all_preds, average='macro') * 100
        
        self.val_accuracies.append(val_acc)
        return val_acc, val_f1, all_preds, all_targets
    
    def train(self, num_epochs=15):
        """Main advanced training loop."""
        self.logger.info(f"🎯 Starting advanced training for {num_epochs} epochs")
        
        patience_counter = 0
        max_patience = 6
        
        for epoch in range(num_epochs):
            self.epoch = epoch + 1
            
            # Train epoch
            train_loss, train_acc, train_f1 = self.train_epoch()
            
            # Validate
            val_acc, val_f1, val_preds, val_targets = self.validate()
            
            # Check for improvement
            is_best = val_f1 > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_f1
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), f"{self.experiment_dir}/best_model.pth")
                self.logger.info(f"💾 New best model saved: {val_f1:.2f}%")
            else:
                patience_counter += 1
            
            # Log progress
            memory_mb = get_memory_usage()
            self.logger.info(f"Epoch {self.epoch}/{num_epochs} - "
                           f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train F1: {train_f1:.2f}%, "
                           f"Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.2f}%, "
                           f"Best F1: {self.best_val_f1:.2f}%, Memory: {memory_mb:.1f}MB")
            
            # Early stopping
            if patience_counter >= max_patience:
                self.logger.info(f"⏹️  Early stopping after {patience_counter} epochs without improvement")
                break
        
        # Final test evaluation
        return self.final_test()
    
    def final_test(self):
        """Final test evaluation with comprehensive TTA."""
        self.logger.info("🔍 Final test evaluation with comprehensive TTA...")
        
        # Load best model
        best_model_path = f"{self.experiment_dir}/best_model.pth"
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            self.logger.info("📥 Loaded best model")
        
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                # Move multi-scale data to device
                multiscale_data = {}
                for scale, scale_data in data.items():
                    multiscale_data[scale] = scale_data.to(self.device)
                target = target.to(self.device)
                
                # Comprehensive TTA
                outputs = []
                
                # Original
                outputs.append(self.model(multiscale_data))
                
                # Multiple temporal and spatial variations
                for _ in range(5):  # 5 different augmentations
                    temp_data = {}
                    for scale, scale_data in multiscale_data.items():
                        # Random temporal crop
                        if scale_data.size(2) >= 32:
                            start_idx = random.randint(0, min(4, scale_data.size(2) - 28))
                            crop_data = scale_data[:, :, start_idx:start_idx+28, :, :]
                            crop_data = F.pad(crop_data, (0, 0, 0, 0, 0, 4))
                        else:
                            crop_data = scale_data
                        
                        # Random spatial jitter
                        dx, dy = random.randint(-2, 2), random.randint(-2, 2)
                        crop_data = torch.roll(crop_data, shifts=(dx, dy), dims=(3, 4))
                        
                        temp_data[scale] = crop_data
                    
                    outputs.append(self.model(temp_data))
                
                # Average all predictions
                output = torch.stack(outputs).mean(0)
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        test_acc = 100. * correct / total
        test_f1 = f1_score(all_targets, all_preds, average='macro') * 100
        
        # Generate classification report
        class_names = ['doctor', 'glasses', 'help', 'phone', 'pillow']
        report = classification_report(all_targets, all_preds, target_names=class_names)
        
        self.logger.info(f"🎯 ADVANCED FINAL RESULTS:")
        self.logger.info(f"   • Test Accuracy: {test_acc:.2f}%")
        self.logger.info(f"   • Test F1 Score: {test_f1:.2f}%")
        self.logger.info(f"   • Best Val F1: {self.best_val_f1:.2f}%")
        self.logger.info(f"📊 Classification Report:\n{report}")
        
        # Save results
        results = {
            'test_accuracy': test_acc,
            'test_f1_score': test_f1,
            'best_val_f1_score': self.best_val_f1,
            'total_epochs': self.epoch,
            'classification_report': report
        }
        
        with open(f"{self.experiment_dir}/final_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return test_acc

def create_balanced_splits(dataset_path="corrected_balanced_dataset"):
    """Create balanced splits for advanced training."""
    print("📊 Creating balanced data splits for advanced training...")
    
    video_files = list(Path(dataset_path).glob("*.mp4"))
    if len(video_files) == 0:
        raise ValueError(f"No video files found in {dataset_path}")
    
    print(f"Found {len(video_files)} videos")
    
    # Organize by class
    class_videos = {'doctor': [], 'glasses': [], 'help': [], 'phone': [], 'pillow': []}
    
    for video_file in video_files:
        class_name = video_file.stem.split('_')[0]
        if class_name in class_videos:
            class_videos[class_name].append(str(video_file))
    
    # Create balanced splits
    train_videos, train_labels = [], []
    val_videos, val_labels = [], []
    test_videos, test_labels = [], []
    
    class_to_idx = {'doctor': 0, 'glasses': 1, 'help': 2, 'phone': 3, 'pillow': 4}
    
    for class_name, videos in class_videos.items():
        random.shuffle(videos)
        
        train_videos.extend(videos[:8])
        train_labels.extend([class_to_idx[class_name]] * 8)
        
        if len(videos) > 8:
            val_videos.append(videos[8])
            val_labels.append(class_to_idx[class_name])
        
        if len(videos) > 9:
            test_videos.append(videos[9])
            test_labels.append(class_to_idx[class_name])
    
    print(f"📊 Splits: Train={len(train_videos)}, Val={len(val_videos)}, Test={len(test_videos)}")
    return (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels)

def main():
    """Main advanced training for 80% accuracy."""
    print("🎯 PHASE 5: ADVANCED TECHNIQUES FOR 80% ACCURACY")
    print("=" * 70)
    print("ADVANCED FEATURES:")
    print("• Multi-scale processing (64x64, 112x112, 160x160)")
    print("• Feature fusion with attention")
    print("• Focal loss for hard examples")
    print("• Advanced augmentations")
    print("• Comprehensive TTA")
    print("• OneCycle learning rate scheduling")
    print("• TARGET: 80%+ accuracy")
    print("=" * 70)
    print(f"💾 Initial memory: {get_memory_usage():.1f} MB")
    
    # Set random seeds
    set_random_seeds(42)
    
    # Device
    device = torch.device('cpu')
    print(f"🖥️  Device: {device}")
    
    # Create balanced data splits
    (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels) = create_balanced_splits()
    
    # Create multi-scale datasets
    scales = [64, 112, 160]
    train_dataset = MultiScaleLipDataset(train_videos, train_labels, split='train', scales=scales)
    val_dataset = MultiScaleLipDataset(val_videos, val_labels, split='val', scales=scales)
    test_dataset = MultiScaleLipDataset(test_videos, test_labels, split='test', scales=scales)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"💾 After data loading: {get_memory_usage():.1f} MB")
    
    # Create multi-scale model
    model = MultiScaleModel(num_classes=5, scales=scales).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"🧠 Advanced Multi-Scale Model:")
    print(f"   • Total parameters: {total_params:,}")
    print(f"   • Scales: {scales}")
    print(f"   • Features: Multi-scale fusion + attention")
    print(f"   • Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    print(f"💾 After model creation: {get_memory_usage():.1f} MB")
    
    # Create advanced trainer
    trainer = AdvancedTrainer(model, train_loader, val_loader, test_loader, device)
    
    print(f"💾 After trainer setup: {get_memory_usage():.1f} MB")
    
    # Start advanced training
    print("\n🎯 STARTING ADVANCED TRAINING FOR 80% ACCURACY")
    print("=" * 60)
    print("Multi-scale processing + Feature fusion + Advanced TTA")
    print("=" * 60)
    
    start_time = time.time()
    final_accuracy = trainer.train(num_epochs=15)
    training_time = (time.time() - start_time) / 60
    
    print(f"\n🎉 ADVANCED TRAINING COMPLETED!")
    print("=" * 50)
    print(f"🎯 Final Test Accuracy: {final_accuracy:.2f}%")
    print(f"⏱️  Training Time: {training_time:.1f} minutes")
    print(f"💾 Final Memory Usage: {get_memory_usage():.1f} MB")
    print(f"📁 Results Directory: {trainer.experiment_dir}")
    
    # Success evaluation
    if final_accuracy >= 80:
        print(f"🏆 MISSION ACCOMPLISHED: 80% target achieved! ({final_accuracy:.1f}%)")
    elif final_accuracy >= 75:
        print(f"🌟 EXCELLENT: Very close to target! ({final_accuracy:.1f}%)")
    elif final_accuracy >= 70:
        print(f"✅ GREAT: Significant improvement! ({final_accuracy:.1f}%)")
    elif final_accuracy >= 60:
        print(f"📈 GOOD: Solid progress! ({final_accuracy:.1f}%)")
    else:
        print(f"⚠️  NEEDS MORE WORK: ({final_accuracy:.1f}%)")
    
    print(f"\n📊 FINAL IMPROVEMENT SUMMARY:")
    baseline = 40.0
    improvement = final_accuracy - baseline
    print(f"   • Original baseline: 40.0%")
    print(f"   • Advanced techniques result: {final_accuracy:.1f}%")
    print(f"   • Total improvement: {improvement:+.1f} percentage points")
    print(f"   • Relative improvement: {((final_accuracy / baseline) - 1) * 100:+.1f}%")
    
    return final_accuracy

if __name__ == "__main__":
    try:
        final_accuracy = main()
        
        if final_accuracy >= 80:
            print(f"\n🎯 80% ACCURACY ACHIEVED!")
            print(f"   Advanced techniques successful!")
        else:
            print(f"\n🔄 CONTINUE OPTIMIZATION:")
            print(f"   Consider ensemble methods or more data")
            
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
