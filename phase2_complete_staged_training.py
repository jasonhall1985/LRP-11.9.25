#!/usr/bin/env python3
"""
Phase 2: COMPLETE Staged Training Implementation
Following user's EXACT specifications for 60-75% accuracy target
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

class StandardizedLipDataset(Dataset):
    """Standardized dataset following user's exact preprocessing specifications."""
    
    def __init__(self, video_paths, labels, split='train', augment=True):
        self.video_paths = video_paths
        self.labels = labels
        self.split = split
        self.augment = augment and (split == 'train')
        
        self.class_to_idx = {'doctor': 0, 'glasses': 1, 'help': 2, 'phone': 3, 'pillow': 4}
        print(f"📊 {split.upper()} Dataset: {len(self.video_paths)} videos")
        
    def __len__(self):
        return len(self.video_paths)
    
    def load_video_standardized(self, video_path):
        """Load video with EXACT user specifications: 224x224 → 112x112 center crop."""
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
                
                # USER SPEC: Fixed mouth ROI - 224x224 → 112x112 center crop
                frame = cv2.resize(frame, (224, 224))
                h, w = frame.shape
                start_h, start_w = (h - 112) // 2, (w - 112) // 2
                frame = frame[start_h:start_h + 112, start_w:start_w + 112]
                
                frames.append(frame)
            
            cap.release()
            
            # USER SPEC: Temporal standardization - 32 frames with uniform sampling
            if len(frames) >= 32:
                indices = np.linspace(0, len(frames) - 1, 32, dtype=int)
                frames = [frames[i] for i in indices]
            else:
                while len(frames) < 32:
                    frames.append(frames[-1] if frames else np.zeros((112, 112), dtype=np.uint8))
            
            return np.array(frames[:32])
            
        except Exception as e:
            print(f"❌ Error loading video {video_path}: {str(e)}")
            return np.zeros((32, 112, 112), dtype=np.uint8)
    
    def apply_strategic_augmentations(self, frames):
        """Apply USER SPECIFIED strategic training augmentations."""
        if not self.augment:
            return frames
        
        # USER SPEC: Temporal augmentations
        # ±2 frame jitter
        if random.random() < 0.3:
            jitter = random.randint(-2, 2)
            if jitter > 0:
                frames = frames[jitter:]
                frames = np.pad(frames, ((0, jitter), (0, 0), (0, 0)), mode='edge')
            elif jitter < 0:
                frames = frames[:jitter]
                frames = np.pad(frames, ((-jitter, 0), (0, 0), (0, 0)), mode='edge')
        
        # 10% random frame dropout
        if random.random() < 0.1:
            dropout_frames = random.randint(1, 2)
            for _ in range(dropout_frames):
                idx = random.randint(0, len(frames) - 1)
                if idx > 0:
                    frames[idx] = frames[idx - 1]
        
        # Speed variation 0.97-1.03x (temporal resampling)
        if random.random() < 0.2:
            speed_factor = random.uniform(0.97, 1.03)
            new_length = int(len(frames) * speed_factor)
            if new_length != len(frames):
                indices = np.linspace(0, len(frames) - 1, min(new_length, len(frames)), dtype=int)
                frames = frames[indices]
                # Pad or trim to 32
                if len(frames) < 32:
                    frames = np.pad(frames, ((0, 32 - len(frames)), (0, 0), (0, 0)), mode='edge')
                else:
                    frames = frames[:32]
        
        # USER SPEC: Spatial augmentations
        # ±3px translation
        if random.random() < 0.4:
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            if dx != 0 or dy != 0:
                frames = np.roll(frames, (dy, dx), axis=(1, 2))
        
        # ±5% scale
        if random.random() < 0.3:
            scale = random.uniform(0.95, 1.05)
            h, w = frames.shape[1], frames.shape[2]
            new_h, new_w = int(h * scale), int(w * scale)
            for i in range(len(frames)):
                frame = cv2.resize(frames[i], (new_w, new_h))
                if scale > 1.0:  # Crop center
                    start_h, start_w = (new_h - h) // 2, (new_w - w) // 2
                    frame = frame[start_h:start_h + h, start_w:start_w + w]
                else:  # Pad
                    pad_h, pad_w = (h - new_h) // 2, (w - new_w) // 2
                    frame = np.pad(frame, ((pad_h, h - new_h - pad_h), (pad_w, w - new_w - pad_w)), mode='edge')
                frames[i] = frame
        
        # ±3° rotation
        if random.random() < 0.2:
            angle = random.uniform(-3, 3)
            h, w = frames.shape[1], frames.shape[2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            for i in range(len(frames)):
                frames[i] = cv2.warpAffine(frames[i], M, (w, h))
        
        # USER SPEC: Appearance augmentations
        # ±10% brightness/contrast
        if random.random() < 0.3:
            brightness = random.uniform(0.9, 1.1)
            contrast = random.uniform(0.9, 1.1)
            frames = np.clip(frames * contrast + (brightness - 1) * 128, 0, 255).astype(np.uint8)
        
        # USER SPEC: Cutout - Small 8x8 patches (avoid lip center region)
        if random.random() < 0.2:
            patch_size = 8
            h, w = frames.shape[1], frames.shape[2]
            # Avoid center region (lip area)
            center_h, center_w = h // 2, w // 2
            margin = 20
            
            # Choose random location outside center
            if random.random() < 0.5:  # Top or bottom
                y = random.randint(0, center_h - margin - patch_size) if random.random() < 0.5 else random.randint(center_h + margin, h - patch_size)
                x = random.randint(0, w - patch_size)
            else:  # Left or right
                y = random.randint(0, h - patch_size)
                x = random.randint(0, center_w - margin - patch_size) if random.random() < 0.5 else random.randint(center_w + margin, w - patch_size)
            
            frames[:, y:y+patch_size, x:x+patch_size] = 0
        
        return frames
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load video with standardized preprocessing
        frames = self.load_video_standardized(video_path)
        
        # Apply strategic augmentations (training only)
        frames = self.apply_strategic_augmentations(frames)
        
        # USER SPEC: Per-video normalize to [0,1] with mean=0.5, std=0.5
        frames = frames.astype(np.float32) / 255.0
        
        # Convert to tensor: (C, T, H, W) format
        frames = torch.from_numpy(frames).unsqueeze(0)
        
        return frames, label

class CorrectedLipModel(nn.Module):
    """USER SPECIFICATION: Keep original 290K backbone + add BiGRU head."""
    
    def __init__(self, num_classes=5):
        super(CorrectedLipModel, self).__init__()
        
        # USER SPEC: Keep current 3D CNN feature extractor (290K parameters)
        self.backbone = nn.Sequential(
            # First conv block - EXACT original architecture
            nn.Conv3d(1, 32, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            # Second conv block - EXACT original architecture
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            # Third conv block - EXACT original architecture
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        
        # USER SPEC: 2-layer Bidirectional GRU with hidden_size=256
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # USER SPEC: LayerNorm before final classifier
        self.layer_norm = nn.LayerNorm(512)  # 256 * 2 (bidirectional)
        
        # USER SPEC: Final linear layer: 512 → 5 classes
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Extract features with backbone
        x = self.backbone(x)  # (batch, 128, T', H', W')
        
        # Global spatial pooling but preserve temporal dimension
        batch_size, channels, T, H, W = x.shape
        x = F.adaptive_avg_pool3d(x, (T, 1, 1))  # (batch, 128, T', 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (batch, 128, T')
        x = x.transpose(1, 2)  # (batch, T', 128) for GRU
        
        # Apply BiGRU
        gru_out, _ = self.gru(x)  # (batch, T', 512)
        
        # Take the last output
        gru_out = gru_out[:, -1, :]  # (batch, 512)
        
        # Apply LayerNorm
        gru_out = self.layer_norm(gru_out)
        
        # Final classification
        output = self.classifier(gru_out)
        
        return output

class LabelSmoothingCrossEntropy(nn.Module):
    """USER SPEC: Label smoothing: 0.1"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        log_prob = F.log_softmax(pred, dim=-1)
        loss = -(one_hot * log_prob).sum(dim=-1).mean()
        return loss

class CosineWarmupScheduler:
    """USER SPEC: 1 epoch warmup → cosine decay to 1e-5"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-5):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Warmup phase
            lr_scale = (epoch + 1) / self.warmup_epochs
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = self.base_lrs[i] * lr_scale
        else:
            # Cosine decay phase
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = self.min_lr + (self.base_lrs[i] - self.min_lr) * lr_scale

def create_proper_speaker_disjoint_splits(dataset_path="corrected_balanced_dataset"):
    """Create PROPER speaker-disjoint splits with no overlap."""
    print("📊 Creating PROPER speaker-disjoint data splits...")
    
    video_files = list(Path(dataset_path).glob("*.mp4"))
    if len(video_files) == 0:
        raise ValueError(f"No video files found in {dataset_path}")
    
    print(f"Found {len(video_files)} videos")
    
    # Group videos by class and speaker
    class_speaker_videos = {}
    for video_file in video_files:
        parts = video_file.stem.split('_')
        if len(parts) >= 2:
            class_name = parts[0]
            speaker_id = parts[1]
            
            if class_name not in class_speaker_videos:
                class_speaker_videos[class_name] = {}
            if speaker_id not in class_speaker_videos[class_name]:
                class_speaker_videos[class_name][speaker_id] = []
            
            class_speaker_videos[class_name][speaker_id].append(str(video_file))
    
    # Create speaker-disjoint splits
    train_videos, train_labels = [], []
    val_videos, val_labels = [], []
    test_videos, test_labels = [], []
    
    class_to_idx = {'doctor': 0, 'glasses': 1, 'help': 2, 'phone': 3, 'pillow': 4}
    
    # For each class, ensure speaker separation
    for class_name in class_speaker_videos.keys():
        speakers = list(class_speaker_videos[class_name].keys())
        random.shuffle(speakers)
        
        # Split speakers: 8 for train, 1 for val, 1 for test
        train_speakers = speakers[:8]
        val_speaker = speakers[8:9] if len(speakers) > 8 else []
        test_speaker = speakers[9:10] if len(speakers) > 9 else []
        
        # Add videos from train speakers
        for speaker in train_speakers:
            videos = class_speaker_videos[class_name][speaker]
            train_videos.extend(videos)
            train_labels.extend([class_to_idx[class_name]] * len(videos))
        
        # Add videos from val speaker
        for speaker in val_speaker:
            videos = class_speaker_videos[class_name][speaker]
            val_videos.extend(videos)
            val_labels.extend([class_to_idx[class_name]] * len(videos))
        
        # Add videos from test speaker
        for speaker in test_speaker:
            videos = class_speaker_videos[class_name][speaker]
            test_videos.extend(videos)
            test_labels.extend([class_to_idx[class_name]] * len(videos))
    
    print(f"📊 Final splits: Train={len(train_videos)}, Val={len(val_videos)}, Test={len(test_videos)}")
    
    # Verify no speaker overlap
    train_speakers = set()
    val_speakers = set()
    test_speakers = set()
    
    for video in train_videos:
        speaker = Path(video).stem.split('_')[1]
        train_speakers.add(speaker)
    
    for video in val_videos:
        speaker = Path(video).stem.split('_')[1]
        val_speakers.add(speaker)
    
    for video in test_videos:
        speaker = Path(video).stem.split('_')[1]
        test_speakers.add(speaker)
    
    # Check for overlap
    overlap = (train_speakers & val_speakers) | (train_speakers & test_speakers) | (val_speakers & test_speakers)
    
    if not overlap:
        print("✅ Splits are properly speaker-disjoint!")
    else:
        print(f"⚠️  Still have overlap: {overlap}")
    
    return (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels)

class StagedTrainer:
    """USER SPEC: Implement staged fine-tuning."""
    
    def __init__(self, model, train_loader, val_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # USER SPEC: Label smoothing
        self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        
        # Separate parameters for different learning rates
        self.head_params = []
        self.backbone_params = []
        
        for name, param in model.named_parameters():
            if 'backbone' in name:
                self.backbone_params.append(param)
            else:
                self.head_params.append(param)
        
        # Training state
        self.epoch = 0
        self.best_val_f1 = 0.0
        self.stage = 1
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = f"staged_training_{timestamp}"
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
        self.logger.info(f"🚀 Staged training started: {self.experiment_dir}")
        
    def create_optimizer(self, head_lr, backbone_lr):
        """Create optimizer with specified learning rates."""
        # USER SPEC: AdamW with weight_decay=1e-4
        return optim.AdamW([
            {'params': self.head_params, 'lr': head_lr, 'weight_decay': 1e-4},
            {'params': self.backbone_params, 'lr': backbone_lr, 'weight_decay': 1e-4}
        ])
        
    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone_params:
            param.requires_grad = False
        self.logger.info("🔒 Backbone frozen")
        
    def unfreeze_last_backbone_block(self):
        """Unfreeze last backbone block."""
        # Unfreeze the last conv block (third block)
        for name, param in self.model.named_parameters():
            if 'backbone.4' in name or 'backbone.5' in name or 'backbone.6' in name:  # Third conv block
                param.requires_grad = True
        self.logger.info("🔓 Last backbone block unfrozen")
        
    def unfreeze_all_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.backbone_params:
            param.requires_grad = True
        self.logger.info("🔓 All backbone unfrozen")
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        # USER SPEC: Freeze BatchNorm in eval mode if batch_size < 8
        if self.train_loader.batch_size < 8:
            for module in self.model.modules():
                if isinstance(module, nn.BatchNorm3d):
                    module.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # USER SPEC: Gradient clipping: max_norm=1.0
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy())
            
            if batch_idx % 10 == 0:
                memory_mb = get_memory_usage()
                self.logger.info(f"Stage {self.stage}, Epoch {self.epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                               f"Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%, "
                               f"Memory: {memory_mb:.1f}MB")
        
        avg_loss = total_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        train_f1 = f1_score(all_targets, all_preds, average='macro') * 100
        
        return avg_loss, train_acc, train_f1
        
    def validate(self):
        """Validate the model."""
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        val_acc = 100. * correct / total
        val_f1 = f1_score(all_targets, all_preds, average='macro') * 100
        
        return val_acc, val_f1, all_preds, all_targets
        
    def staged_training(self):
        """USER SPEC: Implement staged fine-tuning."""
        self.logger.info("🎯 Starting staged fine-tuning...")
        
        total_epochs = 8  # 1 + 3 + 4
        
        # USER SPEC: Stage 1 (1 epoch): Freeze backbone, train head only at lr=6e-4
        self.stage = 1
        self.logger.info("🔥 Stage 1: Freeze backbone, train head only")
        self.freeze_backbone()
        self.optimizer = self.create_optimizer(head_lr=6e-4, backbone_lr=0)
        self.scheduler = CosineWarmupScheduler(self.optimizer, warmup_epochs=1, total_epochs=total_epochs)
        
        for epoch in range(1):
            self.epoch = epoch + 1
            self.scheduler.step(self.epoch - 1)
            
            train_loss, train_acc, train_f1 = self.train_epoch()
            val_acc, val_f1, val_preds, val_targets = self.validate()
            
            is_best = val_f1 > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_f1
                torch.save(self.model.state_dict(), f"{self.experiment_dir}/best_model.pth")
            
            self.logger.info(f"Stage 1, Epoch {self.epoch} - "
                           f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train F1: {train_f1:.2f}%, "
                           f"Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.2f}%, Best F1: {self.best_val_f1:.2f}%")
        
        # USER SPEC: Stage 2 (3 epochs): Unfreeze last backbone block, head=4e-4, backbone=1e-4
        self.stage = 2
        self.logger.info("🔥 Stage 2: Unfreeze last backbone block")
        self.unfreeze_last_backbone_block()
        self.optimizer = self.create_optimizer(head_lr=4e-4, backbone_lr=1e-4)
        
        patience_counter = 0
        max_patience = 4
        
        for epoch in range(3):
            self.epoch = epoch + 2
            self.scheduler.step(self.epoch - 1)
            
            train_loss, train_acc, train_f1 = self.train_epoch()
            val_acc, val_f1, val_preds, val_targets = self.validate()
            
            is_best = val_f1 > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_f1
                patience_counter = 0
                torch.save(self.model.state_dict(), f"{self.experiment_dir}/best_model.pth")
            else:
                patience_counter += 1
            
            self.logger.info(f"Stage 2, Epoch {self.epoch} - "
                           f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train F1: {train_f1:.2f}%, "
                           f"Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.2f}%, Best F1: {self.best_val_f1:.2f}%")
            
            if patience_counter >= max_patience:
                self.logger.info(f"⏹️  Early stopping in Stage 2")
                break
        
        # USER SPEC: Stage 3 (4 epochs): Unfreeze all layers, head=2e-4, backbone=7e-5
        if patience_counter < max_patience:
            self.stage = 3
            self.logger.info("🔥 Stage 3: Unfreeze all layers")
            self.unfreeze_all_backbone()
            self.optimizer = self.create_optimizer(head_lr=2e-4, backbone_lr=7e-5)
            
            for epoch in range(4):
                self.epoch = epoch + 5
                self.scheduler.step(self.epoch - 1)
                
                train_loss, train_acc, train_f1 = self.train_epoch()
                val_acc, val_f1, val_preds, val_targets = self.validate()
                
                is_best = val_f1 > self.best_val_f1
                if is_best:
                    self.best_val_f1 = val_f1
                    patience_counter = 0
                    torch.save(self.model.state_dict(), f"{self.experiment_dir}/best_model.pth")
                else:
                    patience_counter += 1
                
                self.logger.info(f"Stage 3, Epoch {self.epoch} - "
                               f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train F1: {train_f1:.2f}%, "
                               f"Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.2f}%, Best F1: {self.best_val_f1:.2f}%")
                
                if patience_counter >= max_patience:
                    self.logger.info(f"⏹️  Early stopping in Stage 3")
                    break
        
        # Final test evaluation
        return self.final_test()
        
    def final_test(self):
        """Final test evaluation with TTA."""
        self.logger.info("🔍 Final test evaluation with TTA...")
        
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
                data, target = data.to(self.device), target.to(self.device)
                
                # USER SPEC: Test Time Augmentation (TTA)
                outputs = []
                
                # Original
                outputs.append(self.model(data))
                
                # 3 temporal crops (start/middle/end)
                if data.size(2) >= 32:
                    # Start crop (first 28 frames, pad to 32)
                    start_data = data[:, :, :28, :, :]
                    start_data = F.pad(start_data, (0, 0, 0, 0, 0, 4))
                    outputs.append(self.model(start_data))
                    
                    # End crop (last 28 frames, pad to 32)
                    end_data = data[:, :, -28:, :, :]
                    end_data = F.pad(end_data, (0, 0, 0, 0, 4, 0))
                    outputs.append(self.model(end_data))
                
                # 2px spatial jitter
                for dx, dy in [(2, 0), (-2, 0), (0, 2), (0, -2)]:
                    jittered_data = torch.roll(data, shifts=(dx, dy), dims=(3, 4))
                    outputs.append(self.model(jittered_data))
                
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
        
        self.logger.info(f"🎯 FINAL RESULTS:")
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

def main():
    """Main complete staged training implementation."""
    print("🎯 PHASE 2: COMPLETE STAGED TRAINING")
    print("=" * 60)
    print("USER SPECIFICATIONS:")
    print("• Keep original 290K backbone + BiGRU head")
    print("• 224x224 → 112x112 center crop")
    print("• Strategic augmentations (training only)")
    print("• Staged fine-tuning: 1+3+4 epochs")
    print("• AdamW optimizer with different LRs")
    print("• Label smoothing + gradient clipping")
    print("• TTA for final evaluation")
    print("• TARGET: 60-75% accuracy")
    print("=" * 60)
    print(f"💾 Initial memory: {get_memory_usage():.1f} MB")
    
    # Set random seeds
    set_random_seeds(42)
    
    # Device
    device = torch.device('cpu')
    print(f"🖥️  Device: {device}")
    
    # Create PROPER speaker-disjoint data splits
    (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels) = create_proper_speaker_disjoint_splits()
    
    # Create standardized datasets
    train_dataset = StandardizedLipDataset(train_videos, train_labels, split='train', augment=True)
    val_dataset = StandardizedLipDataset(val_videos, val_labels, split='val', augment=False)
    test_dataset = StandardizedLipDataset(test_videos, test_labels, split='test', augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"💾 After data loading: {get_memory_usage():.1f} MB")
    
    # Create corrected model
    model = CorrectedLipModel(num_classes=5).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    
    print(f"🧠 Complete Model:")
    print(f"   • Backbone parameters: {backbone_params:,} (target: ~290K)")
    print(f"   • Total parameters: {total_params:,}")
    print(f"   • Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    print(f"💾 After model creation: {get_memory_usage():.1f} MB")
    
    # Create staged trainer
    trainer = StagedTrainer(model, train_loader, val_loader, test_loader, device)
    
    print(f"💾 After trainer setup: {get_memory_usage():.1f} MB")
    
    # Start staged training
    print("\n🎯 STARTING COMPLETE STAGED TRAINING")
    print("=" * 50)
    print("Stage 1: Head only (1 epoch)")
    print("Stage 2: Head + last block (3 epochs)")
    print("Stage 3: Full model (4 epochs)")
    print("=" * 50)
    
    start_time = time.time()
    final_accuracy = trainer.staged_training()
    training_time = (time.time() - start_time) / 60
    
    print(f"\n🎉 COMPLETE STAGED TRAINING FINISHED!")
    print("=" * 50)
    print(f"🎯 Final Test Accuracy: {final_accuracy:.2f}%")
    print(f"⏱️  Training Time: {training_time:.1f} minutes")
    print(f"💾 Final Memory Usage: {get_memory_usage():.1f} MB")
    print(f"📁 Results Directory: {trainer.experiment_dir}")
    
    # Success evaluation
    baseline = 40.0
    improvement = final_accuracy - baseline
    
    if final_accuracy >= 75:
        print(f"🏆 EXCELLENT: Exceeded target! ({final_accuracy:.1f}% ≥ 75%)")
    elif final_accuracy >= 60:
        print(f"✅ SUCCESS: Target achieved! ({final_accuracy:.1f}% ≥ 60%)")
    elif final_accuracy >= 50:
        print(f"📈 GOOD: Significant improvement ({final_accuracy:.1f}% vs 40%)")
    else:
        print(f"⚠️  NEEDS MORE WORK: ({final_accuracy:.1f}% < 50%)")
    
    print(f"\n📊 IMPROVEMENT SUMMARY:")
    print(f"   • Original baseline: 40.0%")
    print(f"   • Complete staged result: {final_accuracy:.1f}%")
    print(f"   • Improvement: {improvement:+.1f} percentage points")
    print(f"   • Relative improvement: {((final_accuracy / baseline) - 1) * 100:+.1f}%")
    
    return final_accuracy

if __name__ == "__main__":
    try:
        final_accuracy = main()
        
        if final_accuracy >= 60:
            print(f"\n🎯 MISSION ACCOMPLISHED: Staged training successful!")
            print(f"   Ready for Phase 5: Advanced techniques to reach 80%")
        else:
            print(f"\n🔄 CONTINUE OPTIMIZATION: Apply fallback improvements")
            
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
