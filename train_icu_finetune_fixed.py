#!/usr/bin/env python3
"""
Enhanced ICU fine-tuning with all fixes:
- ID normalization and fixed label mapping
- Small FC head (<100k params)
- GRID encoder pretraining
- Progressive unfreezing
- Curriculum learning
- Temporal subclips augmentation
"""
import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Add project paths
sys.path.append('.')
from models.heads.small_fc import SmallFCHead
from models.heads.cosine_fc import CosineFCHead, ArcFaceLoss
from utils.id_norm import validate_label_consistency
from advanced_training_components import (
    ComprehensiveVideoDataset,
    StandardizedPreprocessor,
    LabelSmoothingCrossEntropy,
    create_weighted_sampler
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedLightweightCNNLSTM(nn.Module):
    """Enhanced model with small FC head and GRID pretraining support."""
    
    def __init__(self, num_classes=4, dropout=0.6, head_type="small_fc"):
        super().__init__()
        
        # 3D CNN Encoder (same as before)
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv3d(1, 32, (3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),
            
            # Block 2
            nn.Conv3d(32, 64, (3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),
            
            # Block 3
            nn.Conv3d(64, 128, (3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),
            
            # Block 4 (last block for unfreezing)
            nn.Conv3d(128, 256, (3, 3, 3), padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((4, 2, 2))
        )
        
        # LSTM
        self.lstm = nn.LSTM(256 * 2 * 2, 256, batch_first=True, dropout=dropout)
        
        # Classification head
        if head_type == "small_fc":
            self.classifier = SmallFCHead(256, num_classes, dropout)
        elif head_type == "cosine_fc":
            self.classifier = CosineFCHead(256, num_classes)
        else:  # standard
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(256, num_classes)
            )
    
    def forward(self, x, labels=None, use_arcface=False):
        batch_size, channels, frames, height, width = x.shape

        # CNN encoding
        x = self.encoder(x)  # [B, 256, T', H', W']

        # Reshape for LSTM
        x = x.permute(0, 2, 1, 3, 4)  # [B, T', 256, H', W']
        x = x.contiguous().view(batch_size, x.size(1), -1)  # [B, T', 256*H'*W']

        # LSTM
        lstm_out, _ = self.lstm(x)  # [B, T', 256]

        # Use last timestep
        features = lstm_out[:, -1, :]  # [B, 256]

        # Classification (support ArcFace for cosine head)
        if hasattr(self.classifier, 'forward') and 'labels' in self.classifier.forward.__code__.co_varnames:
            return self.classifier(features, labels, use_arcface)
        else:
            return self.classifier(features)
    
    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen")
    
    def unfreeze_last_block(self):
        """Unfreeze only the last CNN block."""
        # Unfreeze block 4 (last block)
        for param in self.encoder[9:].parameters():  # Block 4 starts at index 9
            param.requires_grad = True
        logger.info("Last CNN block unfrozen")
    
    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("All parameters unfrozen")

def load_global_label_map():
    """Load the global label mapping."""
    with open('checkpoints/label2idx.json', 'r') as f:
        return json.load(f)

def load_grid_encoder(model, checkpoint_path):
    """Load GRID pretrained encoder weights."""
    if not os.path.exists(checkpoint_path):
        logger.warning(f"GRID checkpoint not found: {checkpoint_path}")
        return model
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract encoder weights
        encoder_state = {}
        for key, value in checkpoint.items():
            if key.startswith('encoder.'):
                encoder_state[key] = value
        
        # Load encoder weights
        model.load_state_dict(encoder_state, strict=False)
        logger.info(f"Loaded GRID encoder from: {checkpoint_path}")
        
    except Exception as e:
        logger.warning(f"Failed to load GRID encoder: {e}")
    
    return model

def create_model(num_classes, dropout=0.6, head_type="small_fc", grid_checkpoint=None):
    """Create model with optional GRID pretraining."""
    model = EnhancedLightweightCNNLSTM(num_classes, dropout, head_type)
    
    if grid_checkpoint:
        model = load_grid_encoder(model, grid_checkpoint)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model created: {total_params:,} total params, {trainable_params:,} trainable")
    
    return model

def train_epoch(model, dataloader, criterion, optimizer, device, curriculum_epoch=None, use_arcface=False):
    """Train for one epoch with optional curriculum learning."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, batch_data in enumerate(dataloader):
        if len(batch_data) == 3:
            videos, labels, metadata = batch_data
        else:
            videos, labels = batch_data
            metadata = {}

        videos = videos.to(device)
        labels = labels.to(device)

        # Curriculum learning: skip low-quality clips in early epochs
        if curriculum_epoch is not None and curriculum_epoch < 3:
            # Filter based on ROI quality (if available in metadata)
            roi_quality = metadata.get('roi_quality', [1.0] * len(videos))
            good_indices = [i for i, q in enumerate(roi_quality) if q > 0.5]

            if len(good_indices) < len(videos) * 0.5:  # Keep at least 50%
                good_indices = list(range(len(videos)))

            if good_indices:
                videos = videos[good_indices]
                labels = labels[good_indices]
        
        optimizer.zero_grad()
        outputs = model(videos, labels, use_arcface)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # Collect predictions
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data in dataloader:
            if len(batch_data) == 3:
                videos, labels, _ = batch_data
            else:
                videos, labels = batch_data
            videos = videos.to(device)
            labels = labels.to(device)
            
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, accuracy, f1

def main():
    parser = argparse.ArgumentParser(description="Enhanced ICU Fine-tuning")
    parser.add_argument("--splits-dir", required=True, help="Directory containing LOSO splits")
    parser.add_argument("--data-root", default="data/stabilized_subclips", help="Data root directory")
    parser.add_argument("--init-from", help="GRID encoder checkpoint path")
    parser.add_argument("--head", choices=["small_fc", "cosine_fc", "standard"], default="small_fc", help="Head type")
    parser.add_argument("--loss", choices=["cross_entropy", "arcface"], default="cross_entropy", help="Loss function type")
    parser.add_argument("--freeze-encoder", type=int, default=5, help="Epochs to freeze encoder")
    parser.add_argument("--unfreeze-last-block", action="store_true", help="Unfreeze last block only")
    parser.add_argument("--curriculum", action="store_true", help="Use curriculum learning")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.02, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.6, help="Dropout rate")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--restore-best-weights", action="store_true", help="Restore best weights after training")
    parser.add_argument("--output-dir", default="checkpoints/icu_finetune_fixed", help="Output directory")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load global label map
    global_label_map = load_global_label_map()
    num_classes = len(global_label_map)
    logger.info(f"Global label map: {global_label_map}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load splits info
    splits_file = os.path.join(args.splits_dir, "loso_splits_info.json")
    with open(splits_file, 'r') as f:
        splits_info = json.load(f)
    
    logger.info(f"Starting LOSO training with {len(splits_info)} folds")
    
    # Process each fold
    all_results = []
    
    for fold_idx, (held_out_speaker, split_info) in enumerate(splits_info.items(), 1):
        logger.info(f"\n🎯 FOLD {fold_idx}/{len(splits_info)}: Held-out speaker '{held_out_speaker}'")
        
        # Create model
        model = create_model(
            num_classes=num_classes,
            dropout=args.dropout,
            head_type=args.head,
            grid_checkpoint=args.init_from
        ).to(device)
        
        # Progressive unfreezing setup
        if args.freeze_encoder > 0:
            model.freeze_encoder()
        
        # Create datasets
        preprocessor = StandardizedPreprocessor(target_size=(64, 96), target_frames=32)

        train_dataset = ComprehensiveVideoDataset(
            split_info['train_csv'], preprocessor, args.data_root
        )
        val_dataset = ComprehensiveVideoDataset(
            split_info['val_csv'], preprocessor, args.data_root
        )
        
        # Validate label consistency
        validate_label_consistency(
            train_dataset.labels, val_dataset.labels, list(global_label_map.keys())
        )
        
        # Create data loaders
        train_sampler = create_weighted_sampler(train_dataset.labels)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )
        
        # Training setup
        if args.loss == "arcface":
            criterion = ArcFaceLoss(margin=0.5, label_smoothing=args.label_smoothing)
        else:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop with early stopping
        best_val_acc = 0
        best_epoch = 0
        patience_counter = 0
        best_model_path = os.path.join(args.output_dir, f"fold_{fold_idx}_best.pth")

        for epoch in range(args.epochs):
            # Progressive unfreezing
            if epoch == args.freeze_encoder and args.unfreeze_last_block:
                model.unfreeze_last_block()
            elif epoch == args.freeze_encoder + 5:
                model.unfreeze_all()
            
            # Training
            curriculum_epoch = epoch if args.curriculum else None
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, curriculum_epoch,
                use_arcface=(args.loss == "arcface")
            )
            
            # Validation
            val_loss, val_acc, val_f1 = validate_epoch(model, val_loader, criterion, device)
            
            # Scheduler step
            scheduler.step(val_loss)
            
            # Save best model and early stopping logic
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
            
            logger.info(
                f"Fold {fold_idx} Epoch {epoch+1:2d}/{args.epochs} | "
                f"Train: {train_loss:.4f}/{train_acc:.1%} | "
                f"Val: {val_loss:.4f}/{val_acc:.1%}/{val_f1:.4f} | "
                f"Best: {best_val_acc:.1%} (epoch {best_epoch+1}) | "
                f"Patience: {patience_counter}/{args.early_stopping_patience}"
            )

            # Early stopping check
            if patience_counter >= args.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Restore best weights if requested
        if args.restore_best_weights and os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            logger.info(f"Restored best weights from epoch {best_epoch+1}")

        # Store results
        fold_result = {
            'fold': fold_idx,
            'held_out_speaker': held_out_speaker,
            'best_val_accuracy': best_val_acc,
            'best_val_f1': val_f1,
            'train_videos': split_info['train_count'],
            'val_videos': split_info['val_count']
        }
        all_results.append(fold_result)
        
        logger.info(f"✅ Fold {fold_idx} complete: {best_val_acc:.1%} accuracy")
    
    # Final summary
    accuracies = [r['best_val_accuracy'] for r in all_results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    logger.info(f"\n🎯 LOSO RESULTS SUMMARY:")
    logger.info(f"Mean Accuracy: {mean_acc:.1%} ± {std_acc:.1%}")
    logger.info(f"Individual Folds: {[f'{acc:.1%}' for acc in accuracies]}")
    
    # Save results
    results_file = os.path.join(args.output_dir, "loso_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'individual_results': all_results
        }, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")
    
    if mean_acc >= 0.82:
        logger.info("🎉 TARGET ACHIEVED: >82% LOSO accuracy!")
    elif mean_acc >= 0.70:
        logger.info("✅ Good progress: >70% LOSO accuracy")
    else:
        logger.info("📈 Need more improvements to reach 82% target")

if __name__ == "__main__":
    main()
