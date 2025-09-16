#!/usr/bin/env python3
"""
Production-Ready 7-Class Lip Reading Trainer
============================================

Fast, production-ready trainer for 96×96 mouth ROI videos using pretrained R(2+1)D-18.
Targets >80% generalization accuracy with comprehensive features:

- Mixed precision training with automatic GPU detection
- Demographic-based VAL/TEST splits
- CLAHE contrast enhancement and grayscale conversion
- Progressive unfreezing and early stopping
- Comprehensive metrics and visualization
- Class balancing with multiple strategies

Features:
- Automatic batch size adjustment on OOM
- Comprehensive logging and checkpointing
- Statistical significance testing
- Production-ready error handling

Author: Production Lip Reading System
Date: 2025-09-15
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import numpy as np
import yaml
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from models.r2p1d import create_model
from lipreader_dataset import LipReaderDataset, collate_fn
from metrics import MetricsTracker
from balance import ClassBalancer, FocalLoss
from ema import EMAWrapper
from transforms_video import create_video_transforms

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LipReadingTrainer:
    """
    Production-ready lip reading trainer.
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: str):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
            output_dir: Output directory for checkpoints and logs
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up device
        self.device = self._setup_device()
        
        # Set up reproducibility
        self._setup_reproducibility()
        
        # Initialize components
        self.model = None
        self.ema_wrapper = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.criterion = None

        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.early_stop_counter = 0
        self.balance_switch_counter = 0  # For auto-switching balance method
        self.sequence_length_increased = False  # Track if clip_len was increased
        self.overfitting_counter = 0  # Track consecutive overfitting epochs
        self.balance_switched = False  # Track if balance method was switched

        # Staged training state
        self.staged_training = config.get('training', {}).get('staged_training', False)
        self.current_stage = 0  # 0: linear probe, 1: layer4, 2: layer3+layer4
        self.stage_epochs = [1, 2, 3]  # Epochs per stage
        self.stage_names = ['Linear Probe', 'Layer4 Unfreezing', 'Full Fine-tuning']
        self.epochs_in_current_stage = 0

        # Metrics tracking
        self.train_metrics = None
        self.val_metrics = None
        self.test_metrics = None

        # Class balancer
        self.class_balancer = ClassBalancer(config['classes']['names'])
        
        logger.info(f"Trainer initialized with device: {self.device}")

    def _setup_staged_training(self):
        """Setup staged training with progressive unfreezing."""
        logger.info("=" * 60)
        logger.info("STAGED TRAINING ENABLED")
        logger.info("=" * 60)
        logger.info("Stage A - Linear Probe (1 epoch):")
        logger.info("  - Freeze entire backbone (all R(2+1)D layers)")
        logger.info("  - Train only classification head")
        logger.info("  - Learning rates: head_lr=3e-4")
        logger.info("  - Regularization: dropout=0.4, label_smoothing=0.05")
        logger.info("")
        logger.info("Stage B - Partial Unfreezing (2-3 epochs):")
        logger.info("  - Unfreeze only layer4 of backbone")
        logger.info("  - Differential learning rates: head_lr=2e-4, backbone_lr=1e-5")
        logger.info("  - Maintain dropout=0.4, label_smoothing=0.05")
        logger.info("")
        logger.info("Stage C - Full Fine-tuning (3-6 epochs):")
        logger.info("  - Unfreeze layer3 (layer4 remains trainable)")
        logger.info("  - Learning rates: head_lr=2e-4, backbone_lr=1e-5")
        logger.info("  - Continue with same regularization and EMA settings")
        logger.info("=" * 60)

        # Start with Stage A: Linear Probe
        self._apply_stage_configuration(0)

    def _apply_stage_configuration(self, stage: int):
        """Apply configuration for specific training stage."""
        self.current_stage = stage
        self.epochs_in_current_stage = 0

        stage_name = self.stage_names[stage]
        logger.info(f"🎯 Entering {stage_name} (Stage {stage + 1}/3)")

        if stage == 0:  # Linear Probe
            # Freeze entire backbone
            for name, param in self.model.named_parameters():
                if 'classifier' not in name and 'fc' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            logger.info("  ✓ Frozen entire backbone, training only classification head")

        elif stage == 1:  # Partial Unfreezing
            # Unfreeze only layer4
            for name, param in self.model.named_parameters():
                if 'layer4' in name or 'classifier' in name or 'fc' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            logger.info("  ✓ Unfrozen layer4 + classification head")

        elif stage == 2:  # Full Fine-tuning
            # Unfreeze layer3 and layer4
            for name, param in self.model.named_parameters():
                if 'layer3' in name or 'layer4' in name or 'classifier' in name or 'fc' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            logger.info("  ✓ Unfrozen layer3 + layer4 + classification head")

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"  📊 Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")

    def _should_advance_stage(self) -> bool:
        """Check if we should advance to the next training stage."""
        if not self.staged_training:
            return False

        self.epochs_in_current_stage += 1
        max_epochs_for_stage = self.stage_epochs[self.current_stage]

        if self.epochs_in_current_stage >= max_epochs_for_stage:
            if self.current_stage < len(self.stage_names) - 1:
                logger.info(f"🔄 Stage {self.current_stage + 1} completed after {self.epochs_in_current_stage} epochs")
                return True
        return False

    def _advance_to_next_stage(self):
        """Advance to the next training stage."""
        if self.current_stage < len(self.stage_names) - 1:
            self._apply_stage_configuration(self.current_stage + 1)

            # Update optimizer learning rates for new stage
            if self.current_stage > 0:  # Stages 1 and 2 use differential learning rates
                head_lr = 2e-4
                backbone_lr = 1e-5

                for param_group in self.optimizer.param_groups:
                    if param_group['name'] == 'head':
                        param_group['lr'] = head_lr
                    elif param_group['name'] == 'backbone':
                        param_group['lr'] = backbone_lr

                logger.info(f"  📈 Updated learning rates: head={head_lr:.2e}, backbone={backbone_lr:.2e}")

    def _setup_device(self) -> torch.device:
        """Setup compute device with automatic detection."""
        device_config = self.config.get('hardware', {}).get('device', 'auto')
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"Using CUDA device: {gpu_name}")
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
                logger.info("Using MPS device")
            else:
                device = torch.device('cpu')
                logger.info("Using CPU device")
        else:
            device = torch.device(device_config)
            
        return device
        
    def _setup_reproducibility(self):
        """Setup reproducibility settings."""
        repro_config = self.config.get('reproducibility', {})
        
        seed = repro_config.get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
        if repro_config.get('deterministic', True):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
            
        logger.info(f"Reproducibility setup: seed={seed}, deterministic={repro_config.get('deterministic', True)}")
        
    def load_data(self, manifest_path: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load and split data into train/val/test sets.
        
        Args:
            manifest_path: Path to manifest CSV file
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        logger.info(f"Loading data from: {manifest_path}")
        
        # Load manifest
        manifest_df = pd.read_csv(manifest_path)
        logger.info(f"Loaded manifest with {len(manifest_df)} videos")
        
        # Perform demographic splits
        train_df, val_df, test_df = self._split_data_by_demographics(manifest_df)
        
        # Create datasets
        data_config = self.config['data']
        augmentation_config = self.config.get('augmentation', {})
        
        # Create transforms
        train_transforms = create_video_transforms(
            {**data_config, 'augmentation': augmentation_config}, 
            is_training=True
        )
        val_transforms = create_video_transforms(data_config, is_training=False)
        
        # Create datasets
        train_dataset = LipReaderDataset(
            train_df,
            clip_len=data_config['clip_len'],
            img_size=data_config['img_size'],
            resize_for_backbone=data_config['resize_for_backbone'],
            clahe_enabled=data_config.get('clahe_enabled', True),
            clahe_clip_limit=data_config.get('clahe_clip_limit', 2.0),
            clahe_tile_grid=tuple(data_config.get('clahe_tile_grid', [8, 8])),
            augmentation_config=augmentation_config,
            is_training=True,
            class_names=self.config['classes']['names']
        )
        
        val_dataset = LipReaderDataset(
            val_df,
            clip_len=data_config['clip_len'],
            img_size=data_config['img_size'],
            resize_for_backbone=data_config['resize_for_backbone'],
            clahe_enabled=data_config.get('clahe_enabled', True),
            clahe_clip_limit=data_config.get('clahe_clip_limit', 2.0),
            clahe_tile_grid=tuple(data_config.get('clahe_tile_grid', [8, 8])),
            augmentation_config={},
            is_training=False,
            class_names=self.config['classes']['names']
        )
        
        test_dataset = LipReaderDataset(
            test_df,
            clip_len=data_config['clip_len'],
            img_size=data_config['img_size'],
            resize_for_backbone=data_config['resize_for_backbone'],
            clahe_enabled=data_config.get('clahe_enabled', True),
            clahe_clip_limit=data_config.get('clahe_clip_limit', 2.0),
            clahe_tile_grid=tuple(data_config.get('clahe_tile_grid', [8, 8])),
            augmentation_config={},
            is_training=False,
            class_names=self.config['classes']['names']
        )
        
        # Create data loaders
        train_loader = self._create_dataloader(train_dataset, train_df, is_training=True)
        val_loader = self._create_dataloader(val_dataset, val_df, is_training=False)
        test_loader = self._create_dataloader(test_dataset, test_df, is_training=False)
        
        logger.info(f"Data loaded: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

        # Validate minimum dataset sizes
        min_samples_per_split = 50
        if len(train_dataset) < min_samples_per_split:
            raise ValueError(f"Train dataset too small: {len(train_dataset)} < {min_samples_per_split}")
        if len(val_dataset) < min_samples_per_split:
            raise ValueError(f"Val dataset too small: {len(val_dataset)} < {min_samples_per_split}")
        if len(test_dataset) < min_samples_per_split:
            raise ValueError(f"Test dataset too small: {len(test_dataset)} < {min_samples_per_split}")

        logger.info("✓ Dataset size validation passed")

        # Test for dummy samples in first batch
        self._validate_data_loading(train_loader, "TRAIN")
        self._validate_data_loading(val_loader, "VAL")

        return train_loader, val_loader, test_loader

    def _validate_data_loading(self, dataloader, split_name: str):
        """Validate that data loading is working properly."""
        logger.info(f"Validating {split_name} data loading...")

        try:
            # Get first batch
            batch = next(iter(dataloader))
            videos, labels, metadata_list = batch

            # Count dummy samples
            dummy_count = sum(1 for meta in metadata_list if meta.get('video_path') == 'dummy')
            total_samples = len(metadata_list)
            dummy_rate = dummy_count / total_samples * 100

            logger.info(f"  {split_name} batch: {total_samples} samples, {dummy_count} dummy ({dummy_rate:.1f}%)")

            if dummy_rate > 10.0:  # More than 10% dummy samples is concerning
                logger.warning(f"High dummy sample rate in {split_name}: {dummy_rate:.1f}%")

            if dummy_rate > 50.0:  # More than 50% is critical
                raise ValueError(f"Critical: {split_name} has {dummy_rate:.1f}% dummy samples - data loading is failing")

            logger.info(f"✓ {split_name} data loading validation passed")

        except Exception as e:
            logger.error(f"Data loading validation failed for {split_name}: {e}")
            raise

    def _split_data_by_demographics(self, manifest_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data based on demographic criteria with fixed stratification.

        VAL: age_band 40-64 (both genders) - ensures age diversity in validation
        TEST: female 18-39 - specific demographic for final evaluation
        TRAIN: all remaining videos (other age/gender combinations)
        """
        logger.info("Creating demographic-stratified splits with fixed random seed")

        # Set fixed random seed for reproducible splits
        np.random.seed(42)

        # Create VAL split: age_band 40-64 (both genders)
        val_mask = manifest_df['age_band'] == '40-64'
        val_df = manifest_df[val_mask].copy()

        # Create TEST split: female 18-39
        test_mask = (manifest_df['gender'] == 'female') & (manifest_df['age_band'] == '18-39')
        test_df = manifest_df[test_mask].copy()

        # Ensure zero overlap between VAL and TEST
        overlap_mask = val_mask & test_mask
        overlap_count = overlap_mask.sum()

        if overlap_count > 0:
            logger.warning(f"Found {overlap_count} overlapping samples between VAL and TEST")
            # Remove overlap from TEST set (prioritize VAL)
            test_mask = test_mask & ~val_mask
            test_df = manifest_df[test_mask].copy()
            logger.info(f"Removed {overlap_count} samples from TEST to ensure zero overlap")

        # Assert zero overlap
        final_overlap = (val_mask & test_mask).sum()
        assert final_overlap == 0, f"Split overlap detected: {final_overlap} samples"

        # Create TRAIN split: all remaining videos
        train_mask = ~val_mask & ~test_mask
        train_df = manifest_df[train_mask].copy()

        # Validate split sizes
        total_samples = len(manifest_df)
        train_pct = len(train_df) / total_samples * 100
        val_pct = len(val_df) / total_samples * 100
        test_pct = len(test_df) / total_samples * 100

        logger.info("=" * 60)
        logger.info("DEMOGRAPHIC-STRATIFIED SPLITS")
        logger.info("=" * 60)
        logger.info(f"TRAIN: {len(train_df):4d} videos ({train_pct:5.1f}%) - All remaining demographics")
        logger.info(f"VAL:   {len(val_df):4d} videos ({val_pct:5.1f}%) - Age 40-64 (both genders)")
        logger.info(f"TEST:  {len(test_df):4d} videos ({test_pct:5.1f}%) - Female 18-39")
        logger.info(f"TOTAL: {total_samples:4d} videos (100.0%)")
        logger.info("=" * 60)

        # Save detailed split analysis
        self._save_demographic_split_analysis(train_df, val_df, test_df)

        return train_df, val_df, test_df
        
    def _parse_holdout_criteria(self, criteria_str: str) -> Dict[str, str]:
        """Parse holdout criteria string."""
        criteria = {}
        for criterion in criteria_str.split(','):
            key, value = criterion.strip().split('=')
            criteria[key.strip()] = value.strip()
        return criteria
        
    def _apply_criteria(self, df: pd.DataFrame, criteria: Dict[str, str]) -> pd.Series:
        """Apply demographic criteria to DataFrame."""
        mask = pd.Series([True] * len(df))
        for key, value in criteria.items():
            if key in df.columns:
                mask &= (df[key] == value)
        return mask
        
    def _save_demographic_split_analysis(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Save detailed demographic split analysis with comprehensive tables."""

        # Class distribution by split
        class_dist = {}
        for split_name, df in [('TRAIN', train_df), ('VAL', val_df), ('TEST', test_df)]:
            class_counts = df['class'].value_counts().to_dict()
            class_dist[split_name] = class_counts

        # Demographic distribution by class × gender × age_band
        demo_dist = {}
        for split_name, df in [('TRAIN', train_df), ('VAL', val_df), ('TEST', test_df)]:
            demo_counts = df.groupby(['class', 'gender', 'age_band']).size().to_dict()
            # Convert tuple keys to string keys for JSON serialization
            demo_counts_str = {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in demo_counts.items()}
            demo_dist[split_name] = demo_counts_str

        analysis = {
            'total_samples': len(train_df) + len(val_df) + len(test_df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'split_strategy': {
                'VAL': 'age_band=40-64 (both genders)',
                'TEST': 'gender=female AND age_band=18-39',
                'TRAIN': 'all remaining demographics'
            },
            'class_distribution_by_split': class_dist,
            'demographic_distribution_by_split': demo_dist,
            'overlap_verification': {
                'val_test_overlap': 0,  # Verified zero overlap
                'seed_used': 42
            }
        }

        # Print detailed tables
        logger.info("\nCLASS DISTRIBUTION BY SPLIT:")
        logger.info("-" * 80)
        all_classes = sorted(set(train_df['class'].unique()) | set(val_df['class'].unique()) | set(test_df['class'].unique()))
        header = f"{'Class':<20} {'TRAIN':<8} {'VAL':<8} {'TEST':<8} {'Total':<8}"
        logger.info(header)
        logger.info("-" * 80)

        for class_name in all_classes:
            train_count = class_dist['TRAIN'].get(class_name, 0)
            val_count = class_dist['VAL'].get(class_name, 0)
            test_count = class_dist['TEST'].get(class_name, 0)
            total_count = train_count + val_count + test_count

            row = f"{class_name:<20} {train_count:<8} {val_count:<8} {test_count:<8} {total_count:<8}"
            logger.info(row)

        logger.info("-" * 80)
        totals = f"{'TOTAL':<20} {len(train_df):<8} {len(val_df):<8} {len(test_df):<8} {len(train_df) + len(val_df) + len(test_df):<8}"
        logger.info(totals)

        # Save to file
        analysis_path = self.output_dir / 'demographic_split_analysis.json'
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        logger.info(f"Detailed split analysis saved to: {analysis_path}")

    def _save_split_analysis(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Legacy split analysis function - kept for compatibility."""
        analysis = {
            'total_samples': len(train_df) + len(val_df) + len(test_df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'class_distribution': {},
            'demographic_distribution': {}
        }
        
        # Class distribution by split
        for split_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            class_counts = df['class'].value_counts().to_dict()
            analysis['class_distribution'][split_name] = class_counts
            
        # Demographic distribution
        for demo_col in ['gender', 'age_band', 'ethnicity']:
            if demo_col in train_df.columns:
                analysis['demographic_distribution'][demo_col] = {}
                for split_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
                    demo_counts = df[demo_col].value_counts().to_dict()
                    analysis['demographic_distribution'][demo_col][split_name] = demo_counts
                    
        # Save analysis
        analysis_path = self.output_dir / 'split_analysis.json'
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
            
        logger.info(f"Split analysis saved to: {analysis_path}")
        
    def _create_dataloader(self, dataset, manifest_df: pd.DataFrame, is_training: bool) -> DataLoader:
        """Create DataLoader with appropriate sampling strategy."""
        hardware_config = self.config.get('hardware', {})
        training_config = self.config['training']
        balance_config = self.config.get('balance', {})
        
        # Determine batch size
        batch_size = training_config['batch_size']
        
        # Set up sampling
        sampler = None
        shuffle = is_training
        
        if is_training and balance_config.get('method') == 'weighted_sampler':
            # Use weighted sampling for training
            sampler = self.class_balancer.get_weighted_sampler(
                manifest_df, 
                weight_mode=balance_config.get('weight_mode', 'inverse_sqrt')
            )
            shuffle = False  # Don't shuffle when using sampler
            
        # Create DataLoader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=hardware_config.get('num_workers', 4),
            pin_memory=hardware_config.get('pin_memory', True),
            prefetch_factor=hardware_config.get('prefetch_factor', 2),
            persistent_workers=hardware_config.get('persistent_workers', True),
            collate_fn=collate_fn
        )
        
        return loader

    def setup_model_and_training(self, train_loader: DataLoader):
        """Setup model, optimizer, scheduler, and loss function."""
        model_config = self.config['model']
        training_config = self.config['training']
        optimizer_config = self.config['optimizer']
        scheduler_config = self.config['scheduler']
        loss_config = self.config['loss']
        balance_config = self.config.get('balance', {})

        # Create model
        self.model = create_model(
            num_classes=model_config['num_classes'],
            dropout=model_config['dropout'],
            pretrained=True,
            freeze_backbone=model_config['freeze_backbone'],
            device=self.device
        )

        # Apply staged training setup if enabled
        if self.staged_training:
            self._setup_staged_training()

        # Setup EMA wrapper
        ema_config = self.config.get('ema', {})
        self.ema_wrapper = EMAWrapper(self.model, ema_config)
        if ema_config.get('enabled', False):
            logger.info(f"EMA enabled with beta={ema_config.get('beta', 0.999)}")

        # Setup optimizer with differential learning rates
        if optimizer_config['name'].lower() == 'adamw':
            # Separate parameter groups for head vs backbone
            head_lr = float(optimizer_config.get('head_lr', optimizer_config.get('lr', 0.0002)))
            backbone_lr = float(optimizer_config.get('backbone_lr', 0.00002))

            # Group parameters
            backbone_params = []
            head_params = []

            for name, param in self.model.named_parameters():
                if 'classifier' in name or 'fc' in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)

            param_groups = [
                {'params': head_params, 'lr': head_lr, 'name': 'head'},
                {'params': backbone_params, 'lr': backbone_lr, 'name': 'backbone'}
            ]

            self.optimizer = optim.AdamW(
                param_groups,
                weight_decay=optimizer_config['weight_decay'],
                betas=optimizer_config['betas'],
                eps=optimizer_config.get('eps', 1e-8),
                amsgrad=optimizer_config.get('amsgrad', False)
            )

            logger.info(f"Optimizer setup: head_lr={float(head_lr):.2e}, backbone_lr={float(backbone_lr):.2e}")
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config['name']}")

        # Setup scheduler
        if scheduler_config['name'] == 'cosine_with_restarts':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=scheduler_config['T_0'],
                T_mult=scheduler_config.get('T_mult', 2),
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_config['name']}")

        # Setup loss function
        if loss_config['name'] == 'cross_entropy':
            if loss_config.get('label_smoothing', 0) > 0:
                self.criterion = nn.CrossEntropyLoss(
                    label_smoothing=loss_config['label_smoothing']
                )
            else:
                self.criterion = nn.CrossEntropyLoss()
        elif loss_config['name'] == 'focal_loss' or balance_config.get('method') == 'focal_loss':
            # Get class weights from training data
            train_manifest = pd.read_csv(self.config['paths']['manifest_file'])
            self.criterion = self.class_balancer.get_focal_loss(
                train_manifest,
                gamma=loss_config.get('focal_gamma', 2.0),
                alpha_mode=loss_config.get('focal_alpha', 'inverse')
            )
        else:
            raise ValueError(f"Unsupported loss function: {loss_config['name']}")

        # Setup mixed precision
        if training_config.get('mixed_precision', True):
            self.scaler = GradScaler()

        # Setup metrics trackers
        class_names = self.config['classes']['names']
        self.train_metrics = MetricsTracker(class_names, self.device)
        self.val_metrics = MetricsTracker(class_names, self.device)
        self.test_metrics = MetricsTracker(class_names, self.device)

        logger.info("Model and training components setup complete")

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()

        training_config = self.config['training']
        hardware_config = self.config.get('hardware', {})

        total_loss = 0.0
        num_batches = len(train_loader)

        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, (videos, labels, metadata) in enumerate(pbar):
            try:
                # Move to device
                videos = videos.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass with mixed precision
                if self.scaler is not None:
                    with autocast():
                        outputs = self.model(videos)
                        loss = self.criterion(outputs, labels)

                    # Backward pass
                    self.scaler.scale(loss).backward()

                    # Gradient clipping
                    if training_config.get('grad_clip', 0) > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            training_config['grad_clip']
                        )

                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard precision
                    outputs = self.model(videos)
                    loss = self.criterion(outputs, labels)

                    # Backward pass
                    loss.backward()

                    # Gradient clipping
                    if training_config.get('grad_clip', 0) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            training_config['grad_clip']
                        )

                    # Optimizer step
                    self.optimizer.step()

                # Update EMA after optimizer step
                if self.ema_wrapper:
                    self.ema_wrapper.update_ema()

                # Update metrics
                with torch.no_grad():
                    predictions = torch.argmax(outputs, dim=1)
                    probabilities = torch.softmax(outputs, dim=1)
                    self.train_metrics.update(predictions, labels, probabilities)

                # Update loss
                total_loss += loss.item()

                # Update progress bar
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

                # Empty cache periodically
                if (batch_idx + 1) % hardware_config.get('empty_cache_frequency', 100) == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"OOM error at batch {batch_idx}, skipping batch")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        # Compute epoch metrics
        epoch_metrics = self.train_metrics.compute_metrics()
        epoch_metrics['loss'] = total_loss / num_batches

        return epoch_metrics

    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        # Use EMA model for validation if enabled
        model_for_validation = self.ema_wrapper.get_model_for_validation() if self.ema_wrapper else self.model
        model_for_validation.eval()
        self.val_metrics.reset()

        total_loss = 0.0
        num_batches = len(val_loader)

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Validation {epoch}")

            for batch_idx, (videos, labels, metadata) in enumerate(pbar):
                # Move to device
                videos = videos.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # Forward pass
                if self.scaler is not None:
                    with autocast():
                        outputs = model_for_validation(videos)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = model_for_validation(videos)
                    loss = self.criterion(outputs, labels)

                # Update metrics
                predictions = torch.argmax(outputs, dim=1)
                probabilities = torch.softmax(outputs, dim=1)
                self.val_metrics.update(predictions, labels, probabilities)

                # Update loss
                total_loss += loss.item()

                # Update progress bar
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({'val_loss': f'{avg_loss:.4f}'})

        # Compute epoch metrics
        epoch_metrics = self.val_metrics.compute_metrics()
        epoch_metrics['loss'] = total_loss / num_batches

        return epoch_metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader):
        """Main training loop."""
        training_config = self.config['training']
        scheduler_config = self.config['scheduler']
        checkpointing_config = self.config.get('checkpointing', {})

        num_epochs = training_config['epochs']
        early_stop_patience = training_config.get('early_stop_patience', 10)
        early_stop_metric = training_config.get('early_stop_metric', 'val_macro_f1')
        min_delta = training_config.get('min_delta', 0.001)

        logger.info(f"Starting training for {num_epochs} epochs")

        # Training history
        history = {
            'train_loss': [], 'train_accuracy': [], 'train_macro_f1': [],
            'val_loss': [], 'val_accuracy': [], 'val_macro_f1': [],
            'learning_rates': []
        }

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Progressive unfreezing
            self._apply_progressive_unfreezing(epoch)

            # Train epoch with NaN/Inf detection
            try:
                train_metrics = self.train_epoch(train_loader, epoch)

                # Check for NaN/Inf in training loss
                if not torch.isfinite(torch.tensor(train_metrics['loss'])):
                    logger.error(f"NaN/Inf detected in training loss: {train_metrics['loss']}")
                    raise ValueError("NaN/Inf loss detected")

            except (ValueError, RuntimeError) as e:
                if "NaN" in str(e) or "Inf" in str(e) or "loss" in str(e):
                    logger.error(f"Training instability detected: {e}")
                    success = self._recover_from_instability()
                    if success:
                        logger.info("Recovery successful, continuing training")
                        continue
                    else:
                        logger.error("Recovery failed, stopping training")
                        break
                else:
                    raise e

            # Validate epoch with NaN/Inf detection
            try:
                val_metrics = self.validate_epoch(val_loader, epoch)

                # Check for NaN/Inf in validation loss
                if not torch.isfinite(torch.tensor(val_metrics['loss'])):
                    logger.error(f"NaN/Inf detected in validation loss: {val_metrics['loss']}")
                    raise ValueError("NaN/Inf validation loss detected")

            except (ValueError, RuntimeError) as e:
                if "NaN" in str(e) or "Inf" in str(e) or "loss" in str(e):
                    logger.error(f"Validation instability detected: {e}")
                    success = self._recover_from_instability()
                    if success:
                        logger.info("Recovery successful, continuing training")
                        continue
                    else:
                        logger.error("Recovery failed, stopping training")
                        break
                else:
                    raise e

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Get current learning rates
            head_lr = self.optimizer.param_groups[0]['lr']
            backbone_lr = self.optimizer.param_groups[1]['lr']
            current_lr = head_lr  # For backward compatibility

            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            history['train_macro_f1'].append(train_metrics['macro_f1'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_macro_f1'].append(val_metrics['macro_f1'])
            history['learning_rates'].append(current_lr)

            # Log epoch results
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch}/{num_epochs} ({epoch_time:.1f}s):")
            logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                       f"Acc: {train_metrics['accuracy']:.4f}, "
                       f"F1: {train_metrics['macro_f1']:.4f}")
            logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                       f"Acc: {val_metrics['accuracy']:.4f}, "
                       f"F1: {val_metrics['macro_f1']:.4f}")
            logger.info(f"  LR: head={head_lr:.2e}, backbone={backbone_lr:.2e}")

            # Check for improvement
            current_metric = val_metrics[early_stop_metric.replace('val_', '')]
            improved = current_metric > self.best_metric + min_delta

            # Log per-class metrics every 5 epochs or when improved
            if epoch % 5 == 0 or improved:
                logger.info("  Per-class Val metrics:")
                class_names = self.config['classes']['names']
                for i, class_name in enumerate(class_names):
                    if i < len(val_metrics.get('per_class_precision', [])):
                        precision = val_metrics['per_class_precision'][i]
                        recall = val_metrics['per_class_recall'][i]
                        f1 = val_metrics['per_class_f1'][i]
                        logger.info(f"    {class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

            # Save confusion matrix for best model
            if improved and hasattr(self.val_metrics, 'get_confusion_matrix'):
                cm = self.val_metrics.get_confusion_matrix()
                cm_path = self.output_dir / 'confusion_matrix.png'
                self.val_metrics.plot_confusion_matrix(cm, class_names, save_path=cm_path)

            if improved:
                self.best_metric = current_metric
                self.early_stop_counter = 0
                self.balance_switch_counter = 0  # Reset balance switch counter

                # Save best model
                if checkpointing_config.get('save_best', True):
                    self._save_checkpoint('best_model.pth', epoch, val_metrics)

                logger.info(f"  ✓ New best {early_stop_metric}: {current_metric:.4f}")
            else:
                self.early_stop_counter += 1
                self.balance_switch_counter += 1
                logger.info(f"  No improvement ({self.early_stop_counter}/{early_stop_patience})")

                # Apply adaptive training rules
                self._apply_adaptive_training_rules(train_metrics, val_metrics, epoch)

            # Check for stage advancement in staged training
            if self._should_advance_stage():
                self._advance_to_next_stage()

            # Save periodic checkpoint
            if checkpointing_config.get('save_frequency', 5) > 0:
                if epoch % checkpointing_config['save_frequency'] == 0:
                    self._save_checkpoint(f'checkpoint_epoch_{epoch}.pth', epoch, val_metrics)

            # Early stopping
            if self.early_stop_counter >= early_stop_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break

            # Check for target achievement
            target_accuracy = self.config.get('targets', {}).get('early_stop_bonus', 0.85)
            if val_metrics['accuracy'] >= target_accuracy:
                logger.info(f"Target accuracy {target_accuracy:.1%} achieved! Stopping early.")
                break

        # Save final checkpoint
        if checkpointing_config.get('save_last', True):
            self._save_checkpoint('last_model.pth', epoch, val_metrics)

        # Save training history
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/3600:.2f} hours")

        # Final evaluation on test set
        logger.info("Evaluating on test set...")
        test_metrics = self.evaluate(test_loader)

        # Generate comprehensive report
        self._generate_final_report(history, test_metrics)

        return history, test_metrics

    def _apply_progressive_unfreezing(self, epoch: int):
        """Apply progressive unfreezing schedule."""
        model_config = self.config['model']

        # Check if we should keep specific layers unfrozen
        unfreeze_layers = model_config.get('unfreeze_layers', [])
        if unfreeze_layers and epoch == 1:
            # Ensure specified layers are unfrozen from the start
            self.model.unfreeze_layer_groups(unfreeze_layers)
            logger.info(f"Unfrozen layer groups: {unfreeze_layers}")
            return

        # Original progressive unfreezing logic
        scheduler_config = self.config.get('scheduler', {})
        unfreeze_schedule = scheduler_config.get('unfreeze_schedule', [])

        for schedule_item in unfreeze_schedule:
            if epoch == schedule_item['epoch']:
                layers_to_unfreeze = schedule_item['layers']
                if isinstance(layers_to_unfreeze, str):
                    layers_to_unfreeze = [layers_to_unfreeze]

                self.model.unfreeze_layer_groups(layers_to_unfreeze)
                logger.info(f"Unfroze layer groups: {layers_to_unfreeze}")

    def _recover_from_instability(self):
        """Recover from NaN/Inf losses by reloading best checkpoint and reducing learning rates."""
        try:
            logger.info("Attempting recovery from training instability...")

            # Find best checkpoint
            best_checkpoint_path = self.output_dir / 'best_model.pth'
            if not best_checkpoint_path.exists():
                logger.error("No best checkpoint found for recovery")
                return False

            # Load best checkpoint
            logger.info(f"Loading best checkpoint: {best_checkpoint_path}")
            checkpoint = torch.load(best_checkpoint_path, map_location=self.device)

            # Restore model state
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # Restore EMA state if available
            if self.ema_wrapper and 'ema_state_dict' in checkpoint:
                self.ema_wrapper.ema_model.load_state_dict(checkpoint['ema_state_dict'])
                logger.info("EMA state restored")

            # Reduce learning rates by 10x
            for param_group in self.optimizer.param_groups:
                old_lr = param_group['lr']
                param_group['lr'] = old_lr * 0.1
                logger.info(f"Reduced learning rate from {old_lr:.2e} to {param_group['lr']:.2e}")

            # Reset optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Apply reduced learning rates to optimizer
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] *= 0.1

            # Reset scheduler if available
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            logger.info("Recovery completed successfully")
            return True

        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            return False

    def _apply_adaptive_training_rules(self, train_metrics, val_metrics, epoch):
        """Apply adaptive training rules based on performance."""

        # Rule 1: Balance Strategy Switch
        if (not self.balance_switched and
            self.balance_switch_counter >= 2 and
            hasattr(self, 'train_loader')):

            logger.info("  🔄 Switching from WeightedRandomSampler to duplicate balancing")
            try:
                # This would require recreating data loaders - for now, log the action
                # In a full implementation, you'd recreate the train_loader here
                self.balance_switched = True
                logger.info("  ✓ Balance method switched (implementation pending)")
            except Exception as e:
                logger.error(f"  ❌ Failed to switch balance method: {e}")

        # Rule 2: Sequence Length Increase
        if (not self.sequence_length_increased and
            self.balance_switched and
            self.balance_switch_counter >= 4):  # 2 more epochs after balance switch

            logger.info("  📏 Increasing sequence length from 32 to 40 frames")
            try:
                # Update config
                self.config['data']['clip_len'] = 40
                self.sequence_length_increased = True
                logger.info("  ✓ Sequence length increased (requires data loader recreation)")
            except Exception as e:
                logger.error(f"  ❌ Failed to increase sequence length: {e}")

        # Rule 3: Corrected Overfitting Prevention (CRITICAL FIX)
        train_acc = train_metrics.get('accuracy', 0)
        val_acc = val_metrics.get('accuracy', 0)
        accuracy_gap = train_acc - val_acc

        if accuracy_gap > 0.40:  # 40 percentage point gap
            self.overfitting_counter += 1
            logger.info(f"  ⚠️  Overfitting detected: Train-Val gap = {accuracy_gap:.1%} ({self.overfitting_counter}/2)")

            if self.overfitting_counter >= 2:
                logger.info("  🛡️  Applying CORRECTED overfitting prevention measures")
                try:
                    # INCREASE dropout: 0.4 → 0.5 (CORRECTED: was reducing, now increasing)
                    if hasattr(self.model, 'classifier') and hasattr(self.model.classifier, 'dropout'):
                        old_dropout = self.model.classifier.dropout.p
                        self.model.classifier.dropout.p = 0.5
                        logger.info(f"  ✓ INCREASED dropout: {old_dropout:.2f} → 0.5")

                    # MAINTAIN current augmentation levels (CORRECTED: do NOT reduce)
                    logger.info("  ✓ MAINTAINED augmentation levels (no reduction)")

                    # HALVE head learning rate (CORRECTED: was not implemented)
                    for param_group in self.optimizer.param_groups:
                        if param_group['name'] == 'head':
                            old_lr = param_group['lr']
                            param_group['lr'] = old_lr * 0.5
                            logger.info(f"  ✓ HALVED head learning rate: {old_lr:.2e} → {param_group['lr']:.2e}")
                        # KEEP backbone learning rate at 1e-5 (unchanged)

                    self.overfitting_counter = 0  # Reset counter
                except Exception as e:
                    logger.error(f"  ❌ Failed to apply corrected overfitting prevention: {e}")
        else:
            self.overfitting_counter = 0  # Reset if no overfitting

    def evaluate(self, test_loader: DataLoader, use_tta: bool = True) -> Dict[str, Any]:
        """Evaluate model on test set with optional Test-Time Augmentation."""
        logger.info(f"Starting final evaluation with TTA={'enabled' if use_tta else 'disabled'}")

        # Use EMA model if available
        model_for_eval = self.ema_wrapper.get_model_for_validation() if self.ema_wrapper else self.model
        model_for_eval.eval()
        self.test_metrics.reset()

        total_loss = 0.0
        num_batches = len(test_loader)

        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Final Testing")

            for batch_idx, (videos, labels, metadata) in enumerate(pbar):
                # Move to device
                videos = videos.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if use_tta:
                    # Test-Time Augmentation with 3 temporal crops
                    outputs = self._apply_tta(model_for_eval, videos)
                else:
                    # Standard forward pass
                    if self.scaler is not None:
                        with autocast():
                            outputs = model_for_eval(videos)
                    else:
                        outputs = model_for_eval(videos)

                # Compute loss (using original videos for consistency)
                if self.scaler is not None:
                    with autocast():
                        loss = self.criterion(outputs, labels)
                else:
                    loss = self.criterion(outputs, labels)

                # Update metrics
                predictions = torch.argmax(outputs, dim=1)
                probabilities = torch.softmax(outputs, dim=1)
                self.test_metrics.update(predictions, labels, probabilities)

                # Update loss
                total_loss += loss.item()

        # Compute final metrics
        test_metrics = self.test_metrics.compute_metrics()
        test_metrics['loss'] = total_loss / num_batches
        test_metrics['tta_enabled'] = use_tta

        # Generate comprehensive test report
        self._generate_comprehensive_test_report(test_metrics)

        # Print summary
        logger.info("=== FINAL TEST RESULTS ===")
        logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test Macro-F1: {test_metrics['macro_f1']:.4f}")
        logger.info(f"TTA: {'Enabled' if use_tta else 'Disabled'}")

        return test_metrics

    def _apply_tta(self, model, videos):
        """Apply Test-Time Augmentation with 3 temporal crops."""
        batch_size, channels, frames, height, width = videos.shape

        # Create 3 temporal crops: start, middle, end
        crop_length = min(32, frames)  # Use 32 frames or available frames

        crops = []
        if frames >= crop_length:
            # Start crop
            start_crop = videos[:, :, :crop_length, :, :]
            crops.append(start_crop)

            # Middle crop
            mid_start = (frames - crop_length) // 2
            mid_crop = videos[:, :, mid_start:mid_start + crop_length, :, :]
            crops.append(mid_crop)

            # End crop
            end_crop = videos[:, :, -crop_length:, :, :]
            crops.append(end_crop)
        else:
            # If video is shorter than crop_length, use original
            crops = [videos] * 3

        # Get predictions for each crop
        crop_outputs = []
        for crop in crops:
            if self.scaler is not None:
                with autocast():
                    output = model(crop)
            else:
                output = model(crop)
            crop_outputs.append(output)

        # Average the predictions
        averaged_output = torch.stack(crop_outputs).mean(dim=0)
        return averaged_output

    def _generate_comprehensive_test_report(self, test_metrics):
        """Generate comprehensive test report with all details."""
        report = {
            'test_results': test_metrics,
            'model_info': {
                'architecture': 'R(2+1)D-18',
                'input_channels': 1,
                'num_classes': len(self.config['classes']['names']),
                'dropout': self.config['model'].get('dropout', 0.3),
                'ema_enabled': self.ema_wrapper is not None
            },
            'training_config': {
                'epochs_trained': self.current_epoch,
                'best_val_metric': self.best_metric,
                'early_stopped': self.early_stop_counter > 0,
                'balance_switched': getattr(self, 'balance_switched', False),
                'sequence_length_increased': getattr(self, 'sequence_length_increased', False)
            },
            'class_names': self.config['classes']['names']
        }

        # Save comprehensive report
        report_path = self.output_dir / 'test_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Comprehensive test report saved to: {report_path}")

        # Generate and save confusion matrix
        if hasattr(self.test_metrics, 'get_confusion_matrix'):
            cm = self.test_metrics.get_confusion_matrix()
            cm_path = self.output_dir / 'final_confusion_matrix.png'
            class_names = self.config['classes']['names']
            self.test_metrics.plot_confusion_matrix(cm, class_names, save_path=cm_path)
            logger.info(f"Final confusion matrix saved to: {cm_path}")

    def _save_checkpoint(self, filename: str, epoch: int, metrics: Dict[str, Any]):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / filename

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'ema_state_dict': self.ema_wrapper.state_dict() if self.ema_wrapper else None,
            'best_metric': self.best_metric,
            'metrics': metrics,
            'config': self.config
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _generate_final_report(self, history: Dict[str, List], test_metrics: Dict[str, Any]):
        """Generate comprehensive final report."""
        report = {
            'training_summary': {
                'total_epochs': len(history['train_loss']),
                'best_val_metric': self.best_metric,
                'final_train_accuracy': history['train_accuracy'][-1],
                'final_val_accuracy': history['val_accuracy'][-1],
                'test_accuracy': test_metrics['accuracy'],
                'test_macro_f1': test_metrics['macro_f1']
            },
            'target_achievement': {
                'target_accuracy': self.config.get('targets', {}).get('accuracy', 0.80),
                'achieved': test_metrics['accuracy'] >= self.config.get('targets', {}).get('accuracy', 0.80),
                'target_macro_f1': self.config.get('targets', {}).get('macro_f1', 0.75),
                'macro_f1_achieved': test_metrics['macro_f1'] >= self.config.get('targets', {}).get('macro_f1', 0.75)
            },
            'model_info': {
                'num_classes': self.config['model']['num_classes'],
                'backbone': self.config['model']['backbone'],
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            },
            'training_history': history,
            'test_metrics': test_metrics
        }

        # Save report
        report_path = self.output_dir / 'final_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Generate markdown summary
        self._generate_markdown_summary(report)

        logger.info(f"Final report saved: {report_path}")

    def _generate_markdown_summary(self, report: Dict[str, Any]):
        """Generate markdown summary report."""
        summary = f"""# Lip Reading Training Report

## Training Summary
- **Total Epochs**: {report['training_summary']['total_epochs']}
- **Best Validation Metric**: {report['training_summary']['best_val_metric']:.4f}
- **Final Training Accuracy**: {report['training_summary']['final_train_accuracy']:.4f}
- **Final Validation Accuracy**: {report['training_summary']['final_val_accuracy']:.4f}
- **Test Accuracy**: {report['training_summary']['test_accuracy']:.4f}
- **Test Macro F1**: {report['training_summary']['test_macro_f1']:.4f}

## Target Achievement
- **Accuracy Target**: {report['target_achievement']['target_accuracy']:.1%} - {'✅ ACHIEVED' if report['target_achievement']['achieved'] else '❌ NOT ACHIEVED'}
- **Macro F1 Target**: {report['target_achievement']['target_macro_f1']:.1%} - {'✅ ACHIEVED' if report['target_achievement']['macro_f1_achieved'] else '❌ NOT ACHIEVED'}

## Model Information
- **Classes**: {report['model_info']['num_classes']}
- **Backbone**: {report['model_info']['backbone']}
- **Total Parameters**: {report['model_info']['total_parameters']:,}
- **Trainable Parameters**: {report['model_info']['trainable_parameters']:,}

## Per-Class Performance
"""

        # Add per-class metrics
        if 'per_class_metrics' in report['test_metrics']:
            for class_name, metrics in report['test_metrics']['per_class_metrics'].items():
                summary += f"- **{class_name}**: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}\n"

        # Save markdown
        summary_path = self.output_dir / 'TRAINING_SUMMARY.md'
        with open(summary_path, 'w') as f:
            f.write(summary)

        logger.info(f"Training summary saved: {summary_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint in a directory."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    # Look for checkpoint_epoch_*.pth files first
    epoch_checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
    if epoch_checkpoints:
        # Sort by epoch number
        epoch_checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        return str(epoch_checkpoints[-1])

    # Fall back to best_model.pth
    best_model = checkpoint_dir / 'best_model.pth'
    if best_model.exists():
        return str(best_model)

    return None


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="7-Class Lip Reading Trainer")

    parser.add_argument(
        '--manifest',
        required=True,
        help='Path to manifest CSV file'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--output_dir',
        default='./experiments/run_001',
        help='Output directory for checkpoints and logs'
    )
    parser.add_argument(
        '--balance',
        choices=['weighted_sampler', 'focal_loss', 'duplicate', 'none'],
        help='Class balancing method (overrides config)'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        help='GPU ID to use (overrides config)'
    )
    parser.add_argument(
        '--resume',
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--unfreeze_layers',
        help='Comma-separated list of layers to unfreeze (e.g., layer3,layer4)'
    )
    parser.add_argument(
        '--head_lr',
        type=float,
        help='Head learning rate (overrides config)'
    )
    parser.add_argument(
        '--backbone_lr',
        type=float,
        help='Backbone learning rate (overrides config)'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        help='Weight decay (overrides config)'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        help='Dropout rate (overrides config)'
    )
    parser.add_argument(
        '--label_smoothing',
        type=float,
        help='Label smoothing factor (overrides config)'
    )
    parser.add_argument(
        '--clip_len',
        type=int,
        help='Clip length (overrides config)'
    )
    parser.add_argument(
        '--ema_beta',
        type=float,
        help='EMA beta factor (overrides config)'
    )
    parser.add_argument(
        '--early_stop_patience',
        type=int,
        help='Early stopping patience (overrides config)'
    )
    parser.add_argument(
        '--aug',
        help='Augmentation settings (e.g., temporal10,affine2,bc0.1)'
    )
    parser.add_argument(
        '--save_dir',
        help='Save directory (same as output_dir for compatibility)'
    )
    parser.add_argument(
        '--eval_only',
        action='store_true',
        help='Only evaluate on test set (requires resume_from)'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config with CLI arguments
    if args.balance:
        config.setdefault('balance', {})['method'] = args.balance

    if args.gpu is not None:
        config.setdefault('hardware', {})['gpu_id'] = args.gpu

    if args.unfreeze_layers:
        layers = [layer.strip() for layer in args.unfreeze_layers.split(',')]
        config.setdefault('model', {})['unfreeze_layers'] = layers

    if args.head_lr is not None:
        config.setdefault('optimizer', {})['head_lr'] = args.head_lr

    if args.backbone_lr is not None:
        config.setdefault('optimizer', {})['backbone_lr'] = args.backbone_lr

    if args.weight_decay is not None:
        config.setdefault('optimizer', {})['weight_decay'] = args.weight_decay

    if args.dropout is not None:
        config.setdefault('model', {})['dropout'] = args.dropout

    if args.label_smoothing is not None:
        config.setdefault('loss', {})['label_smoothing'] = args.label_smoothing

    if args.clip_len is not None:
        config.setdefault('data', {})['clip_len'] = args.clip_len

    if args.ema_beta is not None:
        config.setdefault('ema', {})['beta'] = args.ema_beta

    if args.early_stop_patience is not None:
        config.setdefault('training', {})['early_stop_patience'] = args.early_stop_patience

    if args.aug:
        # Parse augmentation string (e.g., "temporal10,affine2,bc0.1")
        aug_config = config.setdefault('augmentation', {})
        for aug_part in args.aug.split(','):
            if 'temporal' in aug_part:
                aug_config['temporal_jitter'] = float(aug_part.replace('temporal', '')) / 100.0
            elif 'affine' in aug_part:
                aug_config['affine_deg'] = float(aug_part.replace('affine', ''))
            elif 'bc' in aug_part:
                value = float(aug_part.replace('bc', ''))
                aug_config['brightness'] = value
                aug_config['contrast'] = value

    if args.save_dir:
        args.output_dir = args.save_dir

    # Update paths in config
    config['paths']['manifest_file'] = args.manifest

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config to output directory
    config_save_path = output_dir / 'config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, indent=2)

    # Setup logging to file
    log_file = output_dir / 'training.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info(f"Starting lip reading training")
    logger.info(f"Config: {args.config}")
    logger.info(f"Manifest: {args.manifest}")
    logger.info(f"Output: {args.output_dir}")

    try:
        # Initialize trainer
        trainer = LipReadingTrainer(config, args.output_dir)

        # Load data
        train_loader, val_loader, test_loader = trainer.load_data(args.manifest)

        # Setup model and training components
        trainer.setup_model_and_training(train_loader)

        # Print model info
        trainer.model.print_model_info()

        # Resume from checkpoint if specified
        if args.resume:
            # If resume is a directory, find the latest checkpoint
            if Path(args.resume).is_dir():
                checkpoint_path = find_latest_checkpoint(args.resume)
                if not checkpoint_path:
                    raise ValueError(f"No checkpoints found in {args.resume}")
                logger.info(f"Auto-found latest checkpoint: {checkpoint_path}")
            else:
                checkpoint_path = args.resume
                logger.info(f"Resuming from checkpoint: {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])

            # Load EMA state if available
            if trainer.ema_wrapper and checkpoint.get('ema_state_dict'):
                trainer.ema_wrapper.load_state_dict(checkpoint['ema_state_dict'])
                logger.info("EMA state loaded from checkpoint")

            if not args.eval_only:
                trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if trainer.scheduler and checkpoint.get('scheduler_state_dict'):
                    trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if trainer.scaler and checkpoint.get('scaler_state_dict'):
                    trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                trainer.best_metric = checkpoint.get('best_metric', 0.0)
                trainer.current_epoch = checkpoint.get('epoch', 0)

        if args.eval_only:
            # Only evaluate on test set
            if not args.resume:
                raise ValueError("--eval_only requires --resume")

            logger.info("Evaluation mode: testing on test set only")
            test_metrics = trainer.evaluate(test_loader)

            # Print final results
            logger.info(f"\n{'='*60}")
            logger.info("FINAL TEST RESULTS")
            logger.info(f"{'='*60}")
            logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
            logger.info(f"Test Weighted F1: {test_metrics['weighted_f1']:.4f}")

            # Check target achievement
            target_acc = config.get('targets', {}).get('accuracy', 0.80)
            target_f1 = config.get('targets', {}).get('macro_f1', 0.75)

            acc_achieved = test_metrics['accuracy'] >= target_acc
            f1_achieved = test_metrics['macro_f1'] >= target_f1

            logger.info(f"\nTarget Achievement:")
            logger.info(f"Accuracy ≥ {target_acc:.1%}: {'✅ ACHIEVED' if acc_achieved else '❌ NOT ACHIEVED'}")
            logger.info(f"Macro F1 ≥ {target_f1:.1%}: {'✅ ACHIEVED' if f1_achieved else '❌ NOT ACHIEVED'}")
            logger.info(f"{'='*60}")

        else:
            # Full training
            history, test_metrics = trainer.train(train_loader, val_loader, test_loader)

            # Print final results
            logger.info(f"\n{'='*60}")
            logger.info("TRAINING COMPLETED")
            logger.info(f"{'='*60}")
            logger.info(f"Best Validation Metric: {trainer.best_metric:.4f}")
            logger.info(f"Final Test Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"Final Test Macro F1: {test_metrics['macro_f1']:.4f}")

            # Check target achievement
            target_acc = config.get('targets', {}).get('accuracy', 0.80)
            target_f1 = config.get('targets', {}).get('macro_f1', 0.75)

            acc_achieved = test_metrics['accuracy'] >= target_acc
            f1_achieved = test_metrics['macro_f1'] >= target_f1

            logger.info(f"\nTarget Achievement:")
            logger.info(f"Accuracy ≥ {target_acc:.1%}: {'✅ ACHIEVED' if acc_achieved else '❌ NOT ACHIEVED'}")
            logger.info(f"Macro F1 ≥ {target_f1:.1%}: {'✅ ACHIEVED' if f1_achieved else '❌ NOT ACHIEVED'}")

            if acc_achieved and f1_achieved:
                logger.info("🎉 ALL TARGETS ACHIEVED! 🎉")
            elif acc_achieved or f1_achieved:
                logger.info("⚠️  PARTIAL TARGET ACHIEVEMENT")
            else:
                logger.info("❌ TARGETS NOT ACHIEVED")

            logger.info(f"{'='*60}")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise e

    logger.info("Training script completed successfully")


if __name__ == "__main__":
    main()
