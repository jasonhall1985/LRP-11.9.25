#!/usr/bin/env python3
"""
Class Balancing Utilities for Lip Reading Dataset
=================================================

Provides multiple strategies for handling class imbalance:
1. weighted_sampler: WeightedRandomSampler with inverse frequency weighting
2. focal_loss: FocalLoss for imbalanced classes
3. duplicate: Create physical copies of videos to balance classes
4. none: No balancing (use natural distribution)

Features:
- Physical file duplication with organized directory structure
- Weighted sampling strategies
- Focal loss implementation
- Comprehensive balancing statistics

Author: Production Lip Reading System
Date: 2025-09-15
"""

import os
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import logging
from tqdm import tqdm
import argparse

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Reference: Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal loss for dense object detection. ICCV, 2017.
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Class weights tensor (num_classes,)
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Predictions (B, num_classes)
            targets: Ground truth labels (B,)
            
        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancer:
    """
    Comprehensive class balancing utilities.
    """
    
    def __init__(self, class_names: List[str]):
        """
        Initialize class balancer.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        
    def analyze_class_distribution(self, manifest_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze class distribution in the dataset.
        
        Args:
            manifest_df: Dataset manifest DataFrame
            
        Returns:
            Dictionary with distribution statistics
        """
        class_counts = manifest_df['class'].value_counts()
        
        # Ensure all classes are represented
        for class_name in self.class_names:
            if class_name not in class_counts:
                class_counts[class_name] = 0
                
        total_samples = len(manifest_df)
        
        stats = {
            'total_samples': total_samples,
            'class_counts': dict(class_counts),
            'class_percentages': {k: (v / total_samples) * 100 for k, v in class_counts.items()},
            'min_count': class_counts.min(),
            'max_count': class_counts.max(),
            'imbalance_ratio': class_counts.max() / max(class_counts.min(), 1),
            'std_dev': class_counts.std()
        }
        
        return stats
        
    def get_weighted_sampler(
        self,
        manifest_df: pd.DataFrame,
        weight_mode: str = 'inverse_sqrt'
    ) -> WeightedRandomSampler:
        """
        Create WeightedRandomSampler for class balancing.
        
        Args:
            manifest_df: Dataset manifest DataFrame
            weight_mode: Weighting strategy ('inverse', 'inverse_sqrt', 'effective_num')
            
        Returns:
            WeightedRandomSampler instance
        """
        class_counts = manifest_df['class'].value_counts()
        
        # Calculate class weights
        if weight_mode == 'inverse':
            class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        elif weight_mode == 'inverse_sqrt':
            class_weights = {cls: 1.0 / np.sqrt(count) for cls, count in class_counts.items()}
        elif weight_mode == 'effective_num':
            # Effective number of samples (Cui et al., 2019)
            beta = 0.9999
            effective_nums = {cls: (1 - beta**count) / (1 - beta) for cls, count in class_counts.items()}
            class_weights = {cls: 1.0 / eff_num for cls, eff_num in effective_nums.items()}
        else:
            raise ValueError(f"Unknown weight_mode: {weight_mode}")
            
        # Normalize weights
        total_weight = sum(class_weights.values())
        class_weights = {cls: weight / total_weight * len(class_weights) 
                        for cls, weight in class_weights.items()}
        
        # Create sample weights
        sample_weights = []
        for _, row in manifest_df.iterrows():
            class_name = row['class']
            sample_weights.append(class_weights[class_name])
            
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
    def get_focal_loss(
        self,
        manifest_df: pd.DataFrame,
        gamma: float = 2.0,
        alpha_mode: str = 'inverse'
    ) -> FocalLoss:
        """
        Create FocalLoss for class balancing.
        
        Args:
            manifest_df: Dataset manifest DataFrame
            gamma: Focusing parameter
            alpha_mode: Alpha calculation mode ('inverse', 'inverse_sqrt', None)
            
        Returns:
            FocalLoss instance
        """
        if alpha_mode is None:
            alpha = None
        else:
            class_counts = manifest_df['class'].value_counts()
            
            if alpha_mode == 'inverse':
                alpha_dict = {cls: 1.0 / count for cls, count in class_counts.items()}
            elif alpha_mode == 'inverse_sqrt':
                alpha_dict = {cls: 1.0 / np.sqrt(count) for cls, count in class_counts.items()}
            else:
                raise ValueError(f"Unknown alpha_mode: {alpha_mode}")
                
            # Normalize alpha values
            total_alpha = sum(alpha_dict.values())
            alpha_dict = {cls: alpha / total_alpha * len(alpha_dict) 
                         for cls, alpha in alpha_dict.items()}
            
            # Create alpha tensor in class order
            alpha_values = [alpha_dict.get(cls, 1.0) for cls in self.class_names]
            alpha = torch.FloatTensor(alpha_values)
            
        return FocalLoss(alpha=alpha, gamma=gamma)
        
    def create_physical_duplicates(
        self,
        manifest_df: pd.DataFrame,
        output_dir: str,
        balance_source_demo: str = "gender=male,age_band=18-39"
    ) -> pd.DataFrame:
        """
        Create physical duplicate files to balance classes.
        
        Args:
            manifest_df: Dataset manifest DataFrame
            output_dir: Output directory for duplicated files
            balance_source_demo: Demographic criteria for source videos
            
        Returns:
            Updated manifest DataFrame with duplicate entries
        """
        logger.info("Creating physical duplicates for class balancing...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Parse balance source demographic criteria
        balance_criteria = {}
        for criterion in balance_source_demo.split(','):
            key, value = criterion.strip().split('=')
            balance_criteria[key.strip()] = value.strip()
            
        # Analyze current distribution
        stats = self.analyze_class_distribution(manifest_df)
        target_count = stats['max_count']
        
        logger.info(f"Target count per class: {target_count}")
        logger.info(f"Balance source demographic: {balance_criteria}")
        
        # Find source videos for duplication
        source_mask = pd.Series([True] * len(manifest_df))
        for key, value in balance_criteria.items():
            if key in manifest_df.columns:
                source_mask &= (manifest_df[key] == value)
                
        source_videos = manifest_df[source_mask].copy()
        logger.info(f"Found {len(source_videos)} source videos for duplication")
        
        if len(source_videos) == 0:
            logger.warning("No source videos found for duplication!")
            return manifest_df
            
        # Create duplicates for each class
        duplicated_entries = []
        duplication_stats = {}
        
        for class_name in self.class_names:
            current_count = stats['class_counts'][class_name]
            needed_count = target_count - current_count
            
            if needed_count <= 0:
                duplication_stats[class_name] = 0
                continue
                
            # Get source videos for this class
            class_source_videos = source_videos[source_videos['class'] == class_name]
            
            if len(class_source_videos) == 0:
                logger.warning(f"No source videos found for class '{class_name}'")
                duplication_stats[class_name] = 0
                continue
                
            # Create class-specific output directory
            class_output_dir = output_path / f"augmented_{class_name}"
            class_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create duplicates
            duplicates_created = 0
            for i in tqdm(range(needed_count), desc=f"Duplicating {class_name}"):
                # Cycle through available source videos
                source_idx = i % len(class_source_videos)
                source_video = class_source_videos.iloc[source_idx].copy()
                source_path = Path(source_video['path'])
                
                # Create duplicate filename
                duplicate_filename = f"duplicate_{i+1:04d}_{source_path.name}"
                duplicate_path = class_output_dir / duplicate_filename
                
                try:
                    # Copy the video file
                    shutil.copy2(source_path, duplicate_path)
                    
                    # Create manifest entry for duplicate
                    duplicate_entry = source_video.copy()
                    duplicate_entry['path'] = str(duplicate_path)
                    duplicate_entry['is_duplicate'] = True
                    duplicate_entry['duplicate_id'] = i + 1
                    duplicate_entry['duplicate_source'] = str(source_path)
                    duplicate_entry['duplicate_class_dir'] = str(class_output_dir)
                    
                    duplicated_entries.append(duplicate_entry)
                    duplicates_created += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to copy {source_path} to {duplicate_path}: {e}")
                    
            duplication_stats[class_name] = duplicates_created
            
        # Create updated manifest
        if duplicated_entries:
            # Add duplicate flags to original DataFrame
            manifest_df['is_duplicate'] = False
            manifest_df['duplicate_id'] = 0
            manifest_df['duplicate_source'] = ''
            manifest_df['duplicate_class_dir'] = ''
            
            # Combine original and duplicated entries
            duplicates_df = pd.DataFrame(duplicated_entries)
            balanced_df = pd.concat([manifest_df, duplicates_df], ignore_index=True)
        else:
            # Add duplicate flags even if no duplicates created
            manifest_df['is_duplicate'] = False
            manifest_df['duplicate_id'] = 0
            manifest_df['duplicate_source'] = ''
            manifest_df['duplicate_class_dir'] = ''
            balanced_df = manifest_df
            
        # Log results
        total_duplicates = sum(duplication_stats.values())
        logger.info(f"Physical duplication completed:")
        logger.info(f"  Total duplicates created: {total_duplicates}")
        logger.info(f"  Output directory: {output_path}")
        
        for class_name, count in duplication_stats.items():
            if count > 0:
                logger.info(f"  {class_name}: +{count} duplicates")
                
        # Verify final distribution
        final_stats = self.analyze_class_distribution(balanced_df)
        logger.info(f"Final class distribution:")
        for class_name in self.class_names:
            count = final_stats['class_counts'][class_name]
            logger.info(f"  {class_name}: {count}")
            
        return balanced_df
        
    def print_balance_analysis(self, manifest_df: pd.DataFrame):
        """Print comprehensive class balance analysis."""
        stats = self.analyze_class_distribution(manifest_df)
        
        logger.info("\n" + "="*60)
        logger.info("CLASS BALANCE ANALYSIS")
        logger.info("="*60)
        
        logger.info(f"\n📊 Overall Statistics:")
        logger.info(f"  Total samples: {stats['total_samples']:,}")
        logger.info(f"  Number of classes: {self.num_classes}")
        logger.info(f"  Min class count: {stats['min_count']:,}")
        logger.info(f"  Max class count: {stats['max_count']:,}")
        logger.info(f"  Imbalance ratio: {stats['imbalance_ratio']:.2f}")
        logger.info(f"  Standard deviation: {stats['std_dev']:.2f}")
        
        logger.info(f"\n📋 Class Distribution:")
        for class_name in self.class_names:
            count = stats['class_counts'][class_name]
            percentage = stats['class_percentages'][class_name]
            logger.info(f"  {class_name}: {count:,} ({percentage:.1f}%)")
            
        # Imbalance severity assessment
        if stats['imbalance_ratio'] < 2.0:
            severity = "MILD"
        elif stats['imbalance_ratio'] < 5.0:
            severity = "MODERATE"
        elif stats['imbalance_ratio'] < 10.0:
            severity = "SEVERE"
        else:
            severity = "EXTREME"
            
        logger.info(f"\n⚠️  Imbalance Severity: {severity}")
        
        # Recommendations
        logger.info(f"\n💡 Recommendations:")
        if stats['imbalance_ratio'] < 2.0:
            logger.info("  - No balancing needed")
            logger.info("  - Standard cross-entropy loss should work well")
        elif stats['imbalance_ratio'] < 5.0:
            logger.info("  - Consider weighted sampling or focal loss")
            logger.info("  - Monitor per-class metrics carefully")
        else:
            logger.info("  - Strong balancing recommended")
            logger.info("  - Use focal loss or physical duplication")
            logger.info("  - Consider data augmentation for minority classes")
            
        logger.info("="*60)


def main():
    """Main function for class balancing utilities."""
    parser = argparse.ArgumentParser(description="Class balancing utilities")
    
    parser.add_argument(
        '--manifest',
        required=True,
        help='Input manifest CSV file'
    )
    parser.add_argument(
        '--method',
        choices=['analyze', 'duplicate'],
        default='analyze',
        help='Balancing method'
    )
    parser.add_argument(
        '--output_dir',
        default='./augmented',
        help='Output directory for duplicated files'
    )
    parser.add_argument(
        '--balance_source_demo',
        default='gender=male,age_band=18-39',
        help='Demographic criteria for source videos'
    )
    
    args = parser.parse_args()
    
    # Load manifest
    manifest_df = pd.read_csv(args.manifest)
    
    # Initialize balancer
    class_names = [
        "help", "doctor", "glasses", "phone", "pillow", 
        "i_need_to_move", "my_mouth_is_dry"
    ]
    balancer = ClassBalancer(class_names)
    
    if args.method == 'analyze':
        # Analyze class distribution
        balancer.print_balance_analysis(manifest_df)
        
    elif args.method == 'duplicate':
        # Create physical duplicates
        balanced_df = balancer.create_physical_duplicates(
            manifest_df, args.output_dir, args.balance_source_demo
        )
        
        # Save updated manifest
        output_manifest = args.manifest.replace('.csv', '_balanced.csv')
        balanced_df.to_csv(output_manifest, index=False)
        logger.info(f"Balanced manifest saved to: {output_manifest}")
        
        # Print final analysis
        balancer.print_balance_analysis(balanced_df)


if __name__ == "__main__":
    main()
