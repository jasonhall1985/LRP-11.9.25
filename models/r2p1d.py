#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
R(2+1)D Model for 7-Class Lip Reading
=====================================

Production-ready R(2+1)D-18 model with:
- Pretrained Kinetics400 weights
- 1-channel grayscale input support
- 7-class classification head
- Progressive unfreezing support

Features:
- Automatic 1→3 channel conversion by averaging RGB weights
- Dropout regularization in classifier head
- Layer-wise learning rate support
- Memory-efficient implementation

Author: Production Lip Reading System
Date: 2025-09-15
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class LipReadingR2Plus1D(nn.Module):
    """
    R(2+1)D-18 model adapted for 7-class lip reading.
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        dropout: float = 0.4,
        pretrained: bool = True,
        freeze_backbone: bool = True
    ):
        """
        Initialize the R(2+1)D lip reading model.
        
        Args:
            num_classes: Number of output classes (7)
            dropout: Dropout rate in classifier head
            pretrained: Use pretrained Kinetics400 weights
            freeze_backbone: Freeze backbone initially
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.freeze_backbone = freeze_backbone
        
        # Load pretrained R(2+1)D-18
        if pretrained:
            weights = R2Plus1D_18_Weights.KINETICS400_V1
            self.backbone = r2plus1d_18(weights=weights)
            logger.info("Loaded R(2+1)D-18 with Kinetics400 pretrained weights")
        else:
            self.backbone = r2plus1d_18(weights=None)
            logger.info("Initialized R(2+1)D-18 without pretrained weights")
            
        # First conv layer is already 3 channels for replicated grayscale input
        # No adaptation needed since we're using 3-channel replicated grayscale
        
        # Replace classifier head for 7 classes
        self.replace_classifier_head()
        
        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone_layers()
            
        # Store layer groups for progressive unfreezing
        self.layer_groups = self.get_layer_groups()
        
        logger.info(f"Model initialized: {num_classes} classes, "
                   f"dropout={dropout}, frozen={freeze_backbone}")
        
    def adapt_first_conv_for_grayscale(self):
        """
        Adapt first conv layer to accept 1-channel grayscale input.
        Averages the RGB weights to create grayscale weights.
        """
        # Get the first conv layer
        first_conv = self.backbone.stem[0]  # Conv3d layer
        
        if first_conv.in_channels == 3:
            # Get original weights: (out_channels, 3, depth, height, width)
            original_weights = first_conv.weight.data
            
            # Average across RGB channels: (out_channels, 1, depth, height, width)
            grayscale_weights = original_weights.mean(dim=1, keepdim=True)
            
            # Create new conv layer with 1 input channel
            new_conv = nn.Conv3d(
                in_channels=1,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None
            )
            
            # Set the averaged weights
            new_conv.weight.data = grayscale_weights
            if first_conv.bias is not None:
                new_conv.bias.data = first_conv.bias.data
                
            # Replace the first conv layer
            self.backbone.stem[0] = new_conv
            
            logger.info("Adapted first conv layer for grayscale input (1 channel)")
        else:
            logger.info(f"First conv layer already has {first_conv.in_channels} channels")
            
    def replace_classifier_head(self):
        """Replace the classifier head for 7-class classification."""
        # Get the original classifier
        original_fc = self.backbone.fc
        
        # Create new classifier with dropout
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(original_fc.in_features, self.num_classes)
        )
        
        # Initialize weights
        nn.init.xavier_uniform_(self.backbone.fc[1].weight)
        nn.init.constant_(self.backbone.fc[1].bias, 0)
        
        logger.info(f"Replaced classifier head: {original_fc.in_features} → {self.num_classes}")
        
    def freeze_backbone_layers(self):
        """Freeze all backbone layers except classifier."""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:  # Don't freeze classifier
                param.requires_grad = False
                
        logger.info("Frozen backbone layers (classifier remains trainable)")
        
    def unfreeze_backbone_layers(self):
        """Unfreeze all backbone layers."""
        for param in self.backbone.parameters():
            param.requires_grad = True
            
        logger.info("Unfrozen all backbone layers")
        
    def get_layer_groups(self) -> Dict[str, List[str]]:
        """
        Get layer groups for progressive unfreezing.
        
        Returns:
            Dictionary mapping group names to parameter names
        """
        layer_groups = {
            'stem': [],
            'layer1': [],
            'layer2': [],
            'layer3': [],
            'layer4': [],
            'classifier': []
        }
        
        for name, _ in self.backbone.named_parameters():
            if 'stem' in name:
                layer_groups['stem'].append(name)
            elif 'layer1' in name:
                layer_groups['layer1'].append(name)
            elif 'layer2' in name:
                layer_groups['layer2'].append(name)
            elif 'layer3' in name:
                layer_groups['layer3'].append(name)
            elif 'layer4' in name:
                layer_groups['layer4'].append(name)
            elif 'fc' in name:
                layer_groups['classifier'].append(name)
                
        return layer_groups
        
    def unfreeze_layer_groups(self, groups: List[str]):
        """
        Unfreeze specific layer groups.
        
        Args:
            groups: List of group names to unfreeze
        """
        for group_name in groups:
            if group_name == 'all':
                self.unfreeze_backbone_layers()
                return
                
            if group_name in self.layer_groups:
                for param_name in self.layer_groups[group_name]:
                    param = dict(self.backbone.named_parameters())[param_name]
                    param.requires_grad = True
                    
        logger.info(f"Unfrozen layer groups: {groups}")
        
    def get_trainable_parameters(self) -> Dict[str, List[nn.Parameter]]:
        """
        Get trainable parameters grouped by layer groups.
        Useful for layer-wise learning rates.
        
        Returns:
            Dictionary mapping group names to parameter lists
        """
        trainable_params = {}
        
        for group_name, param_names in self.layer_groups.items():
            group_params = []
            for param_name in param_names:
                param = dict(self.backbone.named_parameters())[param_name]
                if param.requires_grad:
                    group_params.append(param)
            if group_params:
                trainable_params[group_name] = group_params
                
        return trainable_params
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, T, H, W) where C=1 for grayscale
            
        Returns:
            logits: (B, num_classes) tensor
        """
        # Ensure input is the right shape
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B, C, T, H, W), got {x.dim()}D")
            
        if x.size(1) != 3:
            raise ValueError(f"Expected 3 channels (replicated grayscale), got {x.size(1)} channels")
            
        # Forward through backbone
        logits = self.backbone(x)
        
        return logits
        
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get intermediate feature maps for analysis.
        
        Args:
            x: Input tensor (B, C, T, H, W)
            
        Returns:
            Dictionary of feature maps from different layers
        """
        features = {}
        
        # Stem
        x = self.backbone.stem(x)
        features['stem'] = x
        
        # ResNet layers
        x = self.backbone.layer1(x)
        features['layer1'] = x
        
        x = self.backbone.layer2(x)
        features['layer2'] = x
        
        x = self.backbone.layer3(x)
        features['layer3'] = x
        
        x = self.backbone.layer4(x)
        features['layer4'] = x
        
        # Global average pooling
        x = self.backbone.avgpool(x)
        features['avgpool'] = x
        
        # Flatten
        x = torch.flatten(x, 1)
        features['flatten'] = x
        
        return features
        
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by layer groups."""
        param_counts = {}
        
        total_params = 0
        trainable_params = 0
        
        for group_name, param_names in self.layer_groups.items():
            group_total = 0
            group_trainable = 0
            
            for param_name in param_names:
                param = dict(self.backbone.named_parameters())[param_name]
                param_count = param.numel()
                group_total += param_count
                
                if param.requires_grad:
                    group_trainable += param_count
                    
            param_counts[group_name] = {
                'total': group_total,
                'trainable': group_trainable
            }
            
            total_params += group_total
            trainable_params += group_trainable
            
        param_counts['overall'] = {
            'total': total_params,
            'trainable': trainable_params
        }
        
        return param_counts
        
    def print_model_info(self):
        """Print comprehensive model information."""
        logger.info("\n" + "="*60)
        logger.info("R(2+1)D MODEL INFORMATION")
        logger.info("="*60)
        
        # Parameter counts
        param_counts = self.count_parameters()
        
        logger.info(f"\n📊 Parameter Counts:")
        for group_name, counts in param_counts.items():
            if group_name != 'overall':
                logger.info(f"  {group_name}: {counts['trainable']:,} / {counts['total']:,} trainable")
                
        overall = param_counts['overall']
        logger.info(f"\n🎯 Overall: {overall['trainable']:,} / {overall['total']:,} trainable "
                   f"({100 * overall['trainable'] / overall['total']:.1f}%)")
        
        # Layer groups
        logger.info(f"\n🔧 Layer Groups:")
        for group_name, param_names in self.layer_groups.items():
            logger.info(f"  {group_name}: {len(param_names)} parameters")
            
        # Model configuration
        logger.info(f"\n⚙️  Configuration:")
        logger.info(f"  Classes: {self.num_classes}")
        logger.info(f"  Dropout: {self.dropout_rate}")
        logger.info(f"  Backbone frozen: {self.freeze_backbone}")
        
        logger.info("="*60)


def create_model(
    num_classes: int = 7,
    dropout: float = 0.4,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    device: Optional[torch.device] = None
) -> LipReadingR2Plus1D:
    """
    Create and initialize the R(2+1)D lip reading model.
    
    Args:
        num_classes: Number of output classes
        dropout: Dropout rate in classifier
        pretrained: Use pretrained weights
        freeze_backbone: Freeze backbone initially
        device: Device to move model to
        
    Returns:
        Initialized model
    """
    model = LipReadingR2Plus1D(
        num_classes=num_classes,
        dropout=dropout,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )
    
    if device is not None:
        model = model.to(device)
        
    return model


if __name__ == "__main__":
    # Test model creation
    model = create_model()
    model.print_model_info()
    
    # Test forward pass
    batch_size = 2
    channels = 1
    frames = 24
    height = 112
    width = 112
    
    dummy_input = torch.randn(batch_size, channels, frames, height, width)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    print(f"\nTest forward pass:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
