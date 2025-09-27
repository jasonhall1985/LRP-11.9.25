#!/usr/bin/env python3
"""
Cosine Similarity Classification Head with ArcFace Loss Support
==============================================================

Implements cosine similarity-based classification head for better speaker-invariant
features in lip-reading tasks. Supports ArcFace loss for larger inter-class margins.

Key Features:
- L2 normalized features and weights for cosine similarity
- Temperature scaling for calibrated predictions
- ArcFace margin support for enhanced discrimination
- Parameter count <100k constraint maintained

Author: Augment Agent
Date: 2025-09-27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CosineFCHead(nn.Module):
    """
    Cosine similarity-based classification head with ArcFace support.
    
    Uses L2 normalized features and weights to compute cosine similarity,
    which is more robust to speaker variations than standard linear layers.
    """
    
    def __init__(self, input_dim=256, num_classes=4, temperature=10.0, margin=0.5):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.temperature = temperature
        self.margin = margin
        
        # Feature normalization layer
        self.feature_norm = nn.LayerNorm(input_dim)
        
        # Cosine similarity weights (no bias needed)
        self.weight = nn.Parameter(torch.randn(num_classes, input_dim))
        nn.init.xavier_uniform_(self.weight)
        
        # Learnable temperature parameter
        self.scale = nn.Parameter(torch.tensor(temperature))
        
        # Calculate parameter count
        self.param_count = self._count_parameters()
        print(f"CosineFCHead created with {self.param_count:,} parameters")
        
        # Verify parameter constraint
        if self.param_count >= 100000:
            raise ValueError(f"Parameter count {self.param_count} exceeds 100k limit")
    
    def _count_parameters(self):
        """Count total parameters in the head"""
        total = 0
        for param in self.parameters():
            total += param.numel()
        return total
    
    def forward(self, x, labels=None, use_arcface=False):
        """
        Forward pass with optional ArcFace margin.
        
        Args:
            x: Input features [batch_size, input_dim]
            labels: Ground truth labels for ArcFace [batch_size] (optional)
            use_arcface: Whether to apply ArcFace margin during training
            
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        batch_size = x.size(0)
        
        # Normalize input features
        x = self.feature_norm(x)
        x_norm = F.normalize(x, p=2, dim=1)
        
        # Normalize weights
        w_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cosine = F.linear(x_norm, w_norm)  # [batch_size, num_classes]
        
        # Apply ArcFace margin during training if requested
        if use_arcface and labels is not None and self.training:
            cosine = self._apply_arcface_margin(cosine, labels)
        
        # Scale by temperature
        logits = cosine * self.scale
        
        return logits
    
    def _apply_arcface_margin(self, cosine, labels):
        """Apply ArcFace margin to target class cosine similarities"""
        # Convert labels to one-hot for indexing
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        # Compute theta (angle) from cosine
        theta = torch.acos(torch.clamp(cosine, -1 + 1e-7, 1 - 1e-7))
        
        # Add margin to target class angles
        target_theta = theta + self.margin
        target_cosine = torch.cos(target_theta)
        
        # Replace target class cosines with margin-adjusted values
        cosine_with_margin = cosine * (1 - one_hot) + target_cosine * one_hot
        
        return cosine_with_margin


class ArcFaceLoss(nn.Module):
    """
    ArcFace loss implementation for enhanced discrimination.
    
    Combines cross-entropy loss with angular margin penalty to increase
    inter-class separability and intra-class compactness.
    """
    
    def __init__(self, margin=0.5, scale=64.0, label_smoothing=0.0):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.label_smoothing = label_smoothing
        
        if label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, cosine_logits, labels):
        """
        Compute ArcFace loss.
        
        Args:
            cosine_logits: Cosine similarity logits from CosineFCHead
            labels: Ground truth labels
            
        Returns:
            loss: ArcFace loss value
        """
        # The margin is already applied in the forward pass of CosineFCHead
        # when use_arcface=True, so we just compute cross-entropy here
        return self.criterion(cosine_logits, labels)


def create_cosine_head(input_dim=256, num_classes=4, temperature=10.0, margin=0.5):
    """
    Factory function to create cosine classification head.
    
    Args:
        input_dim: Input feature dimension
        num_classes: Number of output classes
        temperature: Temperature scaling parameter
        margin: ArcFace margin for angular penalty
        
    Returns:
        head: CosineFCHead instance
    """
    return CosineFCHead(
        input_dim=input_dim,
        num_classes=num_classes,
        temperature=temperature,
        margin=margin
    )


if __name__ == "__main__":
    # Test the cosine head
    print("Testing CosineFCHead...")
    
    # Create head
    head = create_cosine_head(input_dim=256, num_classes=4)
    
    # Test forward pass
    batch_size = 8
    x = torch.randn(batch_size, 256)
    labels = torch.randint(0, 4, (batch_size,))
    
    # Standard forward pass
    logits = head(x)
    print(f"Standard logits shape: {logits.shape}")
    
    # ArcFace forward pass
    arcface_logits = head(x, labels, use_arcface=True)
    print(f"ArcFace logits shape: {arcface_logits.shape}")
    
    # Test loss
    loss_fn = ArcFaceLoss(margin=0.5, label_smoothing=0.05)
    loss = loss_fn(arcface_logits, labels)
    print(f"ArcFace loss: {loss.item():.4f}")
    
    print(f"✅ CosineFCHead test passed! Parameters: {head.param_count:,}")
