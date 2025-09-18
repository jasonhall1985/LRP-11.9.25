#!/usr/bin/env python3
"""
Debug script to determine correct tensor shapes for the CNN-LSTM model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DebugCNN_LSTM(nn.Module):
    """Debug version to print tensor shapes."""
    
    def __init__(self):
        super(DebugCNN_LSTM, self).__init__()
        
        # 3D CNN layers
        self.conv3d1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=1)
        self.bn3d1 = nn.BatchNorm3d(32)
        self.pool3d1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        
        self.conv3d2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1)
        self.bn3d2 = nn.BatchNorm3d(64)
        self.pool3d2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv3d3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.bn3d3 = nn.BatchNorm3d(128)
        self.pool3d3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
    
    def forward(self, x):
        print(f"Input shape: {x.shape}")
        
        # 3D CNN feature extraction
        x = F.relu(self.bn3d1(self.conv3d1(x)))
        print(f"After conv3d1: {x.shape}")
        x = self.pool3d1(x)
        print(f"After pool3d1: {x.shape}")
        
        x = F.relu(self.bn3d2(self.conv3d2(x)))
        print(f"After conv3d2: {x.shape}")
        x = self.pool3d2(x)
        print(f"After pool3d2: {x.shape}")
        
        x = F.relu(self.bn3d3(self.conv3d3(x)))
        print(f"After conv3d3: {x.shape}")
        x = self.pool3d3(x)
        print(f"After pool3d3: {x.shape}")
        
        # Calculate actual feature size
        batch_size = x.size(0)
        print(f"Batch size: {batch_size}")
        
        # Reshape for LSTM: (B, T, Features)
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        print(f"After permute: {x.shape}")
        
        # Flatten spatial dimensions
        x = x.contiguous().view(batch_size, x.size(1), -1)
        print(f"After reshape for LSTM: {x.shape}")
        print(f"Feature size should be: {x.size(-1)}")
        
        return x

def main():
    print("🔍 Debugging CNN-LSTM tensor shapes")
    print("=" * 50)
    
    # Create debug model
    model = DebugCNN_LSTM()
    
    # Create sample input: (batch_size, channels, time, height, width)
    # Our videos are: (B, 1, 32, 64, 96)
    sample_input = torch.randn(2, 1, 32, 64, 96)  # Batch of 2
    
    print(f"Sample input shape: {sample_input.shape}")
    print()
    
    # Forward pass to see shapes
    with torch.no_grad():
        output = model(sample_input)
    
    print()
    print(f"✅ Final feature size: {output.size(-1)}")
    print(f"✅ Temporal dimension: {output.size(1)}")

if __name__ == "__main__":
    main()
