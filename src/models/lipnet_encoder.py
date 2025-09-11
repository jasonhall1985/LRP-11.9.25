"""
LipNet-inspired encoder architecture for lip reading.
Implements 3D convolutions followed by BiGRU layers for temporal modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Conv3DBlock(nn.Module):
    """3D Convolutional block with normalization and activation."""
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int, int] = (3, 3, 3),
                 stride: Tuple[int, int, int] = (1, 1, 1),
                 padding: Tuple[int, int, int] = (1, 1, 1),
                 dropout_rate: float = 0.2):
        """
        Initialize 3D convolutional block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size (T, H, W)
            stride: Convolution stride (T, H, W)
            padding: Convolution padding (T, H, W)
            dropout_rate: Dropout probability
        """
        super(Conv3DBlock, self).__init__()
        
        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        
        # Use Instance Normalization for better performance with small batches
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
            
        Returns:
            Output tensor after 3D convolution, normalization, activation, and dropout
        """
        x = self.conv3d(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class LipNetEncoder(nn.Module):
    """
    LipNet-inspired encoder for lip reading.
    
    Architecture:
    - 3D Convolutional layers for spatial-temporal feature extraction
    - Temporal pooling to reduce sequence length
    - BiGRU layers for temporal modeling
    - Output embedding for downstream tasks
    """
    
    def __init__(self,
                 input_channels: int = 1,
                 conv_channels: Tuple[int, ...] = (32, 64, 96),
                 conv_kernel_size: int = 3,
                 conv_stride: int = 1,
                 conv_padding: int = 1,
                 dropout_rate: float = 0.2,
                 gru_hidden_size: int = 256,
                 gru_num_layers: int = 2,
                 gru_dropout: float = 0.2,
                 embedding_dim: int = 512):
        """
        Initialize LipNet encoder.
        
        Args:
            input_channels: Number of input channels (1 for grayscale)
            conv_channels: Number of channels for each 3D conv layer
            conv_kernel_size: Kernel size for 3D convolutions
            conv_stride: Stride for 3D convolutions
            conv_padding: Padding for 3D convolutions
            dropout_rate: Dropout rate for conv layers
            gru_hidden_size: Hidden size for GRU layers
            gru_num_layers: Number of GRU layers
            gru_dropout: Dropout rate for GRU layers
            embedding_dim: Output embedding dimension
        """
        super(LipNetEncoder, self).__init__()
        
        self.input_channels = input_channels
        self.conv_channels = conv_channels
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.embedding_dim = embedding_dim
        
        # 3D Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        
        for out_channels in conv_channels:
            self.conv_layers.append(
                Conv3DBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(conv_kernel_size, conv_kernel_size, conv_kernel_size),
                    stride=(conv_stride, conv_stride, conv_stride),
                    padding=(conv_padding, conv_padding, conv_padding),
                    dropout_rate=dropout_rate
                )
            )
            in_channels = out_channels
        
        # Temporal pooling layers (reduce temporal dimension)
        self.temporal_pools = nn.ModuleList([
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),  # After first conv
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),  # After second conv
        ])
        
        # Spatial pooling (reduce spatial dimensions)
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 4, 4))  # Keep temporal, pool spatial to 4x4
        
        # Calculate the flattened feature size after conv layers
        # This will be computed dynamically in the first forward pass
        self.conv_output_size = None
        
        # BiGRU layers for temporal modeling
        self.gru = nn.GRU(
            input_size=conv_channels[-1] * 4 * 4,  # Assuming 4x4 spatial after pooling
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=gru_dropout if gru_num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output projection to embedding dimension
        self.output_projection = nn.Linear(
            gru_hidden_size * 2,  # *2 for bidirectional
            embedding_dim
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def _compute_conv_output_size(self, x: torch.Tensor) -> int:
        """Compute the output size after convolutional layers."""
        with torch.no_grad():
            # Pass through conv layers
            for i, conv_layer in enumerate(self.conv_layers):
                x = conv_layer(x)
                # Apply temporal pooling after first two conv layers
                if i < len(self.temporal_pools):
                    x = self.temporal_pools[i](x)
            
            # Apply spatial pooling
            x = self.spatial_pool(x)
            
            # Get the feature size per timestep
            B, C, T, H, W = x.shape
            return C * H * W
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape (B, T, H, W, C) or (B, C, T, H, W)
            
        Returns:
            Encoded features of shape (B, T, embedding_dim)
        """
        # Ensure input is in the correct format (B, C, T, H, W)
        if x.dim() == 5 and x.shape[1] != self.input_channels:
            # Input is (B, T, H, W, C), convert to (B, C, T, H, W)
            x = x.permute(0, 4, 1, 2, 3)
        elif x.dim() == 4:
            # Input is (B, T, H, W), add channel dimension
            x = x.unsqueeze(1)  # (B, 1, T, H, W)
        
        batch_size = x.shape[0]
        
        # Pass through 3D convolutional layers
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            # Apply temporal pooling after first two conv layers
            if i < len(self.temporal_pools):
                x = self.temporal_pools[i](x)
        
        # Apply spatial pooling
        x = self.spatial_pool(x)  # (B, C, T, 4, 4)
        
        # Compute conv output size if not already computed
        if self.conv_output_size is None:
            self.conv_output_size = x.shape[1] * x.shape[3] * x.shape[4]  # C * H * W
        
        # Reshape for GRU: (B, T, C*H*W)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        x = x.contiguous().view(B, T, C * H * W)  # (B, T, C*H*W)
        
        # Pass through BiGRU
        gru_output, _ = self.gru(x)  # (B, T, hidden_size * 2)
        
        # Apply dropout
        gru_output = self.dropout(gru_output)
        
        # Project to embedding dimension
        output = self.output_projection(gru_output)  # (B, T, embedding_dim)
        
        return output
    
    def get_feature_extractor(self) -> nn.Module:
        """
        Get the convolutional feature extractor part of the encoder.
        Useful for transfer learning.
        """
        class FeatureExtractor(nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.conv_layers = encoder.conv_layers
                self.temporal_pools = encoder.temporal_pools
                self.spatial_pool = encoder.spatial_pool
                
            def forward(self, x):
                for i, conv_layer in enumerate(self.conv_layers):
                    x = conv_layer(x)
                    if i < len(self.temporal_pools):
                        x = self.temporal_pools[i](x)
                x = self.spatial_pool(x)
                return x
        
        return FeatureExtractor(self)


def test_lipnet_encoder():
    """Test the LipNet encoder with sample input."""
    # Create sample input (B=2, T=16, H=96, W=96, C=1)
    batch_size = 2
    seq_length = 16
    height, width = 96, 96
    channels = 1
    
    # Test with different input formats
    print("Testing LipNet Encoder...")
    
    # Format 1: (B, T, H, W, C)
    x1 = torch.randn(batch_size, seq_length, height, width, channels)
    print(f"Input shape (B, T, H, W, C): {x1.shape}")
    
    # Format 2: (B, C, T, H, W)
    x2 = torch.randn(batch_size, channels, seq_length, height, width)
    print(f"Input shape (B, C, T, H, W): {x2.shape}")
    
    # Create encoder
    encoder = LipNetEncoder(
        input_channels=1,
        conv_channels=(32, 64, 96),
        gru_hidden_size=256,
        gru_num_layers=2,
        embedding_dim=512
    )
    
    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # Test forward pass
    with torch.no_grad():
        output1 = encoder(x1)
        output2 = encoder(x2)
        
        print(f"Output shape 1: {output1.shape}")
        print(f"Output shape 2: {output2.shape}")
        
        # Check that outputs have the expected shape
        expected_seq_len = seq_length // 4  # Due to temporal pooling
        assert output1.shape == (batch_size, expected_seq_len, 512)
        assert output2.shape == (batch_size, expected_seq_len, 512)
        
        print("✅ All tests passed!")


if __name__ == "__main__":
    test_lipnet_encoder()
