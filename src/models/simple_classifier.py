"""
Simple baseline classifier for ICU lip reading.
Uses basic CNN + LSTM architecture for faster training and demonstration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SimpleLipClassifier(nn.Module):
    """
    Simple baseline classifier for ICU lip reading.
    
    Architecture:
    - 2D CNN for spatial feature extraction per frame
    - LSTM for temporal modeling
    - Classification head
    """
    
    def __init__(self,
                 num_classes: int = 5,
                 input_channels: int = 1,
                 cnn_channels: Tuple[int, ...] = (16, 32, 64),
                 lstm_hidden_size: int = 128,
                 lstm_num_layers: int = 2,
                 dropout_rate: float = 0.3):
        """
        Initialize simple classifier.
        
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels (1 for grayscale)
            cnn_channels: Number of channels for CNN layers
            lstm_hidden_size: Hidden size for LSTM
            lstm_num_layers: Number of LSTM layers
            dropout_rate: Dropout rate
        """
        super(SimpleLipClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        
        # 2D CNN for per-frame feature extraction
        self.cnn_layers = nn.ModuleList()
        in_channels = input_channels
        
        for out_channels in cnn_channels:
            self.cnn_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout2d(dropout_rate)
                )
            )
            in_channels = out_channels
        
        # Calculate CNN output size
        # After 3 pooling layers: 96 -> 48 -> 24 -> 12
        self.cnn_output_size = cnn_channels[-1] * 12 * 12
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),  # *2 for bidirectional
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden_size, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, T, H, W, C) or (B, T, H, W)
            
        Returns:
            Class logits of shape (B, num_classes)
        """
        # Handle different input formats
        if x.dim() == 5:  # (B, T, H, W, C)
            if x.shape[-1] == self.input_channels:
                x = x.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)
            else:
                # Assume (B, C, T, H, W) format
                x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        elif x.dim() == 4:  # (B, T, H, W)
            x = x.unsqueeze(2)  # (B, T, 1, H, W)
        
        batch_size, seq_length = x.shape[:2]
        
        # Extract CNN features for each frame
        cnn_features = []
        for t in range(seq_length):
            frame = x[:, t]  # (B, C, H, W)
            
            # Pass through CNN layers
            for cnn_layer in self.cnn_layers:
                frame = cnn_layer(frame)
            
            # Flatten spatial dimensions
            frame = frame.reshape(batch_size, -1)  # (B, cnn_output_size)
            cnn_features.append(frame)
        
        # Stack features across time
        cnn_features = torch.stack(cnn_features, dim=1)  # (B, T, cnn_output_size)
        
        # Pass through LSTM
        lstm_output, _ = self.lstm(cnn_features)  # (B, T, lstm_hidden_size * 2)
        
        # Use last timestep output for classification
        final_output = lstm_output[:, -1]  # (B, lstm_hidden_size * 2)
        
        # Classify
        logits = self.classifier(final_output)  # (B, num_classes)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class indices."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)


def create_simple_classifier(num_classes: int = 5, **kwargs) -> SimpleLipClassifier:
    """
    Factory function to create simple classifier.
    
    Args:
        num_classes: Number of output classes
        **kwargs: Additional arguments for SimpleLipClassifier
        
    Returns:
        Simple classifier instance
    """
    default_config = {
        'input_channels': 1,
        'cnn_channels': (16, 32, 64),
        'lstm_hidden_size': 128,
        'lstm_num_layers': 2,
        'dropout_rate': 0.3
    }
    
    config = {**default_config, **kwargs}
    return SimpleLipClassifier(num_classes=num_classes, **config)


def test_simple_classifier():
    """Test the simple classifier."""
    print("Testing Simple Classifier...")
    
    # Create sample input
    batch_size = 2
    seq_length = 16
    height, width = 96, 96
    channels = 1
    
    x = torch.randn(batch_size, seq_length, height, width, channels)
    print(f"Input shape: {x.shape}")
    
    # Create classifier
    classifier = create_simple_classifier(num_classes=5)
    
    print(f"Classifier parameters: {sum(p.numel() for p in classifier.parameters()):,}")
    
    # Test forward pass
    with torch.no_grad():
        logits = classifier(x)
        probs = classifier.predict_proba(x)
        predictions = classifier.predict(x)
        
        print(f"Logits shape: {logits.shape}")
        print(f"Probabilities shape: {probs.shape}")
        print(f"Predictions shape: {predictions.shape}")
        
        # Check outputs
        assert logits.shape == (batch_size, 5)
        assert probs.shape == (batch_size, 5)
        assert predictions.shape == (batch_size,)
        
        # Check probabilities sum to 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-6)
        
        print("✅ All tests passed!")


if __name__ == "__main__":
    test_simple_classifier()
