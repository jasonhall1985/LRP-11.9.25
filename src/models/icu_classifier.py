"""
ICU 5-Class Classifier with pretrained LipNet encoder.
Designed for critical ICU communication words: doctor, glasses, phone, pillow, help.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

# Handle imports for both module and script execution
try:
    from .lipnet_encoder import LipNetEncoder
except ImportError:
    from lipnet_encoder import LipNetEncoder


class TemporalAggregator(nn.Module):
    """Temporal aggregation module for sequence-level classification."""
    
    def __init__(self, 
                 input_dim: int,
                 aggregation_method: str = "attention"):
        """
        Initialize temporal aggregator.
        
        Args:
            input_dim: Input feature dimension
            aggregation_method: Method for temporal aggregation
                - "mean": Global average pooling
                - "max": Global max pooling
                - "attention": Self-attention based aggregation
                - "last": Use last timestep
        """
        super(TemporalAggregator, self).__init__()
        
        self.aggregation_method = aggregation_method
        self.input_dim = input_dim
        
        if aggregation_method == "attention":
            # Self-attention for temporal aggregation
            self.attention = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Aggregate temporal features.
        
        Args:
            x: Input features of shape (B, T, D)
            mask: Optional mask for variable length sequences (B, T)
            
        Returns:
            Aggregated features of shape (B, D)
        """
        if self.aggregation_method == "mean":
            if mask is not None:
                # Masked average
                mask_expanded = mask.unsqueeze(-1).float()  # (B, T, 1)
                x_masked = x * mask_expanded
                return x_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                return x.mean(dim=1)
                
        elif self.aggregation_method == "max":
            if mask is not None:
                # Masked max
                x_masked = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
                return x_masked.max(dim=1)[0]
            else:
                return x.max(dim=1)[0]
                
        elif self.aggregation_method == "last":
            if mask is not None:
                # Get last valid timestep for each sequence
                lengths = mask.sum(dim=1)  # (B,)
                batch_indices = torch.arange(x.size(0), device=x.device)
                return x[batch_indices, lengths - 1]
            else:
                return x[:, -1]  # Last timestep
                
        elif self.aggregation_method == "attention":
            # Self-attention aggregation
            if mask is not None:
                # Convert mask to attention mask format
                attn_mask = ~mask  # Invert mask (True = ignore)
            else:
                attn_mask = None
            
            # Apply self-attention
            attn_output, _ = self.attention(x, x, x, key_padding_mask=attn_mask)
            attn_output = self.layer_norm(attn_output + x)  # Residual connection
            
            # Global average pooling over attended features
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                attn_masked = attn_output * mask_expanded
                return attn_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                return attn_output.mean(dim=1)
        
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")


class ICUClassifier(nn.Module):
    """
    ICU 5-Class Classifier for critical communication words.
    
    Uses a pretrained LipNet encoder followed by temporal aggregation
    and classification layers.
    """
    
    def __init__(self,
                 encoder: Optional[LipNetEncoder] = None,
                 encoder_config: Optional[Dict[str, Any]] = None,
                 num_classes: int = 5,
                 hidden_dim: int = 256,
                 dropout_rate: float = 0.3,
                 aggregation_method: str = "attention",
                 freeze_encoder: bool = False):
        """
        Initialize ICU classifier.
        
        Args:
            encoder: Pretrained LipNet encoder (optional)
            encoder_config: Configuration for creating new encoder if encoder is None
            num_classes: Number of output classes (5 for ICU words)
            hidden_dim: Hidden dimension for classifier layers
            dropout_rate: Dropout rate for classifier
            aggregation_method: Method for temporal aggregation
            freeze_encoder: Whether to freeze encoder parameters
        """
        super(ICUClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder
        
        # Initialize encoder
        if encoder is not None:
            self.encoder = encoder
        else:
            # Create new encoder with default or provided config
            if encoder_config is None:
                encoder_config = {
                    'input_channels': 1,
                    'conv_channels': (32, 64, 96),
                    'gru_hidden_size': 256,
                    'gru_num_layers': 2,
                    'embedding_dim': 512
                }
            self.encoder = LipNetEncoder(**encoder_config)
        
        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Get encoder output dimension
        self.encoder_dim = self.encoder.embedding_dim
        
        # Temporal aggregation
        self.temporal_aggregator = TemporalAggregator(
            input_dim=self.encoder_dim,
            aggregation_method=aggregation_method
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_classifier_weights()
    
    def _initialize_classifier_weights(self):
        """Initialize classifier weights using Xavier initialization."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the classifier.
        
        Args:
            x: Input tensor of shape (B, T, H, W, C) or (B, C, T, H, W)
            mask: Optional mask for variable length sequences (B, T)
            
        Returns:
            Class logits of shape (B, num_classes)
        """
        # Extract features using encoder
        encoder_features = self.encoder(x)  # (B, T, encoder_dim)
        
        # Aggregate temporal features
        aggregated_features = self.temporal_aggregator(encoder_features, mask)  # (B, encoder_dim)
        
        # Classify
        logits = self.classifier(aggregated_features)  # (B, num_classes)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get class probabilities.
        
        Args:
            x: Input tensor
            mask: Optional mask for variable length sequences
            
        Returns:
            Class probabilities of shape (B, num_classes)
        """
        logits = self.forward(x, mask)
        return F.softmax(logits, dim=-1)
    
    def predict(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get predicted class indices.
        
        Args:
            x: Input tensor
            mask: Optional mask for variable length sequences
            
        Returns:
            Predicted class indices of shape (B,)
        """
        logits = self.forward(x, mask)
        return torch.argmax(logits, dim=-1)
    
    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.freeze_encoder = True
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.freeze_encoder = False
    
    def load_pretrained_encoder(self, encoder_path: str, strict: bool = True):
        """
        Load pretrained encoder weights.
        
        Args:
            encoder_path: Path to encoder checkpoint
            strict: Whether to strictly enforce key matching
        """
        checkpoint = torch.load(encoder_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            encoder_state_dict = checkpoint['model_state_dict']
        elif 'encoder_state_dict' in checkpoint:
            encoder_state_dict = checkpoint['encoder_state_dict']
        else:
            encoder_state_dict = checkpoint
        
        # Load encoder weights
        self.encoder.load_state_dict(encoder_state_dict, strict=strict)
        print(f"Loaded pretrained encoder from {encoder_path}")
    
    def get_feature_extractor(self) -> nn.Module:
        """Get the feature extraction part (encoder + aggregator)."""
        class FeatureExtractor(nn.Module):
            def __init__(self, classifier):
                super().__init__()
                self.encoder = classifier.encoder
                self.temporal_aggregator = classifier.temporal_aggregator
                
            def forward(self, x, mask=None):
                encoder_features = self.encoder(x)
                aggregated_features = self.temporal_aggregator(encoder_features, mask)
                return aggregated_features
        
        return FeatureExtractor(self)


def create_icu_classifier(pretrained_encoder_path: Optional[str] = None,
                         freeze_encoder: bool = False,
                         **kwargs) -> ICUClassifier:
    """
    Factory function to create ICU classifier.
    
    Args:
        pretrained_encoder_path: Path to pretrained encoder weights
        freeze_encoder: Whether to freeze encoder parameters
        **kwargs: Additional arguments for ICUClassifier
        
    Returns:
        ICU classifier instance
    """
    # Default configuration
    default_config = {
        'num_classes': 5,
        'hidden_dim': 256,
        'dropout_rate': 0.3,
        'aggregation_method': 'attention'
    }
    
    # Update with provided kwargs
    config = {**default_config, **kwargs}
    
    # Create classifier
    classifier = ICUClassifier(freeze_encoder=freeze_encoder, **config)
    
    # Load pretrained encoder if provided
    if pretrained_encoder_path:
        classifier.load_pretrained_encoder(pretrained_encoder_path)
    
    return classifier


def test_icu_classifier():
    """Test the ICU classifier with sample input."""
    print("Testing ICU Classifier...")
    
    # Create sample input
    batch_size = 2
    seq_length = 16
    height, width = 96, 96
    channels = 1
    
    x = torch.randn(batch_size, seq_length, height, width, channels)
    print(f"Input shape: {x.shape}")
    
    # Create classifier
    classifier = ICUClassifier(
        num_classes=5,
        hidden_dim=256,
        aggregation_method="attention"
    )
    
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
    test_icu_classifier()
