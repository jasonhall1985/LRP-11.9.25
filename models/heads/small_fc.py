"""
Small fully connected head for efficient classification.
Designed to keep head parameters under 100k.
"""
import torch
import torch.nn as nn

class SmallFCHead(nn.Module):
    """
    Lightweight classification head with <100k parameters.
    
    Args:
        in_ch: Input channel dimension from encoder
        num_classes: Number of output classes
        p: Dropout probability
    """
    def __init__(self, in_ch, num_classes, p=0.6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(128, num_classes)
        )

        # Calculate and verify parameter count
        self._verify_param_count()

    def _verify_param_count(self):
        """Ensure head has <100k parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        if total_params >= 100000:
            raise ValueError(f"Head has {total_params} parameters, exceeds 100k limit")
        print(f"SmallFCHead created with {total_params:,} parameters")

    def forward(self, x):
        """Forward pass through classification layers."""
        return self.net(x)

def create_small_fc_head(in_channels, num_classes, dropout=0.6):
    """Factory function to create SmallFCHead."""
    return SmallFCHead(in_channels, num_classes, dropout)
