#!/usr/bin/env python3
"""
🔄 LOAD 75.9% CHECKPOINT
Simple script to load the restored 75.9% validation accuracy model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

class DoctorFocusedModel(nn.Module):
    """
    EXACT architecture from the 75.9% validation accuracy checkpoint.
    2.98M parameters, proven 4-class lip-reading model.
    """
    
    def __init__(self):
        super(DoctorFocusedModel, self).__init__()

        # 3D CNN layers
        self.conv3d1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=1)
        self.bn3d1 = nn.BatchNorm3d(32)
        self.pool3d1 = nn.MaxPool3d(kernel_size=(1, 2, 2))  # Spatial only

        self.conv3d2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1)
        self.bn3d2 = nn.BatchNorm3d(64)
        self.pool3d2 = nn.MaxPool3d(kernel_size=(2, 2, 2))  # Temporal + spatial

        self.conv3d3 = nn.Conv3d(64, 96, kernel_size=(3, 3, 3), padding=1)
        self.bn3d3 = nn.BatchNorm3d(96)
        self.pool3d3 = nn.MaxPool3d(kernel_size=(2, 2, 2))  # Temporal + spatial

        self.conv3d4 = nn.Conv3d(96, 128, kernel_size=(3, 3, 3), padding=1)
        self.bn3d4 = nn.BatchNorm3d(128)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((3, 3, 4))  # Adaptive pooling

        # Feature size: 128 * 3 * 3 * 4 = 4,608
        self.feature_size = 128 * 3 * 3 * 4

        # Fully connected layers
        self.dropout1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)

        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)

        self.dropout3 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 32)

        # 4-class output
        self.fc_out = nn.Linear(32, 4)

    def forward(self, x):
        # 3D CNN feature extraction
        x = F.relu(self.bn3d1(self.conv3d1(x)))
        x = self.pool3d1(x)

        x = F.relu(self.bn3d2(self.conv3d2(x)))
        x = self.pool3d2(x)

        x = F.relu(self.bn3d3(self.conv3d3(x)))
        x = self.pool3d3(x)

        x = F.relu(self.bn3d4(self.conv3d4(x)))
        x = self.adaptive_pool(x)

        # Flatten and classify
        x = x.view(x.size(0), -1)

        x = self.dropout1(x)
        x = F.relu(self.bn_fc1(self.fc1(x)))

        x = self.dropout2(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))

        x = self.dropout3(x)
        x = F.relu(self.fc3(x))
        x = self.fc_out(x)

        return x

def load_checkpoint():
    """Load the 75.9% checkpoint and return model ready for use."""
    
    # Class mapping from the checkpoint
    class_to_idx = {
        'my_mouth_is_dry': 0,
        'i_need_to_move': 1,
        'doctor': 2,
        'pillow': 3
    }
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    print("🔄 Loading 75.9% validation accuracy checkpoint...")
    
    # Try to find the checkpoint
    checkpoint_paths = [
        Path("doctor_focused_results/best_doctor_focused_model.pth"),
        Path("backup_75.9_success_20250921_004410/best_doctor_focused_model.pth"),
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if path.exists():
            checkpoint_path = path
            break
    
    if checkpoint_path is None:
        raise FileNotFoundError("Could not find the 75.9% checkpoint file!")
    
    # Load the checkpoint
    device = torch.device('cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create and load the model
    model = DoctorFocusedModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Checkpoint loaded from: {checkpoint_path}")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🎯 Classes: {list(class_to_idx.keys())}")
    
    return model, class_to_idx, idx_to_class, checkpoint

def test_model(model, class_to_idx, idx_to_class):
    """Test the loaded model with dummy input."""
    print("\n🧪 Testing model with dummy input...")
    
    # Create dummy input (batch_size=1, channels=1, frames=32, height=64, width=96)
    dummy_input = torch.randn(1, 1, 32, 64, 96)
    
    with torch.no_grad():
        output = model(dummy_input)
        probabilities = torch.softmax(output, dim=1)
        predicted_class_idx = torch.argmax(output, dim=1).item()
        predicted_class = idx_to_class[predicted_class_idx]
        confidence = probabilities[0, predicted_class_idx].item()
    
    print(f"✅ Model test successful!")
    print(f"   Output shape: {output.shape}")
    print(f"   Predicted class: {predicted_class}")
    print(f"   Confidence: {confidence:.3f}")
    
    return True

if __name__ == "__main__":
    try:
        # Load the checkpoint
        model, class_to_idx, idx_to_class, checkpoint = load_checkpoint()
        
        # Test the model
        test_model(model, class_to_idx, idx_to_class)
        
        print(f"\n🎉 SUCCESS! 75.9% checkpoint is ready for use!")
        print(f"📋 Usage example:")
        print(f"   from load_75_9_checkpoint import load_checkpoint")
        print(f"   model, class_to_idx, idx_to_class, checkpoint = load_checkpoint()")
        print(f"   # Model is now ready for inference or further training")
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
