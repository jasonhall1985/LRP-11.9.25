#!/usr/bin/env python3
"""
🔍 Backup Integrity Verification Script
Verifies that our 75.9% success backup can be successfully loaded and used.
"""

import torch
import torch.nn as nn
import os
import sys
from pathlib import Path

class DoctorFocusedModel(nn.Module):
    """EXACT architecture from our successful 75.9% training"""
    def __init__(self):
        super(DoctorFocusedModel, self).__init__()

        # IDENTICAL architecture from successful 4-class training
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

        # Feature size: 128 * 3 * 3 * 4 = 4,608 (IDENTICAL)
        self.feature_size = 128 * 3 * 3 * 4

        # IDENTICAL fully connected layers
        self.dropout1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)

        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)

        self.dropout3 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 32)

        # 4-class output (same as successful training)
        self.fc_out = nn.Linear(32, 4)

    def forward(self, x):
        import torch.nn.functional as F

        # IDENTICAL forward pass from successful training
        x = F.relu(self.bn3d1(self.conv3d1(x)))
        x = self.pool3d1(x)

        x = F.relu(self.bn3d2(self.conv3d2(x)))
        x = self.pool3d2(x)

        x = F.relu(self.bn3d3(self.conv3d3(x)))
        x = self.pool3d3(x)

        x = F.relu(self.bn3d4(self.conv3d4(x)))
        x = self.adaptive_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.dropout1(x)
        x = F.relu(self.bn_fc1(self.fc1(x)))

        x = self.dropout2(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))

        x = self.dropout3(x)
        x = F.relu(self.fc3(x))

        x = self.fc_out(x)
        return x

def verify_backup_integrity():
    """Verify that our backup can be successfully loaded and used"""
    
    backup_dir = "backup_75.9_success_20250921_004410"
    model_path = os.path.join(backup_dir, "best_doctor_focused_model.pth")
    
    print("🔍 BACKUP INTEGRITY VERIFICATION")
    print("=" * 50)
    
    # Check if backup directory exists
    if not os.path.exists(backup_dir):
        print(f"❌ ERROR: Backup directory not found: {backup_dir}")
        return False
    
    print(f"✅ Backup directory found: {backup_dir}")
    
    # Check critical files
    critical_files = [
        "best_doctor_focused_model.pth",
        "doctor_focused_trainer.py",
        "doctor_focused_report.txt",
        "BACKUP_MANIFEST.md"
    ]
    
    for file in critical_files:
        file_path = os.path.join(backup_dir, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file}: {size:,} bytes")
        else:
            print(f"❌ MISSING: {file}")
            return False
    
    # Test model loading
    print("\n🧠 TESTING MODEL LOADING...")
    try:
        # Create model architecture
        model = DoctorFocusedModel()
        
        # Load the saved model
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model loaded successfully from checkpoint")
            print(f"   - Epoch: {checkpoint.get('epoch', 'N/A')}")
            best_val_acc = checkpoint.get('best_val_acc', 'N/A')
            val_loss = checkpoint.get('val_loss', 'N/A')
            if isinstance(best_val_acc, (int, float)):
                print(f"   - Best Val Acc: {best_val_acc:.1f}%")
            else:
                print(f"   - Best Val Acc: {best_val_acc}")
            if isinstance(val_loss, (int, float)):
                print(f"   - Loss: {val_loss:.4f}")
            else:
                print(f"   - Loss: {val_loss}")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ Model loaded successfully (direct state dict)")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")
        
        # Test forward pass with dummy data
        model.eval()
        dummy_input = torch.randn(1, 1, 32, 64, 48)  # Batch, Channel, Time, Height, Width
        
        with torch.no_grad():
            output = model(dummy_input)
            print(f"   - Output shape: {output.shape}")
            print(f"   - Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        print("✅ Model forward pass successful!")
        
    except Exception as e:
        print(f"❌ ERROR loading model: {str(e)}")
        return False
    
    # Check dataset splits
    dataset_dir = os.path.join(backup_dir, "classifier training 20.9.25")
    if os.path.exists(dataset_dir):
        print(f"\n📊 DATASET SPLITS VERIFIED")
        for split_file in ["demographic_train_manifest.csv", "demographic_validation_manifest.csv"]:
            split_path = os.path.join(dataset_dir, split_file)
            if os.path.exists(split_path):
                with open(split_path, 'r') as f:
                    lines = len(f.readlines()) - 1  # Subtract header
                print(f"   - {split_file}: {lines} videos")
    
    print("\n🎉 BACKUP INTEGRITY VERIFICATION COMPLETE!")
    print("✅ All critical components verified and functional")
    print("✅ Model can be loaded and used for inference")
    print("✅ Ready to proceed with 7-class training")
    
    return True

if __name__ == "__main__":
    success = verify_backup_integrity()
    sys.exit(0 if success else 1)
