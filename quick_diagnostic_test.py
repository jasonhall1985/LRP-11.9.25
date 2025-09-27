#!/usr/bin/env python3
"""
🔍 QUICK DIAGNOSTIC TEST
========================

Test checkpoint 165 model on preprocessed test set to identify the performance gap.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import glob
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime

# Import the exact model architecture from checkpoint 165
class LightweightCNNLSTM(nn.Module):
    """Exact 3D CNN-LSTM architecture from checkpoint 165"""
    
    def __init__(self):
        super(LightweightCNNLSTM, self).__init__()
        
        # Lightweight 3D CNN feature extractor
        self.conv3d1 = nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=1)
        self.bn3d1 = nn.BatchNorm3d(16)
        self.pool3d1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        
        self.conv3d2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1)
        self.bn3d2 = nn.BatchNorm3d(32)
        self.pool3d2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv3d3 = nn.Conv3d(32, 48, kernel_size=(3, 3, 3), padding=1)
        self.bn3d3 = nn.BatchNorm3d(48)
        self.pool3d3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 6))

        # LSTM for temporal modeling
        self.lstm_input_size = 48 * 4 * 6  # 1152 features per timestep
        self.lstm_hidden_size = 128
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        
        # Classifier head with regularization
        self.dropout1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(self.lstm_hidden_size, 64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        
        self.dropout2 = nn.Dropout(0.3)
        self.fc_out = nn.Linear(64, 4)
    
    def forward(self, x):
        # CNN feature extraction
        # Input: (batch, 1, 32, 64, 96)
        x = torch.relu(self.bn3d1(self.conv3d1(x)))
        x = self.pool3d1(x)
        
        x = torch.relu(self.bn3d2(self.conv3d2(x)))
        x = self.pool3d2(x)
        
        x = torch.relu(self.bn3d3(self.conv3d3(x)))
        x = self.pool3d3(x)
        
        x = self.adaptive_pool(x)
        
        # Reshape for LSTM: (batch, timesteps, features)
        batch_size = x.size(0)
        timesteps = x.size(2)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.contiguous().view(batch_size, timesteps, -1)
        
        # LSTM temporal modeling
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last timestep output
        x = lstm_out[:, -1, :]
        
        # Classification head
        x = self.dropout1(x)
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        
        x = self.dropout2(x)
        x = self.fc_out(x)
        
        return x

def load_model():
    """Load checkpoint 165 model"""
    print("🔄 Loading Checkpoint 165 Model...")
    
    model_path = "./checkpoint_enhanced_81_65_percent_success_20250924/best_lightweight_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return None
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Initialize model
        model = LightweightCNNLSTM()
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            val_acc = checkpoint.get('best_val_acc', 0)
            print(f"✅ Model loaded with validation accuracy: {val_acc:.2f}%")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Model loaded (no validation accuracy info)")
        
        model.eval()
        print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def test_on_preprocessed_data():
    """Test model on preprocessed test set"""
    print("\n🎯 Testing on Preprocessed Test Set")
    print("-" * 50)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Load test data
    test_dir = "preprocessed_test_set_24925"
    test_files = glob.glob(os.path.join(test_dir, "*_preprocessed.npy"))
    
    if len(test_files) == 0:
        print(f"❌ No test files found in {test_dir}")
        return
    
    print(f"📁 Found {len(test_files)} test files")
    
    # Class mapping
    class_mapping = {
        'doctor': 0,
        'i_need_to_move': 1,
        'my_mouth_is_dry': 2,
        'pillow': 3
    }
    
    idx_to_class = {v: k for k, v in class_mapping.items()}
    
    # Process test files
    test_data = []
    test_labels = []
    test_filenames = []
    
    for test_file in test_files:
        filename = os.path.basename(test_file).lower()
        
        # Extract ground truth from filename
        ground_truth = None
        if 'doctor' in filename:
            ground_truth = 'doctor'
        elif 'i_need_to_move' in filename or 'need_to_move' in filename:
            ground_truth = 'i_need_to_move'
        elif 'my_mouth_is_dry' in filename or 'mouth_is_dry' in filename:
            ground_truth = 'my_mouth_is_dry'
        elif 'pillow' in filename:
            ground_truth = 'pillow'
        
        # Skip files that don't match our 4 classes
        if ground_truth and ground_truth in class_mapping:
            try:
                video_data = np.load(test_file)
                test_data.append(video_data)
                test_labels.append(class_mapping[ground_truth])
                test_filenames.append(os.path.basename(test_file))
                print(f"✅ Loaded {ground_truth}: {os.path.basename(test_file)}")
            except Exception as e:
                print(f"❌ Failed to load {test_file}: {e}")
        else:
            print(f"⚠️  Skipped (not in 4-class set): {os.path.basename(test_file)}")
    
    if len(test_data) == 0:
        print("❌ No valid test data found")
        return
    
    print(f"\n📊 Testing {len(test_data)} videos from 4-class set")
    
    # Run predictions
    predictions = []
    confidences = []
    
    device = torch.device('cpu')
    model.to(device)
    
    with torch.no_grad():
        for i, video_data in enumerate(test_data):
            try:
                # Convert to tensor: (1, 1, 32, 64, 96)
                video_tensor = torch.FloatTensor(video_data).unsqueeze(0).unsqueeze(0)
                
                # Get prediction
                outputs = model(video_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predictions.append(predicted.item())
                confidences.append(confidence.item())
                
                # Show prediction
                pred_class = idx_to_class[predicted.item()]
                true_class = idx_to_class[test_labels[i]]
                correct = "✅" if predicted.item() == test_labels[i] else "❌"
                
                print(f"{correct} {test_filenames[i][:30]:<30} | True: {true_class:<15} | Pred: {pred_class:<15} | Conf: {confidence.item():.3f}")
                
            except Exception as e:
                print(f"❌ Error processing {test_filenames[i]}: {e}")
                predictions.append(-1)  # Error marker
                confidences.append(0.0)
    
    # Calculate results
    valid_predictions = [p for p in predictions if p != -1]
    valid_labels = [test_labels[i] for i, p in enumerate(predictions) if p != -1]
    
    if len(valid_predictions) > 0:
        accuracy = accuracy_score(valid_labels, valid_predictions) * 100
        
        print(f"\n📈 RESULTS:")
        print(f"   Accuracy: {accuracy:.2f}%")
        print(f"   Videos tested: {len(valid_predictions)}")
        print(f"   Mean confidence: {np.mean([c for c in confidences if c > 0]):.3f}")
        
        # Detailed classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(valid_labels, valid_predictions, 
                                  target_names=[idx_to_class[i] for i in range(4)]))
        
        # Compare to expected validation accuracy
        expected_val_acc = 81.65
        gap = expected_val_acc - accuracy
        
        print(f"\n🔍 PERFORMANCE GAP ANALYSIS:")
        print(f"   Expected (validation): {expected_val_acc:.2f}%")
        print(f"   Actual (test): {accuracy:.2f}%")
        print(f"   Gap: {gap:.2f} percentage points")
        
        if gap > 40:
            print("   🚨 SEVERE PERFORMANCE DEGRADATION DETECTED!")
        elif gap > 20:
            print("   ⚠️  Significant performance degradation")
        elif gap > 10:
            print("   ⚠️  Moderate performance degradation")
        else:
            print("   ✅ Performance within expected range")
    
    else:
        print("❌ No valid predictions made")

if __name__ == "__main__":
    print("🔍 QUICK DIAGNOSTIC TEST - Checkpoint 165")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_on_preprocessed_data()
    
    print(f"\n✅ Diagnostic completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
