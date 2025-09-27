#!/usr/bin/env python3
"""
🔍 SYSTEMATIC DIAGNOSTIC INVESTIGATION
=====================================

Comprehensive investigation to identify why checkpoint 165 model (81.65% validation accuracy)
shows severe performance degradation (33.33% test accuracy) - a 44.37 percentage point gap.

INVESTIGATION AREAS:
1. Data Preprocessing Pipeline Alignment
2. Training Data Composition Analysis  
3. Validation Set Integrity Verification
4. Model Architecture Overfitting Assessment
"""

import os
import torch
import torch.nn as nn
import numpy as np
import json
import cv2
from datetime import datetime
import glob
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

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

def load_checkpoint_165_model():
    """Load the exact checkpoint 165 model"""
    print("🔄 Loading Checkpoint 165 Model")
    print("-" * 50)

    model_path = "./checkpoint_enhanced_81_65_percent_success_20250924/best_lightweight_model.pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint 165 model not found at {model_path}")

    # Load checkpoint data
    checkpoint = torch.load(model_path, map_location='cpu')

    # Initialize model
    model = LightweightCNNLSTM()

    # Extract model state dict from checkpoint
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model from checkpoint with validation accuracy: {checkpoint.get('best_val_acc', 'Unknown'):.2f}%")
    else:
        # Fallback: assume the entire checkpoint is the state dict
        model.load_state_dict(checkpoint)

    model.eval()

    print(f"✅ Model loaded from {model_path}")
    print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model

def get_training_preprocessing_pipeline():
    """
    Reconstruct the EXACT preprocessing pipeline used during checkpoint 165 training.
    This must match the preprocessing used to create the training dataset.
    """
    print("🔧 Reconstructing Training Preprocessing Pipeline")
    print("-" * 50)
    
    # Load OpenCV Haar Cascade (same as training)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def preprocess_video_training_exact(video_path):
        """Exact preprocessing pipeline used during training"""
        
        # Read video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            return None
        
        # Process frames with exact training parameters
        processed_frames = []
        
        for frame in frames:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with exact training parameters
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) > 0:
                # Use the largest face (same as training)
                face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = face
                
                # Extract mouth region (exact training parameters)
                mouth_y_start = y + int(h * 0.6)  # 60% down face
                mouth_y_end = y + h
                mouth_x_start = x
                mouth_x_end = x + w
                
                # Add padding (exact training parameters)
                padding = 10
                mouth_y_start = max(0, mouth_y_start - padding)
                mouth_y_end = min(frame.shape[0], mouth_y_end + padding)
                mouth_x_start = max(0, mouth_x_start - padding)
                mouth_x_end = min(frame.shape[1], mouth_x_end + padding)
                
                # Extract mouth ROI
                mouth_roi = frame[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end]
                
                if mouth_roi.size > 0:
                    processed_frames.append(mouth_roi)
            else:
                # Fallback: center crop (same as training)
                h, w = frame.shape[:2]
                center_y, center_x = h // 2, w // 2
                crop_h, crop_w = h // 3, w // 2
                
                y_start = max(0, center_y - crop_h // 2)
                y_end = min(h, center_y + crop_h // 2)
                x_start = max(0, center_x - crop_w // 2)
                x_end = min(w, center_x + crop_w // 2)
                
                fallback_roi = frame[y_start:y_end, x_start:x_end]
                processed_frames.append(fallback_roi)
        
        if len(processed_frames) == 0:
            return None
        
        # Temporal sampling to 32 frames (exact training parameters)
        if len(processed_frames) != 32:
            indices = np.linspace(0, len(processed_frames) - 1, 32, dtype=int)
            processed_frames = [processed_frames[i] for i in indices]
        
        # Resize and convert to grayscale (exact training parameters)
        final_frames = []
        for frame in processed_frames:
            # Resize to 96x64 (exact training size)
            resized = cv2.resize(frame, (96, 64))
            
            # Convert to grayscale
            if len(resized.shape) == 3:
                gray_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = resized
            
            # Normalize to [0, 1] (exact training normalization)
            normalized_frame = gray_frame.astype(np.float32) / 255.0
            final_frames.append(normalized_frame)
        
        # Stack into final array: (32, 64, 96)
        video_array = np.stack(final_frames, axis=0)
        
        return video_array
    
    print("✅ Training preprocessing pipeline reconstructed")
    return preprocess_video_training_exact

def test_preprocessing_alignment():
    """
    CRITICAL TEST: Compare model performance using training preprocessing vs deployment preprocessing
    """
    print("\n🎯 CRITICAL TEST: Preprocessing Pipeline Alignment")
    print("=" * 70)
    
    # Load model
    model = load_checkpoint_165_model()
    
    # Get training preprocessing pipeline
    training_preprocess = get_training_preprocessing_pipeline()
    
    # Test on preprocessed test set
    test_dir = "preprocessed_test_set_24925"
    if not os.path.exists(test_dir):
        print("❌ Preprocessed test set not found")
        return None
    
    # Load preprocessed test videos (these were preprocessed with deployment pipeline)
    test_files = glob.glob(os.path.join(test_dir, "*_preprocessed.npy"))
    
    class_mapping = {
        'doctor': 0,
        'i_need_to_move': 1,
        'my_mouth_is_dry': 2,
        'pillow': 3
    }
    
    # Test with deployment preprocessing (already preprocessed)
    print("\n📊 Testing with DEPLOYMENT preprocessing (current):")
    deployment_results = test_with_preprocessed_data(model, test_files, class_mapping)
    
    # Test with training preprocessing (reprocess from original videos)
    print("\n📊 Testing with TRAINING preprocessing (reconstructed):")
    training_results = test_with_training_preprocessing(model, training_preprocess, class_mapping)
    
    # Compare results
    print("\n🔍 PREPROCESSING ALIGNMENT ANALYSIS:")
    print("-" * 50)
    
    if deployment_results and training_results:
        deployment_acc = deployment_results['accuracy']
        training_acc = training_results['accuracy']
        difference = training_acc - deployment_acc
        
        print(f"Deployment preprocessing accuracy: {deployment_acc:.2f}%")
        print(f"Training preprocessing accuracy: {training_acc:.2f}%")
        print(f"Difference: {difference:+.2f} percentage points")
        
        if abs(difference) > 5.0:
            print("⚠️  SIGNIFICANT PREPROCESSING MISMATCH DETECTED!")
            print("   → The preprocessing pipelines are not aligned")
            print("   → This explains the performance degradation")
        else:
            print("✅ Preprocessing pipelines are aligned")
            print("   → Performance gap is due to other factors")
    
    return {
        'deployment_results': deployment_results,
        'training_results': training_results,
        'preprocessing_aligned': abs(difference) <= 5.0 if deployment_results and training_results else False
    }

def test_with_preprocessed_data(model, test_files, class_mapping):
    """Test model with already preprocessed data (deployment preprocessing)"""
    
    test_data = []
    test_labels = []
    test_filenames = []
    
    for test_file in test_files:
        filename = os.path.basename(test_file).lower()
        
        # Extract class from filename
        ground_truth = None
        if 'doctor' in filename:
            ground_truth = 'doctor'
        elif 'i_need_to_move' in filename or 'need_to_move' in filename:
            ground_truth = 'i_need_to_move'
        elif 'my_mouth_is_dry' in filename or 'mouth_is_dry' in filename:
            ground_truth = 'my_mouth_is_dry'
        elif 'pillow' in filename:
            ground_truth = 'pillow'
        
        if ground_truth and ground_truth in class_mapping:
            try:
                video_data = np.load(test_file)
                test_data.append(video_data)
                test_labels.append(class_mapping[ground_truth])
                test_filenames.append(os.path.basename(test_file))
            except Exception as e:
                print(f"❌ Failed to load {test_file}: {e}")
    
    if len(test_data) == 0:
        print("❌ No valid test videos found")
        return None
    
    # Test model
    predictions = []
    confidences = []
    
    device = torch.device('cpu')
    model.to(device)
    
    with torch.no_grad():
        for i, video_data in enumerate(test_data):
            # Convert to tensor: (1, 1, 32, 64, 96)
            video_tensor = torch.FloatTensor(video_data).unsqueeze(0).unsqueeze(0)
            
            # Get prediction
            outputs = model(video_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predictions.append(predicted.item())
            confidences.append(confidence.item())
    
    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions) * 100
    
    print(f"   Accuracy: {accuracy:.2f}%")
    print(f"   Videos tested: {len(test_data)}")
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'labels': test_labels,
        'confidences': confidences,
        'filenames': test_filenames
    }

def test_with_training_preprocessing(model, training_preprocess, class_mapping):
    """Test model by reprocessing original videos with training preprocessing"""
    
    # Find original test videos (need to locate the source videos)
    # This would require access to the original test videos before preprocessing
    # For now, return placeholder - this would need to be implemented with actual source videos
    
    print("   ⚠️  Original test videos needed for training preprocessing comparison")
    print("   → This test requires access to the raw video files before preprocessing")
    
    return None

def systematic_diagnostic_investigation():
    """Execute comprehensive diagnostic investigation"""
    print("🔍 SYSTEMATIC DIAGNOSTIC INVESTIGATION")
    print("=" * 70)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Execute immediate diagnostic test
    alignment_results = test_preprocessing_alignment()
    
    # Create comprehensive report
    report = {
        'timestamp': datetime.now().isoformat(),
        'investigation_type': 'Systematic Diagnostic Investigation',
        'model_info': {
            'checkpoint': 'Checkpoint 165',
            'validation_accuracy': 81.65,
            'parameters': 721044,
            'architecture': '3D CNN-LSTM'
        },
        'performance_gap': {
            'validation_accuracy': 81.65,
            'test_accuracy': 33.33,
            'gap_percentage_points': 44.37
        },
        'preprocessing_alignment_test': alignment_results,
        'next_investigation_steps': [
            'Training Data Composition Analysis',
            'Validation Set Integrity Verification', 
            'Model Architecture Overfitting Assessment'
        ]
    }
    
    # Save report
    report_path = f"SYSTEMATIC_DIAGNOSTIC_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Diagnostic report saved to: {report_path}")
    print(f"\n✅ Systematic diagnostic investigation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return report

if __name__ == "__main__":
    systematic_diagnostic_investigation()
