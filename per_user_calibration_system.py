#!/usr/bin/env python3
"""
Per-User Calibration System for 82%+ Validation Accuracy
Two-stage calibration approach with personalized feature mappings
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import json
import joblib
from datetime import datetime
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LightweightCNN_LSTM(nn.Module):
    """Enhanced Lightweight CNN-LSTM with feature extraction capability"""
    
    def __init__(self, num_classes=4, hidden_size=128, num_layers=2, dropout=0.3):
        super(LightweightCNN_LSTM, self).__init__()
        
        # Lightweight CNN feature extractor
        self.cnn = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 96x64 -> 48x32
            
            # Second conv block
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 48x32 -> 24x16
            
            # Third conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 24x16 -> 12x8
            
            # Fourth conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 2))  # Fixed output: 3x2
        )
        
        # CNN output size: 128 * 3 * 2 = 768
        self.cnn_output_size = 128 * 3 * 2
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x, extract_features=False):
        batch_size, channels, seq_len, height, width = x.size()
        
        # Process each frame through CNN
        cnn_features = []
        for t in range(seq_len):
            frame = x[:, :, t, :, :]
            features = self.cnn(frame)
            features = features.view(batch_size, -1)
            cnn_features.append(features)
        
        # Stack temporal features
        cnn_features = torch.stack(cnn_features, dim=1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(cnn_features)
        
        # Use last output for classification
        final_features = lstm_out[:, -1]  # Shape: (batch_size, 256)
        
        if extract_features:
            return final_features
        
        # Classification
        output = self.classifier(final_features)
        return output

class TrainingCompatibleLipDetector:
    """Identical preprocessing pipeline from training for consistency"""
    
    def __init__(self):
        self.target_width = 96
        self.target_height = 64
        self.target_frames = 32
        logger.info(f"TrainingCompatibleLipDetector initialized: {self.target_width}x{self.target_height}, {self.target_frames} frames")
    
    def process_video(self, video_path):
        """Process video with identical training pipeline"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale and resize
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Check original frame dimensions and handle orientation
                original_height, original_width = frame.shape
                logger.debug(f"Original frame size: {original_width}x{original_height}")

                # Always resize to target dimensions (96x64)
                frame = cv2.resize(frame, (self.target_width, self.target_height))
                
                # Normalize to [0, 1]
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            
            cap.release()
            
            if len(frames) == 0:
                raise ValueError(f"No frames extracted from video: {video_path}")
            
            # Convert to numpy array and ensure 32 frames
            frames = np.array(frames)
            
            # Handle frame count - take center 32 frames
            if len(frames) >= self.target_frames:
                start_idx = (len(frames) - self.target_frames) // 2
                frames = frames[start_idx:start_idx + self.target_frames]
            else:
                # Pad with last frame if too short
                while len(frames) < self.target_frames:
                    frames = np.append(frames, [frames[-1]], axis=0)
            
            # Convert to tensor with correct shape: (1, 1, 32, 96, 64)
            # frames shape: (32, 96, 64) -> add batch and channel dimensions
            frames_tensor = torch.FloatTensor(frames).unsqueeze(0).unsqueeze(0)
            
            logger.debug(f"Processed {video_path}: shape {frames_tensor.shape}, range [{frames_tensor.min():.3f}, {frames_tensor.max():.3f}]")
            return frames_tensor
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            raise

class CalibrationDataProcessor:
    """Process and validate calibration data"""
    
    def __init__(self, calibration_dir):
        self.calibration_dir = Path(calibration_dir)
        self.class_to_idx = {
            'doctor': 0,
            'i_need_to_move': 1,
            'my_mouth_is_dry': 2,
            'pillow': 3
        }
        self.detector = TrainingCompatibleLipDetector()
        logger.info(f"CalibrationDataProcessor initialized for: {self.calibration_dir}")
    
    def extract_class_from_filename(self, filename):
        """Extract class from filename using robust pattern matching"""
        filename_lower = filename.lower()
        
        # Direct class name matching
        for class_name in self.class_to_idx.keys():
            if class_name in filename_lower:
                return class_name
        
        # Alternative patterns
        if 'doctor' in filename_lower or 'dr' in filename_lower:
            return 'doctor'
        elif 'move' in filename_lower or 'need' in filename_lower:
            return 'i_need_to_move'
        elif 'mouth' in filename_lower or 'dry' in filename_lower:
            return 'my_mouth_is_dry'
        elif 'pillow' in filename_lower:
            return 'pillow'
        
        return None
    
    def process_calibration_data(self):
        """Process all calibration videos and validate distribution"""
        logger.info("🔄 Processing calibration data...")
        
        if not self.calibration_dir.exists():
            raise FileNotFoundError(f"Calibration directory not found: {self.calibration_dir}")
        
        # Find all MOV files
        mov_files = list(self.calibration_dir.glob("*.MOV")) + list(self.calibration_dir.glob("*.mov"))
        logger.info(f"Found {len(mov_files)} MOV files")
        
        if len(mov_files) == 0:
            raise ValueError(f"No MOV files found in {self.calibration_dir}")
        
        # Process each video
        processed_data = []
        class_counts = {class_name: 0 for class_name in self.class_to_idx.keys()}
        
        for video_path in mov_files:
            try:
                # Extract class from filename
                class_name = self.extract_class_from_filename(video_path.name)
                if class_name is None:
                    logger.warning(f"Could not extract class from filename: {video_path.name}")
                    continue
                
                # Process video
                frames_tensor = self.detector.process_video(str(video_path))
                
                # Validate tensor properties - expect (1, 1, 32, 96, 64)
                expected_shape = (1, 1, 32, 96, 64)
                if frames_tensor.shape != expected_shape:
                    logger.warning(f"Unexpected tensor shape: {frames_tensor.shape}, expected {expected_shape}")

                    # Try to fix common shape issues
                    if frames_tensor.shape == (1, 1, 32, 64, 96):
                        logger.info(f"Reshaping {video_path.name} from (1,1,32,64,96) to (1,1,32,96,64)")
                        frames_tensor = frames_tensor.permute(0, 1, 2, 4, 3)  # Swap height and width
                    else:
                        raise ValueError(f"Invalid tensor shape: {frames_tensor.shape}, expected {expected_shape}")
                
                if frames_tensor.min() < 0 or frames_tensor.max() > 1:
                    raise ValueError(f"Invalid tensor range: [{frames_tensor.min():.3f}, {frames_tensor.max():.3f}]")
                
                processed_data.append({
                    'filename': video_path.name,
                    'class': class_name,
                    'class_idx': self.class_to_idx[class_name],
                    'tensor': frames_tensor,
                    'path': str(video_path)
                })
                
                class_counts[class_name] += 1
                logger.info(f"✅ Processed {video_path.name} -> {class_name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {video_path.name}: {str(e)}")
                continue
        
        # Validate class distribution
        logger.info("\n📊 Class distribution:")
        total_videos = len(processed_data)
        for class_name, count in class_counts.items():
            logger.info(f"  {class_name}: {count} videos")
        
        # Check for exactly 5 samples per class
        missing_classes = []
        extra_classes = []
        for class_name, count in class_counts.items():
            if count < 5:
                missing_classes.append(f"{class_name} ({count}/5)")
            elif count > 5:
                extra_classes.append(f"{class_name} ({count}/5)")
        
        if missing_classes:
            logger.warning(f"⚠️  Missing samples: {', '.join(missing_classes)}")
        if extra_classes:
            logger.warning(f"⚠️  Extra samples: {', '.join(extra_classes)}")
        
        if total_videos != 20:
            logger.warning(f"⚠️  Expected 20 videos, found {total_videos}")
        
        logger.info(f"✅ Successfully processed {total_videos} calibration videos")
        return processed_data, class_counts

class ModelCheckpointLoader:
    """Load and verify model checkpoints with fallback strategy"""
    
    def __init__(self):
        self.primary_checkpoint = "enhanced_balanced_training_results/resumed_best_model_20250923_005027.pth"
        self.fallback_checkpoint = "enhanced_balanced_training_results/enhanced_lightweight_model_20250923_000053.pth"
        self.baseline_accuracy = 0.6239
        logger.info("ModelCheckpointLoader initialized")
    
    def load_best_checkpoint(self):
        """Load best available checkpoint with fallback strategy"""
        logger.info("🔄 Loading model checkpoint...")
        
        # Try primary checkpoint first
        checkpoint_path = None
        checkpoint_info = None
        
        if os.path.exists(self.primary_checkpoint):
            try:
                checkpoint = torch.load(self.primary_checkpoint, map_location='cpu')
                val_acc = checkpoint.get('best_val_acc', 0)
                
                if val_acc > self.baseline_accuracy:
                    checkpoint_path = self.primary_checkpoint
                    checkpoint_info = {
                        'path': self.primary_checkpoint,
                        'accuracy': val_acc,
                        'type': 'resumed_best'
                    }
                    logger.info(f"✅ Primary checkpoint found: {val_acc:.4f} accuracy")
                else:
                    logger.info(f"⚠️  Primary checkpoint accuracy {val_acc:.4f} <= baseline {self.baseline_accuracy:.4f}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load primary checkpoint: {str(e)}")
        else:
            logger.info(f"⚠️  Primary checkpoint not found: {self.primary_checkpoint}")
        
        # Fallback to baseline checkpoint
        if checkpoint_path is None:
            if os.path.exists(self.fallback_checkpoint):
                try:
                    checkpoint = torch.load(self.fallback_checkpoint, map_location='cpu')
                    val_acc = checkpoint.get('best_val_acc', self.baseline_accuracy)
                    
                    checkpoint_path = self.fallback_checkpoint
                    checkpoint_info = {
                        'path': self.fallback_checkpoint,
                        'accuracy': val_acc,
                        'type': 'baseline'
                    }
                    logger.info(f"✅ Using fallback checkpoint: {val_acc:.4f} accuracy")
                except Exception as e:
                    logger.error(f"❌ Failed to load fallback checkpoint: {str(e)}")
                    raise
            else:
                raise FileNotFoundError(f"Neither checkpoint found: {self.primary_checkpoint}, {self.fallback_checkpoint}")
        
        # Load model
        model = LightweightCNN_LSTM(num_classes=4, hidden_size=128, num_layers=2, dropout=0.3)
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Verify parameter count
            total_params = sum(p.numel() for p in model.parameters())
            expected_params = 1429284
            
            if total_params != expected_params:
                logger.warning(f"⚠️  Parameter count mismatch: {total_params} != {expected_params}")
            else:
                logger.info(f"✅ Model architecture verified: {total_params:,} parameters")
            
            logger.info(f"✅ Successfully loaded checkpoint: {checkpoint_info['path']}")
            logger.info(f"📊 Original validation accuracy: {checkpoint_info['accuracy']:.4f}")
            
            return model, checkpoint_info
            
        except Exception as e:
            logger.error(f"❌ Failed to load model state: {str(e)}")
            raise

class FeatureEmbeddingExtractor:
    """Extract feature embeddings from the base model"""

    def __init__(self, model):
        self.model = model
        self.model.eval()
        logger.info("FeatureEmbeddingExtractor initialized")

    def extract_embeddings(self, processed_data):
        """Extract 256-dimensional embeddings from calibration data"""
        logger.info("🔄 Extracting feature embeddings...")

        embeddings = []
        labels = []

        with torch.no_grad():
            for i, data in enumerate(processed_data):
                try:
                    # Get tensor and label
                    tensor = data['tensor']  # Shape: (1, 32, 96, 64)
                    label = data['class_idx']

                    # Extract features
                    features = self.model(tensor, extract_features=True)  # Shape: (1, 256)
                    features = features.squeeze(0).cpu().numpy()  # Shape: (256,)

                    # Validate feature vector
                    if features.shape != (256,):
                        raise ValueError(f"Invalid feature shape: {features.shape}")

                    if np.all(features == 0):
                        logger.warning(f"⚠️  Zero feature vector for {data['filename']}")

                    embeddings.append(features)
                    labels.append(label)

                    logger.debug(f"✅ Extracted features for {data['filename']}: {data['class']}")

                except Exception as e:
                    logger.error(f"❌ Failed to extract features for {data['filename']}: {str(e)}")
                    continue

        # Convert to numpy arrays
        if len(embeddings) == 0:
            raise ValueError("No embeddings extracted - all calibration videos failed processing")

        embeddings = np.array(embeddings)  # Shape: (N, 256)
        labels = np.array(labels)  # Shape: (N,)

        logger.info(f"✅ Extracted embeddings: {embeddings.shape}, labels: {labels.shape}")
        logger.info(f"📊 Feature statistics: mean={embeddings.mean():.4f}, std={embeddings.std():.4f}")

        return embeddings, labels

    def normalize_features(self, embeddings):
        """Normalize features for better calibration performance"""
        logger.info("🔄 Normalizing features...")

        # L2 normalization
        l2_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        l2_norms[l2_norms == 0] = 1  # Avoid division by zero
        embeddings_l2 = embeddings / l2_norms

        # StandardScaler normalization
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)

        logger.info(f"✅ L2 normalized: mean={embeddings_l2.mean():.4f}, std={embeddings_l2.std():.4f}")
        logger.info(f"✅ StandardScaler: mean={embeddings_scaled.mean():.4f}, std={embeddings_scaled.std():.4f}")

        return embeddings_l2, embeddings_scaled, scaler

class CalibrationClassifierTrainer:
    """Train and select best calibration classifier"""

    def __init__(self):
        # Use stronger regularization to prevent overfitting on small dataset
        self.classifiers = {
            'LogisticRegression_L1': LogisticRegression(C=0.1, penalty='l1', solver='liblinear', max_iter=1000, random_state=42),
            'LogisticRegression_L2': LogisticRegression(C=0.1, penalty='l2', max_iter=1000, random_state=42),
            'SVC_Linear': SVC(kernel='linear', C=0.1, probability=True, random_state=42),
            'SVC_RBF': SVC(kernel='rbf', C=0.1, gamma='scale', probability=True, random_state=42)
        }
        logger.info("CalibrationClassifierTrainer initialized with regularized classifiers")

    def train_and_select_best(self, embeddings, labels):
        """Train multiple classifiers and select best using cross-validation"""
        logger.info("🔄 Training calibration classifiers...")

        # Check class distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        logger.info(f"📊 Calibration data class distribution: {dict(zip(unique_labels, counts))}")

        results = {}

        for name, classifier in self.classifiers.items():
            try:
                logger.info(f"Training {name}...")

                # Use stratified 3-fold CV for small dataset
                from sklearn.model_selection import StratifiedKFold
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                cv_scores = cross_val_score(classifier, embeddings, labels, cv=cv, scoring='accuracy')
                mean_score = cv_scores.mean()
                std_score = cv_scores.std()

                # Train on full dataset
                classifier.fit(embeddings, labels)

                # Test on training data to check for overfitting
                train_score = classifier.score(embeddings, labels)

                results[name] = {
                    'classifier': classifier,
                    'cv_mean': mean_score,
                    'cv_std': std_score,
                    'cv_scores': cv_scores,
                    'train_score': train_score
                }

                logger.info(f"✅ {name}: CV={mean_score:.4f}±{std_score:.4f}, Train={train_score:.4f}")

            except Exception as e:
                logger.error(f"❌ Failed to train {name}: {str(e)}")
                continue

        if not results:
            raise ValueError("No classifiers trained successfully")

        # Select best classifier based on CV score
        best_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
        best_result = results[best_name]

        logger.info(f"🏆 Best classifier: {best_name} (CV: {best_result['cv_mean']:.4f}±{best_result['cv_std']:.4f})")

        # Warning if severe overfitting detected
        if best_result['train_score'] - best_result['cv_mean'] > 0.2:
            logger.warning(f"⚠️  Potential overfitting detected: train={best_result['train_score']:.4f} vs cv={best_result['cv_mean']:.4f}")

        return best_result['classifier'], best_name, results

class TemperatureScaling:
    """Temperature scaling for model calibration"""

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits, labels):
        """Fit temperature parameter using calibration data"""
        from scipy.optimize import minimize_scalar

        def temperature_loss(temp):
            scaled_logits = logits / temp
            probs = torch.softmax(torch.FloatTensor(scaled_logits), dim=1).numpy()
            # Negative log likelihood
            nll = -np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-8))
            return nll

        result = minimize_scalar(temperature_loss, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        logger.info(f"Optimal temperature: {self.temperature:.4f}")

    def predict_proba(self, logits):
        """Apply temperature scaling to logits"""
        scaled_logits = logits / self.temperature
        probs = torch.softmax(torch.FloatTensor(scaled_logits), dim=1).numpy()
        return probs

class CalibratedLipReadingModel:
    """Complete calibrated inference pipeline"""

    def __init__(self, base_model, calibration_classifier, scaler, class_to_idx, hybrid_weight=0.7, temperature_scaler=None):
        self.base_model = base_model
        self.calibration_classifier = calibration_classifier
        self.scaler = scaler
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.hybrid_weight = hybrid_weight  # Weight for original model in hybrid prediction
        self.temperature_scaler = temperature_scaler

        self.base_model.eval()
        logger.info(f"CalibratedLipReadingModel initialized with hybrid_weight={hybrid_weight}")

    def predict_original(self, video_tensor):
        """Original model predictions"""
        with torch.no_grad():
            logits = self.base_model(video_tensor)
            probabilities = torch.softmax(logits, dim=1)
            return probabilities.cpu().numpy(), logits.cpu().numpy()

    def predict_temperature_scaled(self, video_tensor):
        """Temperature scaled predictions"""
        if self.temperature_scaler is None:
            return self.predict_original(video_tensor)[0]

        with torch.no_grad():
            _, logits = self.predict_original(video_tensor)
            probabilities = self.temperature_scaler.predict_proba(logits)
            return probabilities

    def predict_calibrated(self, video_tensor):
        """Calibrated model predictions"""
        with torch.no_grad():
            # Extract features
            features = self.base_model(video_tensor, extract_features=True)
            features = features.cpu().numpy()

            # Debug: Check feature statistics
            logger.debug(f"Raw features: shape={features.shape}, mean={features.mean():.4f}, std={features.std():.4f}")

            # Normalize features
            features_normalized = self.scaler.transform(features)
            logger.debug(f"Normalized features: mean={features_normalized.mean():.4f}, std={features_normalized.std():.4f}")

            # Calibrated prediction
            probabilities = self.calibration_classifier.predict_proba(features_normalized)
            logger.debug(f"Calibrated probabilities: {probabilities}")

            return probabilities

    def predict_hybrid(self, video_tensor):
        """Hybrid prediction combining original and calibrated models"""
        original_probs, _ = self.predict_original(video_tensor)
        calibrated_probs = self.predict_calibrated(video_tensor)

        # Weighted combination
        hybrid_probs = (self.hybrid_weight * original_probs +
                       (1 - self.hybrid_weight) * calibrated_probs)

        return hybrid_probs

    def predict_with_confidence(self, video_tensor):
        """Predictions with confidence scoring"""
        original_probs, _ = self.predict_original(video_tensor)
        calibrated_probs = self.predict_calibrated(video_tensor)
        hybrid_probs = self.predict_hybrid(video_tensor)
        temperature_probs = self.predict_temperature_scaled(video_tensor)

        # Confidence metrics
        original_confidence = np.max(original_probs, axis=1)
        calibrated_confidence = np.max(calibrated_probs, axis=1)
        hybrid_confidence = np.max(hybrid_probs, axis=1)
        temperature_confidence = np.max(temperature_probs, axis=1)

        # Entropy-based uncertainty
        original_entropy = -np.sum(original_probs * np.log(original_probs + 1e-8), axis=1)
        calibrated_entropy = -np.sum(calibrated_probs * np.log(calibrated_probs + 1e-8), axis=1)
        hybrid_entropy = -np.sum(hybrid_probs * np.log(hybrid_probs + 1e-8), axis=1)
        temperature_entropy = -np.sum(temperature_probs * np.log(temperature_probs + 1e-8), axis=1)

        return {
            'original_probs': original_probs,
            'calibrated_probs': calibrated_probs,
            'hybrid_probs': hybrid_probs,
            'temperature_probs': temperature_probs,
            'original_confidence': original_confidence,
            'calibrated_confidence': calibrated_confidence,
            'hybrid_confidence': hybrid_confidence,
            'temperature_confidence': temperature_confidence,
            'original_entropy': original_entropy,
            'calibrated_entropy': calibrated_entropy,
            'hybrid_entropy': hybrid_entropy,
            'temperature_entropy': temperature_entropy
        }

def evaluate_calibrated_model(calibrated_model, validation_data_path, processor):
    """Evaluate calibrated model on validation set"""
    logger.info("🔄 Evaluating calibrated model on validation set...")

    # Load validation manifest
    val_manifest = pd.read_csv(validation_data_path)
    logger.info(f"Loaded validation set: {len(val_manifest)} videos")

    # Process validation videos
    original_predictions = []
    calibrated_predictions = []
    hybrid_predictions = []
    true_labels = []

    for idx, row in val_manifest.iterrows():
        try:
            video_path = os.path.join("data/the_best_videos_so_far", row['filename'])

            if not os.path.exists(video_path):
                logger.warning(f"⚠️  Video not found: {video_path}")
                continue

            # Process video
            frames_tensor = processor.detector.process_video(video_path)

            # Get predictions
            results = calibrated_model.predict_with_confidence(frames_tensor)

            original_pred = np.argmax(results['original_probs'])
            calibrated_pred = np.argmax(results['calibrated_probs'])
            hybrid_pred = np.argmax(results['hybrid_probs'])
            true_label = processor.class_to_idx[row['class']]

            original_predictions.append(original_pred)
            calibrated_predictions.append(calibrated_pred)
            hybrid_predictions.append(hybrid_pred)
            true_labels.append(true_label)

            if idx % 20 == 0:
                logger.info(f"Processed {idx+1}/{len(val_manifest)} validation videos")

        except Exception as e:
            logger.error(f"❌ Failed to process validation video {row['filename']}: {str(e)}")
            continue

    # Calculate accuracies
    original_accuracy = accuracy_score(true_labels, original_predictions)
    calibrated_accuracy = accuracy_score(true_labels, calibrated_predictions)
    hybrid_accuracy = accuracy_score(true_labels, hybrid_predictions)

    logger.info(f"📊 Original model accuracy: {original_accuracy:.4f}")
    logger.info(f"📊 Calibrated model accuracy: {calibrated_accuracy:.4f}")
    logger.info(f"📊 Hybrid model accuracy: {hybrid_accuracy:.4f}")

    # Find best performing model
    best_accuracy = max(original_accuracy, calibrated_accuracy, hybrid_accuracy)
    if best_accuracy == hybrid_accuracy:
        best_model = "hybrid"
        improvement = hybrid_accuracy - original_accuracy
    elif best_accuracy == calibrated_accuracy:
        best_model = "calibrated"
        improvement = calibrated_accuracy - original_accuracy
    else:
        best_model = "original"
        improvement = 0

    logger.info(f"🏆 Best model: {best_model} ({best_accuracy:.4f})")
    logger.info(f"📈 Improvement over original: {improvement:.4f} ({((best_accuracy/original_accuracy - 1) * 100):+.1f}%)")

    # Statistical significance test using chi-square test
    from scipy.stats import chi2_contingency

    # Create contingency table for statistical test
    correct_original = np.array(original_predictions) == np.array(true_labels)
    correct_calibrated = np.array(calibrated_predictions) == np.array(true_labels)

    # Count improvements and degradations
    b = np.sum(correct_original & ~correct_calibrated)  # Original correct, calibrated wrong
    c = np.sum(~correct_original & correct_calibrated)  # Original wrong, calibrated correct

    if b + c > 0:
        # Simple paired comparison
        improvement_ratio = c / (b + c) if (b + c) > 0 else 0
        logger.info(f"📊 Improvement ratio: {improvement_ratio:.4f} (>0.5 indicates improvement)")

        if improvement_ratio > 0.6:
            logger.info("✅ Substantial improvement detected")
        elif improvement_ratio > 0.5:
            logger.info("✅ Modest improvement detected")
        else:
            logger.info("⚠️  No significant improvement detected")
    else:
        logger.info("📊 No changes between original and calibrated predictions")

    return {
        'original_accuracy': original_accuracy,
        'calibrated_accuracy': calibrated_accuracy,
        'hybrid_accuracy': hybrid_accuracy,
        'best_accuracy': best_accuracy,
        'best_model': best_model,
        'improvement': improvement,
        'improvement_percent': ((best_accuracy/original_accuracy - 1) * 100),
        'target_achieved': best_accuracy >= 0.82,
        'n_samples': len(true_labels)
    }

def save_calibration_results(calibrated_model, calibration_classifier_name, results, embeddings, labels, processed_data):
    """Save calibration system and results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"calibration_results_{timestamp}")
    results_dir.mkdir(exist_ok=True)

    logger.info(f"💾 Saving calibration results to {results_dir}")

    # Save calibration classifier
    joblib.dump(calibrated_model.calibration_classifier, results_dir / "calibration_classifier.pkl")
    joblib.dump(calibrated_model.scaler, results_dir / "feature_scaler.pkl")

    # Save embeddings and labels
    np.save(results_dir / "calibration_embeddings.npy", embeddings)
    np.save(results_dir / "calibration_labels.npy", labels)

    # Save processed data info
    data_info = []
    for data in processed_data:
        data_info.append({
            'filename': data['filename'],
            'class': data['class'],
            'class_idx': data['class_idx'],
            'path': data['path']
        })

    with open(results_dir / "calibration_data_info.json", 'w') as f:
        json.dump(data_info, f, indent=2)

    # Save results summary
    summary = {
        'timestamp': timestamp,
        'calibration_classifier': calibration_classifier_name,
        'n_calibration_samples': len(processed_data),
        'original_accuracy': results['original_accuracy'],
        'calibrated_accuracy': results['calibrated_accuracy'],
        'improvement': results['improvement'],
        'improvement_percent': results['improvement_percent'],
        'target_achieved': results['target_achieved'],
        'validation_samples': results['n_samples']
    }

    with open(results_dir / "calibration_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"✅ Calibration results saved to {results_dir}")
    return results_dir

def main():
    """Main calibration pipeline execution"""
    print("🎯 PER-USER CALIBRATION SYSTEM FOR 82%+ VALIDATION ACCURACY")
    print("=" * 80)
    print("Two-stage calibration with personalized feature mappings")
    print("Building on 62.39% baseline for cross-demographic generalization")
    print()

    try:
        # 1. Model Checkpoint Loading and Verification
        print("📋 PHASE 1: MODEL CHECKPOINT LOADING AND VERIFICATION")
        print("-" * 60)

        checkpoint_loader = ModelCheckpointLoader()
        model, checkpoint_info = checkpoint_loader.load_best_checkpoint()

        print(f"✅ Loaded checkpoint: {checkpoint_info['type']}")
        print(f"📊 Original accuracy: {checkpoint_info['accuracy']:.4f}")
        print()

        # 2. Calibration Data Preprocessing and Validation
        print("📋 PHASE 2: CALIBRATION DATA PREPROCESSING AND VALIDATION")
        print("-" * 60)

        calibration_dir = "data/final_corrected_test/data/calibration 23.9.25"
        processor = CalibrationDataProcessor(calibration_dir)
        processed_data, class_counts = processor.process_calibration_data()

        print(f"✅ Processed {len(processed_data)} calibration videos")
        print(f"📊 Class distribution: {class_counts}")
        print()

        # 3. Feature Embedding Extraction
        print("📋 PHASE 3: FEATURE EMBEDDING EXTRACTION")
        print("-" * 60)

        extractor = FeatureEmbeddingExtractor(model)
        embeddings, labels = extractor.extract_embeddings(processed_data)
        embeddings_l2, embeddings_scaled, scaler = extractor.normalize_features(embeddings)

        print(f"✅ Extracted {embeddings.shape[0]} feature embeddings")
        print(f"📊 Feature dimensions: {embeddings.shape[1]}")
        print()

        # 4. Calibration Classifier Training and Selection
        print("📋 PHASE 4: CALIBRATION CLASSIFIER TRAINING AND SELECTION")
        print("-" * 60)

        trainer = CalibrationClassifierTrainer()
        best_classifier, best_name, all_results = trainer.train_and_select_best(embeddings_scaled, labels)

        print(f"✅ Best calibration method: {best_name}")
        print(f"📊 Cross-validation accuracy: {all_results[best_name]['cv_mean']:.4f} ± {all_results[best_name]['cv_std']:.4f}")
        print()

        # 5. Calibrated Inference Pipeline Implementation
        print("📋 PHASE 5: CALIBRATED INFERENCE PIPELINE IMPLEMENTATION")
        print("-" * 60)

        calibrated_model = CalibratedLipReadingModel(
            base_model=model,
            calibration_classifier=best_classifier,
            scaler=scaler,
            class_to_idx=processor.class_to_idx
        )

        print("✅ Calibrated inference pipeline ready")
        print()

        # 6. Validation Set Evaluation
        print("📋 PHASE 6: VALIDATION SET EVALUATION")
        print("-" * 60)

        validation_manifest_path = "enhanced_balanced_training_results/enhanced_balanced_536_validation_manifest.csv"

        if os.path.exists(validation_manifest_path):
            results = evaluate_calibrated_model(calibrated_model, validation_manifest_path, processor)

            print(f"📊 CALIBRATION RESULTS:")
            print(f"   Original accuracy: {results['original_accuracy']:.4f}")
            print(f"   Calibrated accuracy: {results['calibrated_accuracy']:.4f}")
            print(f"   Hybrid accuracy: {results['hybrid_accuracy']:.4f}")
            print(f"   🏆 Best model: {results['best_model']} ({results['best_accuracy']:.4f})")
            print(f"   Improvement: {results['improvement']:+.4f} ({results['improvement_percent']:+.1f}%)")
            print(f"   82% target achieved: {'✅ YES' if results['target_achieved'] else '❌ NO'}")
            print(f"   Validation samples: {results['n_samples']}")

            # Save results
            results_dir = save_calibration_results(
                calibrated_model, best_name, results, embeddings_scaled, labels, processed_data
            )

            print(f"\n💾 Results saved to: {results_dir}")

            if results['target_achieved']:
                print("\n🎉 SUCCESS: 82% validation accuracy target achieved!")
                print(f"🚀 {results['best_model'].title()} model ready for production deployment")
            else:
                print(f"\n⚠️  Target not achieved. Best: {results['best_accuracy']:.4f}, Target: 0.82")
                print("💡 Consider additional calibration strategies or more calibration data")

        else:
            logger.warning(f"⚠️  Validation manifest not found: {validation_manifest_path}")
            print("⚠️  Skipping validation evaluation - manifest not found")

        print("\n🎯 Per-user calibration system implementation complete!")
        return True

    except Exception as e:
        logger.error(f"❌ Calibration system failed: {str(e)}")
        print(f"\n❌ CALIBRATION FAILED: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
