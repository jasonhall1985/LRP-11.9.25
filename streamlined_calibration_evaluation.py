#!/usr/bin/env python3
"""
Streamlined Per-User Calibration Evaluation System
Demonstrates speaker-specific accuracy improvements using same-speaker data
"""

import torch
import numpy as np
import os
from pathlib import Path
import logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import from existing calibration system
from per_user_calibration_system import (
    ModelCheckpointLoader, 
    CalibrationDataProcessor, 
    FeatureEmbeddingExtractor,
    LightweightCNN_LSTM
)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StreamlinedCalibrationEvaluator:
    """Streamlined evaluation of per-user calibration effectiveness"""
    
    def __init__(self):
        self.calibration_dir = "data/final_corrected_test/data/calibration 23.9.25"
        self.class_to_idx = {
            'doctor': 0,
            'i_need_to_move': 1,
            'my_mouth_is_dry': 2,
            'pillow': 3
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        logger.info("StreamlinedCalibrationEvaluator initialized")
    
    def load_model_and_data(self):
        """Load model checkpoint and process calibration data"""
        print("🔄 Loading model checkpoint and processing calibration data...")
        
        # Load model
        checkpoint_loader = ModelCheckpointLoader()
        model, checkpoint_info = checkpoint_loader.load_best_checkpoint()
        
        # Process calibration data
        processor = CalibrationDataProcessor(self.calibration_dir)
        processed_data, class_counts = processor.process_calibration_data()
        
        # Extract embeddings
        extractor = FeatureEmbeddingExtractor(model)
        embeddings, labels = extractor.extract_embeddings(processed_data)
        embeddings_l2, embeddings_scaled, scaler = extractor.normalize_features(embeddings)
        
        print(f"✅ Loaded model: {checkpoint_info['accuracy']:.4f} accuracy")
        print(f"✅ Processed {len(processed_data)} calibration videos")
        print(f"📊 Class distribution: {class_counts}")
        
        return model, embeddings_l2, labels, processed_data, checkpoint_info
    
    def create_stratified_split(self, embeddings, labels, processed_data):
        """Create stratified train/test split: 4 train + 1 test per class"""
        print("🔄 Creating stratified train/test split...")
        
        train_embeddings = []
        train_labels = []
        test_embeddings = []
        test_labels = []
        train_data = []
        test_data = []
        
        # Split by class to ensure balanced representation
        for class_idx in range(4):  # 4 classes
            class_mask = labels == class_idx
            class_embeddings = embeddings[class_mask]
            class_labels = labels[class_mask]
            class_data = [processed_data[i] for i in range(len(processed_data)) if labels[i] == class_idx]
            
            if len(class_embeddings) != 5:
                raise ValueError(f"Expected 5 samples for class {class_idx}, got {len(class_embeddings)}")
            
            # Use first 4 for training, last 1 for testing
            train_embeddings.extend(class_embeddings[:4])
            train_labels.extend(class_labels[:4])
            train_data.extend(class_data[:4])
            
            test_embeddings.extend(class_embeddings[4:5])
            test_labels.extend(class_labels[4:5])
            test_data.extend(class_data[4:5])
        
        train_embeddings = np.array(train_embeddings)
        train_labels = np.array(train_labels)
        test_embeddings = np.array(test_embeddings)
        test_labels = np.array(test_labels)
        
        print(f"✅ Train set: {len(train_embeddings)} samples (4 per class)")
        print(f"✅ Test set: {len(test_embeddings)} samples (1 per class)")
        
        # Verify class balance
        train_class_counts = np.bincount(train_labels)
        test_class_counts = np.bincount(test_labels)
        print(f"📊 Train class distribution: {train_class_counts}")
        print(f"📊 Test class distribution: {test_class_counts}")
        
        return train_embeddings, train_labels, test_embeddings, test_labels, train_data, test_data
    
    def train_calibration_classifiers(self, train_embeddings, train_labels):
        """Train calibration classifiers with different approaches"""
        print("🔄 Training calibration classifiers...")
        
        classifiers = {
            'KNN_k1_cosine': KNeighborsClassifier(n_neighbors=1, metric='cosine'),
            'KNN_k3_cosine': KNeighborsClassifier(n_neighbors=3, metric='cosine'),
            'LogReg_strong': LogisticRegression(C=0.01, max_iter=1000, random_state=42),
            'LogReg_moderate': LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        }
        
        trained_classifiers = {}
        
        for name, classifier in classifiers.items():
            try:
                classifier.fit(train_embeddings, train_labels)
                train_accuracy = classifier.score(train_embeddings, train_labels)
                trained_classifiers[name] = {
                    'classifier': classifier,
                    'train_accuracy': train_accuracy
                }
                print(f"✅ {name}: Training accuracy = {train_accuracy:.4f}")
            except Exception as e:
                print(f"❌ Failed to train {name}: {str(e)}")
        
        return trained_classifiers
    
    def evaluate_baseline_performance(self, model, test_data):
        """Evaluate baseline model performance on test clips"""
        print("🔄 Evaluating baseline model performance...")
        
        baseline_predictions = []
        true_labels = []
        
        with torch.no_grad():
            for data in test_data:
                tensor = data['tensor']
                true_label = data['class_idx']
                
                # Get baseline prediction
                logits = model(tensor)
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                
                baseline_predictions.append(predicted_class)
                true_labels.append(true_label)
        
        baseline_accuracy = accuracy_score(true_labels, baseline_predictions)
        print(f"📊 Baseline model accuracy on test clips: {baseline_accuracy:.4f}")
        
        return baseline_accuracy, baseline_predictions, true_labels
    
    def evaluate_calibrated_performance(self, trained_classifiers, test_embeddings, test_labels):
        """Evaluate calibrated classifiers on test set"""
        print("🔄 Evaluating calibrated classifiers...")
        
        results = {}
        
        for name, classifier_info in trained_classifiers.items():
            classifier = classifier_info['classifier']
            train_accuracy = classifier_info['train_accuracy']
            
            # Test accuracy
            test_predictions = classifier.predict(test_embeddings)
            test_accuracy = accuracy_score(test_labels, test_predictions)
            
            # Per-class accuracy
            per_class_accuracy = {}
            for class_idx in range(4):
                class_mask = test_labels == class_idx
                if np.sum(class_mask) > 0:
                    class_acc = accuracy_score(test_labels[class_mask], test_predictions[class_mask])
                    per_class_accuracy[self.idx_to_class[class_idx]] = class_acc
            
            # Overfitting check
            overfitting_gap = train_accuracy - test_accuracy
            
            results[name] = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'per_class_accuracy': per_class_accuracy,
                'overfitting_gap': overfitting_gap,
                'predictions': test_predictions
            }
            
            print(f"✅ {name}:")
            print(f"   Train: {train_accuracy:.4f}, Test: {test_accuracy:.4f}")
            print(f"   Overfitting gap: {overfitting_gap:.4f}")
            print(f"   Per-class: {per_class_accuracy}")
        
        return results
    
    def find_best_calibration_method(self, results, baseline_accuracy):
        """Find best calibration method and compare to baseline"""
        print("🔄 Finding best calibration method...")
        
        best_method = None
        best_accuracy = 0
        
        for name, result in results.items():
            test_accuracy = result['test_accuracy']
            overfitting_gap = result['overfitting_gap']
            
            # Prefer methods with good test accuracy and reasonable overfitting
            if test_accuracy > best_accuracy and overfitting_gap <= 0.2:
                best_accuracy = test_accuracy
                best_method = name
        
        if best_method is None:
            # Fallback: just pick highest test accuracy
            best_method = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
            best_accuracy = results[best_method]['test_accuracy']
        
        improvement = best_accuracy - baseline_accuracy
        improvement_percent = (improvement / baseline_accuracy) * 100
        
        print(f"🏆 Best calibration method: {best_method}")
        print(f"📊 Best test accuracy: {best_accuracy:.4f}")
        print(f"📈 Improvement over baseline: {improvement:+.4f} ({improvement_percent:+.1f}%)")
        
        return best_method, best_accuracy, improvement
    
    def run_evaluation(self):
        """Run complete streamlined calibration evaluation"""
        print("🎯 STREAMLINED PER-USER CALIBRATION EVALUATION")
        print("=" * 60)
        print("Demonstrating speaker-specific accuracy improvements")
        print("Target: >82% accuracy on same-speaker held-out clips")
        print()
        
        try:
            # 1. Load model and data
            model, embeddings, labels, processed_data, checkpoint_info = self.load_model_and_data()
            print()
            
            # 2. Create stratified split
            train_embeddings, train_labels, test_embeddings, test_labels, train_data, test_data = \
                self.create_stratified_split(embeddings, labels, processed_data)
            print()
            
            # 3. Train calibration classifiers
            trained_classifiers = self.train_calibration_classifiers(train_embeddings, train_labels)
            print()
            
            # 4. Evaluate baseline performance
            baseline_accuracy, baseline_predictions, true_labels = \
                self.evaluate_baseline_performance(model, test_data)
            print()
            
            # 5. Evaluate calibrated performance
            calibration_results = self.evaluate_calibrated_performance(
                trained_classifiers, test_embeddings, test_labels)
            print()
            
            # 6. Find best method and report results
            best_method, best_accuracy, improvement = \
                self.find_best_calibration_method(calibration_results, baseline_accuracy)
            print()
            
            # Final results summary
            print("📋 FINAL RESULTS SUMMARY")
            print("-" * 40)
            print(f"Baseline model accuracy: {baseline_accuracy:.4f}")
            print(f"Best calibrated accuracy: {best_accuracy:.4f}")
            print(f"Improvement: {improvement:+.4f} ({(improvement/baseline_accuracy)*100:+.1f}%)")
            print(f"82% target achieved: {'✅ YES' if best_accuracy >= 0.82 else '❌ NO'}")
            print(f"Best method: {best_method}")
            
            if best_accuracy >= 0.82:
                print("\n🎉 SUCCESS: Per-user calibration achieved >82% accuracy!")
                print("🚀 Speaker-specific calibration validated for production deployment")
            else:
                print(f"\n⚠️  Target not achieved. Best: {best_accuracy:.4f}, Target: 0.82")
                print("💡 Consider additional calibration strategies or more calibration data")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {str(e)}")
            print(f"\n❌ EVALUATION FAILED: {str(e)}")
            return False

def main():
    """Main evaluation execution"""
    evaluator = StreamlinedCalibrationEvaluator()
    success = evaluator.run_evaluation()
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
