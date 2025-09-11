"""
Simple training script for baseline ICU classifier.
Fast training for demonstration and proof of concept.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import logging
from pathlib import Path
from tqdm import tqdm
import time

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

from datasets import create_data_loaders
from models.simple_classifier import create_simple_classifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleTrainer:
    """Simple trainer for baseline classifier."""
    
    def __init__(self):
        """Initialize trainer."""
        self.device = torch.device('cpu')  # Use CPU for fast demo
        
        # Create model
        self.model = create_simple_classifier(
            num_classes=5,
            cnn_channels=(8, 16, 32),  # Smaller model for faster training
            lstm_hidden_size=64,
            lstm_num_layers=1,
            dropout_rate=0.2
        )
        self.model.to(self.device)
        
        # Create data loaders
        self.data_loaders = create_data_loaders(
            data_dir="simple_processed_data",
            batch_size=8,  # Smaller batch size
            num_workers=0,  # No multiprocessing for simplicity
            target_frames=16,
            augment_train=False,  # No augmentation for speed
            load_in_memory=True   # Load in memory for speed
        )
        
        # Create optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Create loss function with class weights
        train_dataset = self.data_loaders['train'].dataset
        class_weights = train_dataset.get_class_weights()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Training state
        self.best_val_f1 = 0.0
        self.best_model_state = None
        
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Training samples: {len(self.data_loaders['train'].dataset)}")
        logger.info(f"Validation samples: {len(self.data_loaders['val'].dataset)}")
        logger.info(f"Test samples: {len(self.data_loaders['test'].dataset)}")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        for batch_idx, (data, labels) in enumerate(tqdm(self.data_loaders['train'], desc="Training")):
            data, labels = data.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.data_loaders['train'])
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in tqdm(self.data_loaders['val'], desc="Validation"):
                data, labels = data.to(self.device), labels.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.data_loaders['val'])
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='macro')
        
        return avg_loss, accuracy, f1
    
    def train(self, epochs: int = 20):
        """Main training loop."""
        logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_f1 = self.validate()
            
            epoch_time = time.time() - start_time
            
            # Log progress
            logger.info(
                f"Epoch {epoch + 1}/{epochs} ({epoch_time:.1f}s) - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}"
            )
            
            # Save best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_model_state = self.model.state_dict().copy()
                logger.info(f"New best validation F1: {val_f1:.4f}")
                
                # Save checkpoint
                os.makedirs("models", exist_ok=True)
                torch.save({
                    'model_state_dict': self.best_model_state,
                    'val_f1': val_f1,
                    'epoch': epoch + 1
                }, "models/simple_classifier_best.pt")
            
            # Early stopping if performance is good
            if val_acc >= 0.8 and val_f1 >= 0.8:
                logger.info(f"Target performance reached! Stopping early.")
                break
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Loaded best model with validation F1: {self.best_val_f1:.4f}")
    
    def evaluate_on_test(self):
        """Evaluate model on test set."""
        logger.info("Evaluating on test set...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in tqdm(self.data_loaders['test'], desc="Testing"):
                data, labels = data.to(self.device), labels.to(self.device)
                
                outputs = self.model(data)
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1_macro = f1_score(all_labels, all_predictions, average='macro')
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
        
        # Get class names
        class_names = self.data_loaders['test'].dataset.classes
        
        # Classification report
        report = classification_report(
            all_labels, all_predictions,
            target_names=class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        results = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'classification_report': report,
            'confusion_matrix': cm,
            'class_names': class_names
        }
        
        return results


def main():
    """Main training function."""
    print("="*60)
    print("ICU LIP READING CLASSIFIER - BASELINE TRAINING")
    print("="*60)
    
    # Create trainer and train
    trainer = SimpleTrainer()
    trainer.train(epochs=15)  # Quick training
    
    # Evaluate on test set
    test_results = trainer.evaluate_on_test()
    
    # Print results
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    print(f"Accuracy: {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.1f}%)")
    print(f"Macro F1: {test_results['f1_macro']:.4f} ({test_results['f1_macro']*100:.1f}%)")
    print(f"Weighted F1: {test_results['f1_weighted']:.4f} ({test_results['f1_weighted']*100:.1f}%)")
    
    # Check if targets are met
    target_acc = 0.80
    target_f1 = 0.80
    
    if test_results['accuracy'] >= target_acc and test_results['f1_macro'] >= target_f1:
        print(f"\n✅ TARGET PERFORMANCE ACHIEVED!")
        print(f"   Accuracy: {test_results['accuracy']:.4f} >= {target_acc:.4f}")
        print(f"   Macro F1: {test_results['f1_macro']:.4f} >= {target_f1:.4f}")
    else:
        print(f"\n⚠️  Target performance not met (but this is expected for a simple baseline):")
        acc_status = "✓" if test_results['accuracy'] >= target_acc else "✗"
        f1_status = "✓" if test_results['f1_macro'] >= target_f1 else "✗"
        print(f"   Accuracy: {test_results['accuracy']:.4f} {acc_status} (target: {target_acc:.4f})")
        print(f"   Macro F1: {test_results['f1_macro']:.4f} {f1_status} (target: {target_f1:.4f})")
    
    print("\nPer-class results:")
    for class_name in test_results['class_names']:
        metrics = test_results['classification_report'][class_name]
        print(f"  {class_name:8s}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    print("\nConfusion Matrix:")
    cm = test_results['confusion_matrix']
    class_names = test_results['class_names']
    
    # Print header
    print("     ", end="")
    for name in class_names:
        print(f"{name[:4]:>4s}", end=" ")
    print()
    
    # Print matrix
    for i, name in enumerate(class_names):
        print(f"{name[:4]:>4s}:", end=" ")
        for j in range(len(class_names)):
            print(f"{cm[i,j]:>3d}", end="  ")
        print()
    
    print("="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
