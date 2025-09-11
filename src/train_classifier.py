"""
Training script for ICU 5-class classifier.
Supports transfer learning from pretrained encoder and comprehensive evaluation.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import logging
from pathlib import Path
import argparse
from tqdm import tqdm
import time
from typing import Dict, Tuple, List

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

from datasets import create_data_loaders
from models.icu_classifier import ICUClassifier, create_icu_classifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ICUTrainer:
    """Trainer for ICU 5-class classifier."""
    
    def __init__(self, config: Dict):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Set random seeds for reproducibility
        self._set_random_seeds(config.get('seed', 42))
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize data loaders
        self.data_loaders = self._create_data_loaders()
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize loss function
        self.criterion = self._create_loss_function()
        
        # Training state
        self.current_epoch = 0
        self.best_val_f1 = 0.0
        self.best_model_state = None
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
        
        # Target performance metrics
        self.target_accuracy = config.get('targets', {}).get('min_accuracy', 0.80)
        self.target_f1 = config.get('targets', {}).get('min_macro_f1', 0.80)
        
        logger.info(f"Trainer initialized. Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Target performance: {self.target_accuracy:.1%} accuracy, {self.target_f1:.1%} macro-F1")
    
    def _set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _create_model(self) -> ICUClassifier:
        """Create ICU classifier model."""
        model_config = self.config['model']
        
        # Create classifier
        classifier = create_icu_classifier(
            pretrained_encoder_path=model_config.get('pretrained_encoder'),
            freeze_encoder=False,  # Will be handled during training
            num_classes=model_config['classifier']['num_classes'],
            hidden_dim=model_config['classifier']['hidden_dim'],
            dropout_rate=model_config['classifier']['dropout_rate'],
            aggregation_method="attention"
        )
        
        return classifier
    
    def _create_data_loaders(self) -> Dict[str, DataLoader]:
        """Create data loaders."""
        data_config = self.config['data']
        
        # Use simple_processed_data directory
        data_dir = "simple_processed_data"
        
        return create_data_loaders(
            data_dir=data_dir,
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers'],
            target_frames=data_config['num_frames'],
            augment_train=True,
            load_in_memory=False
        )
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        train_config = self.config['training']
        
        return optim.AdamW(
            self.model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        train_config = self.config['training']
        scheduler_config = train_config.get('scheduler', {})
        
        if scheduler_config.get('type') == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 15),
                gamma=scheduler_config.get('gamma', 0.5)
            )
        elif scheduler_config.get('type') == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs']
            )
        else:
            return None
    
    def _create_loss_function(self):
        """Create loss function with class weighting."""
        train_config = self.config['training']
        
        if train_config.get('class_weights') == 'balanced':
            # Get class weights from training dataset
            train_dataset = self.data_loaders['train'].dataset
            class_weights = train_dataset.get_class_weights()
            class_weights = class_weights.to(self.device)
            logger.info(f"Using balanced class weights: {class_weights}")
        else:
            class_weights = None
        
        return nn.CrossEntropyLoss(weight=class_weights)
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        
        # Handle encoder freezing
        freeze_epochs = self.config['model']['encoder'].get('freeze_epochs', 0)
        if self.current_epoch < freeze_epochs:
            if not self.model.freeze_encoder:
                self.model.freeze_encoder()
                logger.info(f"Encoder frozen for first {freeze_epochs} epochs")
        else:
            if self.model.freeze_encoder:
                self.model.unfreeze_encoder()
                logger.info("Encoder unfrozen for fine-tuning")
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(self.data_loaders['train'], desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (data, labels) in enumerate(progress_bar):
            data, labels = data.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('grad_clip_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip_norm']
                )
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
            })
        
        avg_loss = total_loss / len(self.data_loaders['train'])
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float, float]:
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
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        epochs = self.config['training']['epochs']
        early_stopping_patience = self.config['training']['early_stopping']['patience']
        patience_counter = 0
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_f1 = self.validate()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['val_f1'].append(val_f1)
            
            # Log progress
            logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}"
            )
            
            # Save best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                logger.info(f"New best validation F1: {val_f1:.4f}")
                
                # Save checkpoint
                self.save_checkpoint(f"models/icu_5word_classifier_best.pt")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Check if target performance is reached
            if val_acc >= self.target_accuracy and val_f1 >= self.target_f1:
                logger.info(f"Target performance reached! Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
                break
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Loaded best model with validation F1: {self.best_val_f1:.4f}")
    
    def evaluate_on_test(self) -> Dict:
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
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'epoch': self.current_epoch,
            'best_val_f1': self.best_val_f1,
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train ICU 5-class classifier")
    parser.add_argument("--config", default="configs/icu_classifier.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data_dir", default="simple_processed_data",
                       help="Path to processed data directory")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override data directory if provided
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    
    # Create trainer and train
    trainer = ICUTrainer(config)
    trainer.train()
    
    # Evaluate on test set
    test_results = trainer.evaluate_on_test()
    
    # Print results
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    print(f"Macro F1: {test_results['f1_macro']:.4f}")
    print(f"Weighted F1: {test_results['f1_weighted']:.4f}")
    
    # Check if targets are met
    target_acc = trainer.target_accuracy
    target_f1 = trainer.target_f1
    
    if test_results['accuracy'] >= target_acc and test_results['f1_macro'] >= target_f1:
        print(f"✅ TARGET PERFORMANCE ACHIEVED!")
        print(f"   Accuracy: {test_results['accuracy']:.4f} >= {target_acc:.4f}")
        print(f"   Macro F1: {test_results['f1_macro']:.4f} >= {target_f1:.4f}")
    else:
        print(f"⚠️  Target performance not met:")
        print(f"   Accuracy: {test_results['accuracy']:.4f} < {target_acc:.4f}" if test_results['accuracy'] < target_acc else f"   Accuracy: {test_results['accuracy']:.4f} ✓")
        print(f"   Macro F1: {test_results['f1_macro']:.4f} < {target_f1:.4f}" if test_results['f1_macro'] < target_f1 else f"   Macro F1: {test_results['f1_macro']:.4f} ✓")
    
    print("\nPer-class results:")
    for class_name in test_results['class_names']:
        metrics = test_results['classification_report'][class_name]
        print(f"  {class_name}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    print("="*60)


if __name__ == "__main__":
    main()
