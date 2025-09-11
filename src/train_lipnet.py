"""
LipNet pretraining script with CTC loss on GRID corpus.
Stage 1 of the two-stage training pipeline.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from pathlib import Path
import argparse
from tqdm import tqdm
import time
from typing import Dict, Tuple

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

from grid_dataset import create_grid_data_loaders, GRIDVocabulary
from models.lipnet_encoder import LipNetEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LipNetCTC(nn.Module):
    """LipNet with CTC head for character-level recognition."""
    
    def __init__(self, encoder: LipNetEncoder, vocab_size: int):
        """
        Initialize LipNet with CTC head.
        
        Args:
            encoder: LipNet encoder
            vocab_size: Size of character vocabulary (including blank)
        """
        super(LipNetCTC, self).__init__()
        
        self.encoder = encoder
        self.vocab_size = vocab_size
        
        # CTC projection layer
        self.ctc_projection = nn.Linear(encoder.embedding_dim, vocab_size)
        
        # Initialize CTC layer
        nn.init.xavier_uniform_(self.ctc_projection.weight)
        nn.init.constant_(self.ctc_projection.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LipNet with CTC head.
        
        Args:
            x: Input video tensor
            
        Returns:
            CTC logits of shape (T, B, vocab_size)
        """
        # Extract features using encoder
        encoder_output = self.encoder(x)  # (B, T, embedding_dim)
        
        # Project to vocabulary size
        ctc_logits = self.ctc_projection(encoder_output)  # (B, T, vocab_size)
        
        # CTC expects (T, B, vocab_size)
        ctc_logits = ctc_logits.transpose(0, 1)  # (T, B, vocab_size)
        
        # Apply log softmax for CTC
        ctc_logits = torch.log_softmax(ctc_logits, dim=-1)
        
        return ctc_logits


class LipNetTrainer:
    """Trainer for LipNet pretraining with CTC loss."""
    
    def __init__(self, config: Dict):
        """
        Initialize LipNet trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        
        # Set random seeds
        self._set_random_seeds(config.get('seed', 42))
        
        # Create data loaders and vocabulary
        self.data_loaders, self.vocab = self._create_data_loaders()
        
        # Create model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Create optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Create CTC loss
        self.criterion = nn.CTCLoss(blank=self.vocab.blank_idx, reduction='mean', zero_infinity=True)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.training_history = {
            'train_loss': [],
            'val_loss': []
        }
        
        logger.info(f"LipNet trainer initialized. Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Vocabulary size: {self.vocab.vocab_size}")
    
    def _set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _create_data_loaders(self) -> Tuple[Dict[str, DataLoader], GRIDVocabulary]:
        """Create GRID data loaders."""
        data_config = self.config['data']
        
        return create_grid_data_loaders(
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers'],
            train_samples=data_config.get('train_samples', 1000),
            val_samples=data_config.get('val_samples', 200),
            test_samples=data_config.get('test_samples', 200)
        )
    
    def _create_model(self) -> LipNetCTC:
        """Create LipNet model with CTC head."""
        model_config = self.config['model']
        
        # Create encoder
        encoder = LipNetEncoder(
            input_channels=model_config['input_channels'],
            conv_channels=tuple(model_config['conv_channels']),
            gru_hidden_size=model_config['gru_hidden_size'],
            gru_num_layers=model_config['gru_num_layers'],
            embedding_dim=model_config['embedding_dim'],
            dropout_rate=model_config['dropout_rate']
        )
        
        # Create LipNet with CTC head
        model = LipNetCTC(encoder, self.vocab.vocab_size)
        
        return model
    
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
        
        if scheduler_config.get('type') == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=train_config['epochs']
            )
        elif scheduler_config.get('type') == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 20),
                gamma=scheduler_config.get('gamma', 0.5)
            )
        else:
            return None
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.data_loaders['train'], desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (videos, targets, input_lengths, target_lengths) in enumerate(progress_bar):
            videos = videos.to(self.device)
            targets = targets.to(self.device)
            input_lengths = input_lengths.to(self.device)
            target_lengths = target_lengths.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            ctc_logits = self.model(videos)  # (T, B, vocab_size)
            
            # CTC loss
            loss = self.criterion(ctc_logits, targets, input_lengths, target_lengths)
            
            # Handle NaN/inf losses
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Invalid loss detected: {loss.item()}")
                continue
            
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
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / num_batches:.4f}"
            })
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for videos, targets, input_lengths, target_lengths in tqdm(self.data_loaders['val'], desc="Validation"):
                videos = videos.to(self.device)
                targets = targets.to(self.device)
                input_lengths = input_lengths.to(self.device)
                target_lengths = target_lengths.to(self.device)
                
                # Forward pass
                ctc_logits = self.model(videos)
                
                # CTC loss
                loss = self.criterion(ctc_logits, targets, input_lengths, target_lengths)
                
                # Handle invalid losses
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def train(self):
        """Main training loop."""
        logger.info("Starting LipNet pretraining...")
        
        epochs = self.config['training']['epochs']
        early_stopping_patience = self.config['training']['early_stopping']['patience']
        patience_counter = 0
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            
            # Log progress
            logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f}"
            )
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                logger.info(f"New best validation loss: {val_loss:.4f}")
                
                # Save checkpoint
                self.save_checkpoint("models/lipnet_encoder_pretrained.pt")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Loaded best model with validation loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'encoder_state_dict': self.model.encoder.state_dict(),
            'full_model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'vocab': self.vocab,
            'epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
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
    parser = argparse.ArgumentParser(description="Train LipNet encoder with CTC loss")
    parser.add_argument("--config", default="configs/lipnet_grid.yaml",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create trainer and train
    trainer = LipNetTrainer(config)
    trainer.train()
    
    logger.info("LipNet pretraining completed!")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info("Encoder weights saved to: models/lipnet_encoder_pretrained.pt")


if __name__ == "__main__":
    main()
