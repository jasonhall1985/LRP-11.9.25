"""
Exponential Moving Average (EMA) for model weights
Stabilizes training and improves generalization
"""

import torch
import torch.nn as nn
from typing import Optional
import copy


class ModelEMA:
    """
    Exponential Moving Average of model weights
    
    Args:
        model: PyTorch model to track
        beta: EMA decay factor (default: 0.999)
        update_after_step: Whether to update after each optimizer step
    """
    
    def __init__(self, model: nn.Module, beta: float = 0.999, update_after_step: bool = True):
        self.model = model
        self.beta = beta
        self.update_after_step = update_after_step
        self.step_count = 0
        
        # Create EMA model as a deep copy
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        
        # Disable gradients for EMA model
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
    
    def update(self, model: Optional[nn.Module] = None):
        """
        Update EMA weights
        
        Args:
            model: Model to update from (uses self.model if None)
        """
        if model is None:
            model = self.model
            
        self.step_count += 1
        
        # Calculate effective beta (starts lower, approaches target)
        # This helps with initial convergence
        effective_beta = min(self.beta, (1 + self.step_count) / (10 + self.step_count))
        
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(effective_beta).add_(model_param.data, alpha=1 - effective_beta)
    
    def update_buffers(self, model: Optional[nn.Module] = None):
        """
        Update EMA buffers (batch norm running stats, etc.)
        
        Args:
            model: Model to update from (uses self.model if None)
        """
        if model is None:
            model = self.model
            
        with torch.no_grad():
            for ema_buffer, model_buffer in zip(self.ema_model.buffers(), model.buffers()):
                ema_buffer.data.copy_(model_buffer.data)
    
    def state_dict(self):
        """Return EMA model state dict"""
        return {
            'ema_model': self.ema_model.state_dict(),
            'beta': self.beta,
            'step_count': self.step_count,
            'update_after_step': self.update_after_step
        }
    
    def load_state_dict(self, state_dict):
        """Load EMA model state dict"""
        self.ema_model.load_state_dict(state_dict['ema_model'])
        self.beta = state_dict['beta']
        self.step_count = state_dict['step_count']
        self.update_after_step = state_dict['update_after_step']
    
    def get_model(self):
        """Get EMA model for inference"""
        return self.ema_model
    
    def __call__(self, *args, **kwargs):
        """Forward pass through EMA model"""
        return self.ema_model(*args, **kwargs)


class EMAWrapper:
    """
    Wrapper to easily switch between regular model and EMA model
    """
    
    def __init__(self, model: nn.Module, ema_config: dict):
        self.model = model
        self.ema_enabled = ema_config.get('enabled', False)
        
        if self.ema_enabled:
            self.ema = ModelEMA(
                model=model,
                beta=ema_config.get('beta', 0.999),
                update_after_step=ema_config.get('update_after_step', True)
            )
        else:
            self.ema = None
    
    def update_ema(self):
        """Update EMA weights if enabled"""
        if self.ema_enabled and self.ema is not None:
            self.ema.update()
    
    def get_model_for_validation(self):
        """Get model for validation (EMA if enabled and configured)"""
        if self.ema_enabled and self.ema is not None:
            return self.ema.get_model()
        return self.model
    
    def get_model_for_training(self):
        """Get model for training (always the original model)"""
        return self.model
    
    def state_dict(self):
        """Get state dict including EMA if enabled"""
        state = {'model': self.model.state_dict()}
        if self.ema_enabled and self.ema is not None:
            state['ema'] = self.ema.state_dict()
        return state
    
    def load_state_dict(self, state_dict):
        """Load state dict including EMA if present"""
        self.model.load_state_dict(state_dict['model'])
        if self.ema_enabled and self.ema is not None and 'ema' in state_dict:
            self.ema.load_state_dict(state_dict['ema'])
