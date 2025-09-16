"""
Models package for 7-Class Lip Reading Trainer
==============================================

Contains model architectures and utilities for production-ready lip reading.

Available Models:
- R(2+1)D-18: Pretrained video classification backbone
"""

from .r2p1d import LipReadingR2Plus1D, create_model

__all__ = ['LipReadingR2Plus1D', 'create_model']
