#!/usr/bin/env python3
"""
Comprehensive Metrics for Lip Reading Training
==============================================

Production-ready metrics suite including:
- Top-1 accuracy
- Macro and weighted F1 scores
- Per-class precision/recall/F1
- Confusion matrix visualization
- ROC curves and AUC scores
- Statistical significance testing

Features:
- Real-time metric computation
- Comprehensive visualization
- Statistical analysis
- Class-specific performance tracking
- Production-ready reporting

Author: Production Lip Reading System
Date: 2025-09-15
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.preprocessing import label_binarize
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Comprehensive metrics tracking for lip reading training.
    """
    
    def __init__(
        self,
        class_names: List[str],
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize metrics tracker.
        
        Args:
            class_names: List of class names
            device: Device for tensor operations
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.device = device
        
        # Storage for predictions and targets
        self.reset()
        
        logger.info(f"MetricsTracker initialized for {self.num_classes} classes")
        
    def reset(self):
        """Reset all stored predictions and targets."""
        self.all_predictions = []
        self.all_targets = []
        self.all_probabilities = []
        
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        probabilities: Optional[torch.Tensor] = None
    ):
        """
        Update metrics with new batch.
        
        Args:
            predictions: Predicted class indices (B,)
            targets: Ground truth class indices (B,)
            probabilities: Class probabilities (B, num_classes)
        """
        # Move to CPU and convert to numpy
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        self.all_predictions.extend(pred_np.tolist())
        self.all_targets.extend(target_np.tolist())
        
        if probabilities is not None:
            prob_np = probabilities.detach().cpu().numpy()
            self.all_probabilities.extend(prob_np.tolist())
            
    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute comprehensive metrics.
        
        Returns:
            Dictionary of computed metrics
        """
        if len(self.all_predictions) == 0:
            return {}
            
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)
        
        # Basic accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Precision, recall, F1 scores
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Macro and weighted averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        weighted_precision = np.average(precision, weights=support)
        weighted_recall = np.average(recall, weights=support)
        weighted_f1 = np.average(f1, weights=support)
        
        # Per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            per_class_metrics[class_name] = {
                'precision': float(precision[i]) if i < len(precision) else 0.0,
                'recall': float(recall[i]) if i < len(recall) else 0.0,
                'f1': float(f1[i]) if i < len(f1) else 0.0,
                'support': int(support[i]) if i < len(support) else 0
            }
            
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        
        # ROC AUC (if probabilities available)
        roc_auc_scores = {}
        if len(self.all_probabilities) > 0:
            y_prob = np.array(self.all_probabilities)
            
            # Binarize targets for multi-class ROC
            y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
            
            for i, class_name in enumerate(self.class_names):
                if i < y_prob.shape[1] and np.sum(y_true_bin[:, i]) > 0:
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                    roc_auc_scores[class_name] = auc(fpr, tpr)
                else:
                    roc_auc_scores[class_name] = 0.0
                    
            # Macro average ROC AUC
            macro_roc_auc = np.mean(list(roc_auc_scores.values()))
        else:
            macro_roc_auc = 0.0
            
        metrics = {
            'accuracy': float(accuracy),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1),
            'weighted_precision': float(weighted_precision),
            'weighted_recall': float(weighted_recall),
            'weighted_f1': float(weighted_f1),
            'macro_roc_auc': float(macro_roc_auc),
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm.tolist(),
            'roc_auc_scores': roc_auc_scores,
            'num_samples': len(y_true)
        }
        
        return metrics
        
    def plot_confusion_matrix(
        self,
        save_path: Optional[str] = None,
        normalize: bool = True,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            save_path: Path to save the plot
            normalize: Whether to normalize the matrix
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if len(self.all_predictions) == 0:
            logger.warning("No predictions available for confusion matrix")
            return None
            
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm)  # Handle division by zero
            
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax
        )
        
        ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to: {save_path}")
            
        return fig
        
    def plot_roc_curves(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot ROC curves for all classes.
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if len(self.all_probabilities) == 0:
            logger.warning("No probabilities available for ROC curves")
            return None
            
        y_true = np.array(self.all_targets)
        y_prob = np.array(self.all_probabilities)
        
        # Binarize targets
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot ROC curve for each class
        colors = plt.cm.Set1(np.linspace(0, 1, self.num_classes))
        
        for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
            if i < y_prob.shape[1] and np.sum(y_true_bin[:, i]) > 0:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                
                ax.plot(
                    fpr, tpr,
                    color=color,
                    lw=2,
                    label=f'{class_name} (AUC = {roc_auc:.3f})'
                )
                
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves for All Classes')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to: {save_path}")
            
        return fig
        
    def plot_per_class_metrics(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot per-class precision, recall, and F1 scores.
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        metrics = self.compute_metrics()
        if not metrics:
            logger.warning("No metrics available for plotting")
            return None
            
        per_class = metrics['per_class_metrics']
        
        # Extract metrics
        classes = list(per_class.keys())
        precision = [per_class[cls]['precision'] for cls in classes]
        recall = [per_class[cls]['recall'] for cls in classes]
        f1 = [per_class[cls]['f1'] for cls in classes]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(classes))
        width = 0.25
        
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        # Add value labels on bars
        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
            ax.text(i - width, p + 0.01, f'{p:.2f}', ha='center', va='bottom', fontsize=8)
            ax.text(i, r + 0.01, f'{r:.2f}', ha='center', va='bottom', fontsize=8)
            ax.text(i + width, f + 0.01, f'{f:.2f}', ha='center', va='bottom', fontsize=8)
            
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Per-class metrics plot saved to: {save_path}")
            
        return fig
        
    def generate_report(
        self,
        save_path: Optional[str] = None,
        include_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive metrics report.
        
        Args:
            save_path: Path to save the report
            include_plots: Whether to generate and save plots
            
        Returns:
            Complete metrics report
        """
        metrics = self.compute_metrics()
        if not metrics:
            logger.warning("No metrics available for report")
            return {}
            
        # Add statistical analysis
        if len(self.all_predictions) > 30:  # Minimum sample size for statistical tests
            metrics['statistical_analysis'] = self._compute_statistical_analysis()
            
        # Generate plots if requested
        if include_plots and save_path:
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Confusion matrix
            cm_path = save_dir / 'confusion_matrix.png'
            self.plot_confusion_matrix(str(cm_path))
            
            # ROC curves
            roc_path = save_dir / 'roc_curves.png'
            self.plot_roc_curves(str(roc_path))
            
            # Per-class metrics
            metrics_path = save_dir / 'per_class_metrics.png'
            self.plot_per_class_metrics(str(metrics_path))
            
            metrics['plot_paths'] = {
                'confusion_matrix': str(cm_path),
                'roc_curves': str(roc_path),
                'per_class_metrics': str(metrics_path)
            }
            
        # Save report
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            logger.info(f"Metrics report saved to: {save_path}")
            
        return metrics
        
    def _compute_statistical_analysis(self) -> Dict[str, Any]:
        """Compute statistical significance analysis."""
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)
        
        # Compute confidence intervals for accuracy using bootstrap
        n_bootstrap = 1000
        bootstrap_accuracies = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
            bootstrap_acc = accuracy_score(y_true[indices], y_pred[indices])
            bootstrap_accuracies.append(bootstrap_acc)
            
        # Confidence intervals
        ci_lower = np.percentile(bootstrap_accuracies, 2.5)
        ci_upper = np.percentile(bootstrap_accuracies, 97.5)
        
        # McNemar's test for comparing with random baseline
        random_predictions = np.random.choice(self.num_classes, size=len(y_true))
        
        # Create contingency table for McNemar's test
        correct_model = (y_pred == y_true)
        correct_random = (random_predictions == y_true)
        
        both_correct = np.sum(correct_model & correct_random)
        model_correct_random_wrong = np.sum(correct_model & ~correct_random)
        model_wrong_random_correct = np.sum(~correct_model & correct_random)
        both_wrong = np.sum(~correct_model & ~correct_random)
        
        # McNemar's test statistic
        if model_correct_random_wrong + model_wrong_random_correct > 0:
            mcnemar_stat = ((model_correct_random_wrong - model_wrong_random_correct) ** 2) / \
                          (model_correct_random_wrong + model_wrong_random_correct)
            mcnemar_p_value = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
        else:
            mcnemar_stat = 0.0
            mcnemar_p_value = 1.0
            
        return {
            'accuracy_ci_95': [float(ci_lower), float(ci_upper)],
            'bootstrap_mean_accuracy': float(np.mean(bootstrap_accuracies)),
            'bootstrap_std_accuracy': float(np.std(bootstrap_accuracies)),
            'mcnemar_statistic': float(mcnemar_stat),
            'mcnemar_p_value': float(mcnemar_p_value),
            'significantly_better_than_random': mcnemar_p_value < 0.05
        }
        
    def print_summary(self):
        """Print a summary of current metrics."""
        metrics = self.compute_metrics()
        if not metrics:
            logger.info("No metrics available")
            return
            
        logger.info("\n" + "="*60)
        logger.info("METRICS SUMMARY")
        logger.info("="*60)
        
        logger.info(f"\n📊 Overall Performance:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Macro F1: {metrics['macro_f1']:.4f}")
        logger.info(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
        logger.info(f"  Macro ROC AUC: {metrics['macro_roc_auc']:.4f}")
        logger.info(f"  Samples: {metrics['num_samples']:,}")
        
        logger.info(f"\n📋 Per-Class Performance:")
        for class_name, class_metrics in metrics['per_class_metrics'].items():
            logger.info(f"  {class_name}:")
            logger.info(f"    Precision: {class_metrics['precision']:.4f}")
            logger.info(f"    Recall: {class_metrics['recall']:.4f}")
            logger.info(f"    F1: {class_metrics['f1']:.4f}")
            logger.info(f"    Support: {class_metrics['support']}")
            
        if 'statistical_analysis' in metrics:
            stats_analysis = metrics['statistical_analysis']
            logger.info(f"\n📈 Statistical Analysis:")
            ci_lower, ci_upper = stats_analysis['accuracy_ci_95']
            logger.info(f"  Accuracy 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            logger.info(f"  Significantly better than random: {stats_analysis['significantly_better_than_random']}")
            
        logger.info("="*60)
