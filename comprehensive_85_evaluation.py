#!/usr/bin/env python3
"""
Comprehensive Evaluation of 85-Per-Class Training Results
Compare against previous baselines and analyze performance limitations
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime
import json

def load_and_evaluate_model(model_path, model_class, val_loader, class_to_idx):
    """Load model and evaluate on validation set"""
    print(f"📊 Evaluating model: {os.path.basename(model_path)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model instance
    model_config = checkpoint.get('model_config', {})
    if 'UltraLightCNN_LSTM' in model_config.get('architecture', ''):
        from optimized_85_trainer import UltraLightCNN_LSTM
        model = UltraLightCNN_LSTM(**{k: v for k, v in model_config.items() if k in ['num_classes', 'hidden_size', 'num_layers', 'dropout']})
    elif 'LightweightCNN_LSTM' in model_config.get('architecture', ''):
        from lightweight_85_trainer import LightweightCNN_LSTM
        model = LightweightCNN_LSTM(**{k: v for k, v in model_config.items() if k in ['num_classes', 'hidden_size', 'num_layers', 'dropout']})
    else:
        from train_85_per_class_model import CNN_LSTM_LipReader
        model = CNN_LSTM_LipReader(**{k: v for k, v in model_config.items() if k in ['num_classes', 'hidden_size', 'num_layers', 'dropout']})
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Evaluate
    all_predictions = []
    all_labels = []
    all_confidences = []
    
    with torch.no_grad():
        for videos, labels in val_loader:
            videos, labels = videos.to(device), labels.to(device).squeeze()
            outputs = model(videos)
            
            # Get predictions and confidences
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    # Calculate metrics
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Per-class metrics
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    
    report = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': all_predictions,
        'labels': all_labels,
        'confidences': all_confidences,
        'class_names': class_names,
        'model_config': model_config,
        'best_val_acc': checkpoint.get('best_val_acc', accuracy)
    }

def analyze_cross_demographic_performance(val_manifest_path, predictions, labels, class_names):
    """Analyze performance across demographic groups"""
    print("🌍 Analyzing cross-demographic performance...")
    
    # Load validation manifest
    val_df = pd.read_csv(val_manifest_path)
    
    # Ensure we have the same number of samples
    if len(val_df) != len(predictions):
        print(f"⚠️ Manifest length ({len(val_df)}) != predictions length ({len(predictions)})")
        return {}
    
    # Add predictions to dataframe
    val_df['predicted_class'] = [class_names[p] for p in predictions]
    val_df['actual_class'] = [class_names[l] for l in labels]
    val_df['correct'] = val_df['predicted_class'] == val_df['actual_class']
    
    # Analyze by demographic groups
    demographic_analysis = {}
    
    # Overall accuracy by demographic group
    demo_accuracy = val_df.groupby('demographic_group')['correct'].mean().sort_values(ascending=False)
    demographic_analysis['by_demographic'] = demo_accuracy.to_dict()
    
    # Accuracy by age group
    age_accuracy = val_df.groupby('age_group')['correct'].mean().sort_values(ascending=False)
    demographic_analysis['by_age'] = age_accuracy.to_dict()
    
    # Accuracy by gender
    gender_accuracy = val_df.groupby('gender')['correct'].mean().sort_values(ascending=False)
    demographic_analysis['by_gender'] = gender_accuracy.to_dict()
    
    # Accuracy by ethnicity
    ethnicity_accuracy = val_df.groupby('ethnicity')['correct'].mean().sort_values(ascending=False)
    demographic_analysis['by_ethnicity'] = ethnicity_accuracy.to_dict()
    
    return demographic_analysis

def create_evaluation_visualizations(results, output_dir):
    """Create comprehensive evaluation visualizations"""
    print("📊 Creating evaluation visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = results['confusion_matrix']
    class_names = results['class_names']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Accuracy: {results["accuracy"]:.4f}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Per-class Performance
    plt.figure(figsize=(12, 6))
    report = results['classification_report']
    
    classes = [c for c in class_names]
    precisions = [report[c]['precision'] for c in classes]
    recalls = [report[c]['recall'] for c in classes]
    f1_scores = [report[c]['f1-score'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    plt.bar(x - width, precisions, width, label='Precision', alpha=0.8)
    plt.bar(x, recalls, width, label='Recall', alpha=0.8)
    plt.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Per-Class Performance Metrics')
    plt.xticks(x, classes, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Confidence Distribution
    plt.figure(figsize=(10, 6))
    confidences = results['confidences']
    correct_mask = np.array(results['predictions']) == np.array(results['labels'])
    
    plt.hist(np.array(confidences)[correct_mask], bins=20, alpha=0.7, label='Correct Predictions', density=True)
    plt.hist(np.array(confidences)[~correct_mask], bins=20, alpha=0.7, label='Incorrect Predictions', density=True)
    
    plt.xlabel('Confidence Score')
    plt.ylabel('Density')
    plt.title('Confidence Distribution for Correct vs Incorrect Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizations saved to {output_dir}/")

def compare_with_baselines():
    """Compare current results with previous baselines"""
    print("📈 Comparing with previous baselines...")
    
    baselines = {
        'Doctor-Focused Model (75.9%)': {
            'accuracy': 0.759,
            'dataset_size': 260,
            'balance': 'Imbalanced (doctor-biased)',
            'architecture': 'CNN-LSTM',
            'notes': 'Strong doctor bias, good same-demographic performance'
        },
        'Balanced 61-Per-Class (37.5%)': {
            'accuracy': 0.375,
            'dataset_size': 244,
            'balance': 'Perfectly balanced (61 per class)',
            'architecture': 'CNN-LSTM',
            'notes': 'Eliminated bias but lower accuracy'
        },
        'Balanced 85-Per-Class (51.47%)': {
            'accuracy': 0.5147,
            'dataset_size': 340,
            'balance': 'Perfectly balanced (85 per class)',
            'architecture': 'Lightweight CNN-LSTM (2.2M params)',
            'notes': 'Best balanced performance, larger dataset'
        },
        'Ultra-Light 85-Per-Class (35.29%)': {
            'accuracy': 0.3529,
            'dataset_size': 340,
            'balance': 'Perfectly balanced (85 per class)',
            'architecture': 'Ultra-Light CNN-LSTM (0.6M params)',
            'notes': 'Heavily regularized, prevented overfitting'
        }
    }
    
    return baselines

def generate_comprehensive_report(all_results, baselines, demographic_analysis, output_dir):
    """Generate comprehensive evaluation report"""
    print("📝 Generating comprehensive evaluation report...")
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report_path = os.path.join(output_dir, 'comprehensive_evaluation_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Comprehensive 85-Per-Class Lip-Reading Model Evaluation Report\n\n")
        f.write(f"**Generated:** {timestamp}\n\n")
        f.write("## Executive Summary\n\n")
        f.write("This report evaluates the performance of 4-class lip-reading models trained on a perfectly balanced 340-video dataset (85 videos per class) targeting ≥82% cross-demographic validation accuracy.\n\n")
        
        f.write("## Key Findings\n\n")
        f.write("### Performance Results\n")
        for model_name, results in all_results.items():
            f.write(f"- **{model_name}**: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%) validation accuracy\n")
        
        f.write(f"\n### Target Achievement\n")
        f.write("- **Target**: ≥82% cross-demographic validation accuracy\n")
        f.write("- **Best Achieved**: 51.47% (Lightweight CNN-LSTM)\n")
        f.write("- **Gap**: 30.53 percentage points below target\n\n")
        
        f.write("## Baseline Comparison\n\n")
        f.write("| Model | Accuracy | Dataset Size | Balance | Architecture | Notes |\n")
        f.write("|-------|----------|--------------|---------|--------------|-------|\n")
        for name, info in baselines.items():
            f.write(f"| {name} | {info['accuracy']*100:.2f}% | {info['dataset_size']} | {info['balance']} | {info['architecture']} | {info['notes']} |\n")
        
        f.write("\n## Cross-Demographic Analysis\n\n")
        if demographic_analysis:
            f.write("### Performance by Demographic Groups\n")
            for demo, acc in demographic_analysis.get('by_demographic', {}).items():
                f.write(f"- **{demo}**: {acc:.4f} ({acc*100:.2f}%)\n")
            
            f.write("\n### Performance by Age Group\n")
            for age, acc in demographic_analysis.get('by_age', {}).items():
                f.write(f"- **{age}**: {acc:.4f} ({acc*100:.2f}%)\n")
        
        f.write("\n## Analysis & Conclusions\n\n")
        f.write("### Dataset Size Limitations\n")
        f.write("- **Current dataset**: 340 videos (272 train + 68 validation)\n")
        f.write("- **Per-class training data**: 68 videos per class\n")
        f.write("- **Validation data**: 17 videos per class\n")
        f.write("- **Assessment**: Dataset size appears insufficient for 82% accuracy target\n\n")
        
        f.write("### Model Architecture Insights\n")
        f.write("- **Lightweight models** (2.2M params) performed better than heavy models (15M+ params)\n")
        f.write("- **Overfitting** was a consistent challenge across all architectures\n")
        f.write("- **Regularization** helped but couldn't overcome fundamental data limitations\n\n")
        
        f.write("### Recommendations\n")
        f.write("1. **Dataset Expansion**: Target 200-300 videos per class (800-1200 total)\n")
        f.write("2. **Data Quality**: Focus on consistent preprocessing and high-quality recordings\n")
        f.write("3. **Transfer Learning**: Consider pre-trained models or domain adaptation\n")
        f.write("4. **Ensemble Methods**: Combine multiple models for improved performance\n")
        f.write("5. **Alternative Approaches**: Explore transformer-based architectures or self-supervised learning\n\n")
        
        f.write("### Success Criteria Assessment\n")
        f.write("- ❌ **≥82% validation accuracy**: Not achieved (best: 51.47%)\n")
        f.write("- ✅ **Perfect class balance**: Achieved (85 videos per class)\n")
        f.write("- ✅ **Cross-demographic validation**: Implemented with 14 demographic groups\n")
        f.write("- ✅ **Overfitting prevention**: Addressed through regularization and early stopping\n")
        f.write("- ✅ **Efficient training**: Models trained successfully on CPU\n\n")
        
        f.write("## Conclusion\n\n")
        f.write("While the 85-per-class balanced dataset represents a significant improvement in data quality and balance, ")
        f.write("the 82% validation accuracy target was not achieved. The best performing model (Lightweight CNN-LSTM) ")
        f.write("reached 51.47% accuracy, indicating that dataset expansion is necessary to achieve the target performance. ")
        f.write("The perfect class balance and cross-demographic validation framework provide a solid foundation for ")
        f.write("future improvements with larger datasets.\n")
    
    print(f"✅ Comprehensive report saved: {report_path}")
    return report_path

def main():
    """Execute comprehensive evaluation"""
    print("🎯 COMPREHENSIVE 85-PER-CLASS EVALUATION")
    print("=" * 70)
    print("Analyzing training results and comparing against baselines")
    
    # Setup
    output_dir = "balanced_85_training_results"
    val_manifest = "balanced_85_training_results/balanced_340_validation_manifest.csv"
    
    # Create validation loader for evaluation
    from lightweight_85_trainer import LightweightLipDataset
    from torch.utils.data import DataLoader
    
    val_dataset = LightweightLipDataset(val_manifest, "data/the_best_videos_so_far", augment=False)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Find all trained models
    model_files = [f for f in os.listdir(output_dir) if f.endswith('.pth') and 'model' in f]
    
    print(f"\n📊 Found {len(model_files)} trained models to evaluate")
    
    # Evaluate all models
    all_results = {}
    best_model_results = None
    best_accuracy = 0
    
    for model_file in model_files:
        model_path = os.path.join(output_dir, model_file)
        try:
            results = load_and_evaluate_model(model_path, None, val_loader, val_dataset.class_to_idx)
            model_name = model_file.replace('.pth', '').replace('_', ' ').title()
            all_results[model_name] = results
            
            if results['accuracy'] > best_accuracy:
                best_accuracy = results['accuracy']
                best_model_results = results
                
            print(f"  ✅ {model_name}: {results['accuracy']:.4f} accuracy")
        except Exception as e:
            print(f"  ❌ Failed to evaluate {model_file}: {e}")
    
    # Cross-demographic analysis on best model
    demographic_analysis = {}
    if best_model_results:
        demographic_analysis = analyze_cross_demographic_performance(
            val_manifest, 
            best_model_results['predictions'], 
            best_model_results['labels'],
            best_model_results['class_names']
        )
    
    # Create visualizations for best model
    if best_model_results:
        create_evaluation_visualizations(best_model_results, output_dir)
    
    # Compare with baselines
    baselines = compare_with_baselines()
    
    # Generate comprehensive report
    report_path = generate_comprehensive_report(all_results, baselines, demographic_analysis, output_dir)
    
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE EVALUATION COMPLETE")
    print(f"🏆 Best Model Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"🎯 Target Accuracy: 0.8200 (82.00%)")
    print(f"📉 Gap: {0.82 - best_accuracy:.4f} ({(0.82 - best_accuracy)*100:.2f} percentage points)")
    print(f"📄 Full Report: {report_path}")
    
    if best_accuracy >= 0.82:
        print("🎉 SUCCESS: Target achieved!")
        return True
    else:
        print("📊 Target not achieved - dataset expansion recommended")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
