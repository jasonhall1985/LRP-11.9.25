# 🚀 Production-Ready 7-Class Lip Reading Trainer

Fast, production-ready trainer for **96×96 mouth ROI videos** using pretrained **R(2+1)D-18** backbone. Targets **>80% generalization accuracy** with comprehensive features including mixed precision training, demographic-based splits, CLAHE enhancement, and class balancing.

## ✨ Key Features

- **🎯 Production-Ready**: Comprehensive error handling, logging, and monitoring
- **⚡ Fast Training**: Mixed precision, automatic GPU detection, progressive unfreezing
- **📊 Class Balancing**: Multiple strategies including weighted sampling and physical duplication
- **🔍 CLAHE Enhancement**: Lighting and contrast standardization across datasets
- **📈 Comprehensive Metrics**: ROC curves, confusion matrices, statistical significance testing
- **🎛️ Demographic Splits**: Stratified validation/test splits for robust evaluation
- **💾 Smart Checkpointing**: Best model saving, early stopping, resume capability

## 🏗️ Architecture

- **Backbone**: Pretrained R(2+1)D-18 with Kinetics400 weights
- **Input**: 96×96 grayscale mouth ROI videos (T=24 frames)
- **Output**: 7-class classification
- **Classes**: `help`, `doctor`, `glasses`, `phone`, `pillow`, `i_need_to_move`, `my_mouth_is_dry`

## 🚀 Quick Start

### 1. Automated Training (Recommended)
```bash
# Run the complete pipeline automatically
./run_train.sh
```

### 2. Manual Training Steps

#### Prepare Dataset Manifest
```bash
python prepare_manifest.py \
  --sources \
  "/Users/client/Desktop/LRP classifier 11.9.25/data/videos for training 14.9.25 not cropped completely /13.9.25top7dataset_cropped" \
  "/Users/client/Desktop/training set 2.9.25" \
  "/Users/client/Desktop/VAL set" \
  "/Users/client/Desktop/test set" \
  --processed_dir "fixed_temporal_output/full_processed" \
  --out manifest.csv \
  --balance_classes \
  --balance_source_demo "gender=male,age_band=18-39" \
  --show_stats
```

#### Train Model
```bash
# Recommended: Weighted sampling for balanced classes
python train.py \
  --manifest manifest.csv \
  --config config.yaml \
  --balance weighted_sampler \
  --output_dir ./experiments/run_001 \
  --gpu 0
```

## 📊 Class Balancing

The system automatically balances classes by duplicating videos from the **male 18-39** demographic group to ensure equal representation across all 7 classes.

### Balancing Strategies

1. **Physical Duplication** (in manifest preparation)
   - Creates duplicate entries for underrepresented classes
   - Uses male 18-39 demographic as source for duplication
   - Ensures perfectly balanced training data

2. **Weighted Sampling** (during training)
   - Uses `WeightedRandomSampler` with inverse square root weighting
   - No additional storage required
   - Dynamically balances batches

3. **Focal Loss** (alternative)
   - Addresses imbalance through loss function
   - Focuses on hard examples
   - Good for severe imbalance

## ⚙️ Key Configuration

```yaml
# Class Balancing
balance:
  method: "weighted_sampler"
  weight_mode: "inverse_sqrt"

# Demographic Splits  
splits:
  val_holdout: "gender=male,age_band=40-64"
  test_holdout: "gender=female,age_band=18-39"

# Data Processing
data:
  clip_len: 24
  img_size: 96
  resize_for_backbone: 112
  clahe_enabled: true
  clahe_clip_limit: 2.0

# Training
training:
  batch_size: 16
  epochs: 50
  mixed_precision: true
  early_stop_patience: 8
```

## 🎯 Performance Targets

- **Primary Target**: >80% test accuracy
- **Secondary Target**: >75% macro F1-score
- **Class Balance**: Equal samples across all 7 classes

## 📈 Expected Results

With balanced classes and proper setup:
- **Training Accuracy**: 85-95%
- **Validation Accuracy**: 80-90%
- **Test Accuracy**: >80% (target)
- **Training Time**: 2-4 hours on modern GPU

## 🔧 Troubleshooting

### Common Issues

#### Class Imbalance
```bash
# Check class distribution
python balance.py --method analyze --manifest manifest.csv

# Enable class balancing in manifest preparation
python prepare_manifest.py --balance_classes
```

#### Poor Performance
```bash
# Try focal loss
python train.py --balance focal_loss

# Increase epochs
# Edit training.epochs in config.yaml
```

#### Data Path Issues
```bash
# Verify paths exist
python prepare_manifest.py --verify_videos
```

## 📁 Output Files

```
experiments/run_001/
├── best_model.pth                    # Best model checkpoint
├── final_report.json                 # Comprehensive results
├── TRAINING_SUMMARY.md               # Human-readable summary
├── confusion_matrix.png              # Confusion matrix plot
├── roc_curves.png                    # ROC curves plot
├── training_history.json             # Training curves data
└── split_analysis.json               # Dataset split statistics
```

## 🎉 Success Indicators

Training is successful when you see:
```
🎉 ALL TARGETS ACHIEVED! 🎉
Target Achievement:
Accuracy ≥ 80.0%: ✅ ACHIEVED
Macro F1 ≥ 75.0%: ✅ ACHIEVED
```

## 📋 Requirements

- **Python**: 3.8+
- **GPU**: NVIDIA GPU recommended
- **Memory**: 16GB+ RAM
- **Storage**: 50GB+ for experiments

### Key Dependencies
- `torch>=2.0.0`
- `torchvision>=0.15.0`
- `opencv-python>=4.8.0`
- `pandas>=1.5.0`
- `scikit-learn>=1.3.0`

---

**Built for production lip reading with >80% accuracy and balanced classes** 🎯
