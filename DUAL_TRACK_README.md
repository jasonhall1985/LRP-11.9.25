# Dual-Track Lip-Reading Evaluation System

## Overview

This system implements a **dual-track approach** that provides both honest cross-speaker generalization metrics and practical bedside calibration capabilities for lip-reading models. The approach maintains scientific integrity while addressing real-world deployment needs.

## 🎯 Core Philosophy

**Scientific Integrity + Practical Deployment**

- **Track 1**: Honest generalization metrics using LOSO cross-validation
- **Track 2**: Practical personalization for bedside deployment
- **Never conflate** the two metrics - clear separation maintained
- **Cross-adaptation validation** to detect overfitting

## 📊 System Components

### 1. LOSO Cross-Validation Framework (`loso_cross_validation_framework.py`)

**Purpose**: Provides honest cross-speaker generalization metrics

**Features**:
- Leave-One-Speaker-Out validation across all 6 speakers
- Train on 5 speakers, validate on 1 held-out speaker
- Zero speaker contamination between training and validation
- Comprehensive confusion matrices and per-speaker results

**Usage**:
```bash
python3 loso_cross_validation_framework.py --max-epochs 20 --output-dir loso_results
```

**Expected Results**: 36-44% validation accuracy (honest generalization)

### 2. Few-Shot Personalization Pipeline (`calibrate.py`)

**Purpose**: Rapid speaker adaptation for bedside deployment

**Features**:
- K-shot learning with K=10 or K=20 clips per class
- Head-only fine-tuning with frozen encoder (3-5 epochs)
- Target: >90% within-speaker accuracy in <1 minute
- Conservative augmentation to preserve lip visibility

**Usage**:
```bash
python3 calibrate.py --checkpoint base_model.pth --speaker "speaker 1 " --shots-per-class 10
```

**Expected Results**: >90% within-speaker accuracy in <60 seconds

### 3. Dual-Track Evaluation System (`dual_track_evaluation.py`)

**Purpose**: Comprehensive evaluation with clear metric separation

**Features**:
- Orchestrates both LOSO and personalization evaluations
- Cross-adaptation validation to detect overfitting
- Scientific dual-track reporting with visualizations
- Clear distinction between generalization and personalization

**Usage**:
```bash
python3 dual_track_evaluation.py --base-model base_model.pth --loso-epochs 20
```

### 4. Master Execution Script (`execute_dual_track_evaluation.py`)

**Purpose**: One-command execution of complete dual-track evaluation

**Features**:
- Automated prerequisite checking
- Sequential execution of all evaluation components
- Comprehensive logging and error handling
- Quick test mode for development

**Usage**:
```bash
# Full evaluation
python3 execute_dual_track_evaluation.py --output-dir results

# Quick test
python3 execute_dual_track_evaluation.py --quick-test --output-dir test_results
```

## 🚀 Quick Start

### Prerequisites

1. **Data Structure**:
   ```
   data/speaker sets/
   ├── speaker 1 /
   │   ├── doctor/
   │   ├── i_need_to_move/
   │   ├── my_mouth_is_dry/
   │   └── pillow/
   ├── speaker 2 /
   └── ... (6 speakers total)
   ```

2. **Required Files**:
   - `advanced_training_components.py` (model architecture)
   - All dual-track system files

### Running Complete Evaluation

```bash
# Full dual-track evaluation
python3 execute_dual_track_evaluation.py --output-dir dual_track_results

# Quick test (2 epochs)
python3 execute_dual_track_evaluation.py --quick-test --output-dir test_results
```

### Individual Components

```bash
# LOSO cross-validation only
python3 loso_cross_validation_framework.py --max-epochs 20 --output-dir loso_results

# Personalization for specific speaker
python3 calibrate.py --checkpoint model.pth --speaker "speaker 1 " --shots-per-class 10

# Generate dual-track report
python3 dual_track_evaluation.py --base-model model.pth --output-dir report_results
```

## 📈 Expected Results

### Track 1: LOSO Cross-Validation (Generalization)
- **Mean Accuracy**: 36-44% (honest cross-speaker performance)
- **Interpretation**: True generalization capability without speaker contamination
- **Clinical Relevance**: Robustness across different patients

### Track 2: Few-Shot Personalization (Bedside Calibration)
- **Mean Accuracy**: >90% (within-speaker after adaptation)
- **Adaptation Time**: <60 seconds
- **Interpretation**: Practical bedside usability after brief calibration
- **Clinical Relevance**: Immediate deployment capability

### Cross-Adaptation Validation
- **Overfitting Gap**: <20% (personalization doesn't overfit)
- **Interpretation**: Validates that personalization approach is sound

## 🔬 Scientific Integrity

### Clear Metric Separation
- **Never report** personalized accuracy as generalization performance
- **Always specify** which track results belong to
- **Maintain transparency** about adaptation requirements

### Honest Reporting Format
```
DUAL-TRACK RESULTS:
├── Generalization (LOSO): 38.5% ± 4.2%
└── Personalized (K=10): 92.1% ± 3.8% (after 45s calibration)
```

### Validation Safeguards
- LOSO prevents speaker contamination
- Cross-adaptation detects overfitting
- Conservative augmentation preserves lip visibility
- Head-only fine-tuning prevents catastrophic forgetting

## 🏥 Clinical Deployment Workflow

### Phase 1: Validation (Track 1)
1. Run LOSO cross-validation on development dataset
2. Report honest generalization metrics to clinical team
3. Establish baseline robustness expectations

### Phase 2: Bedside Deployment (Track 2)
1. Deploy base model to bedside system
2. Collect 10-20 clips per class from patient
3. Run 5-epoch personalization (< 1 minute)
4. Achieve >90% within-patient accuracy

### Phase 3: Monitoring
1. Periodic cross-adaptation validation
2. Monitor for overfitting or degradation
3. Retrain base model as needed

## 📁 Output Structure

```
dual_track_results/
├── loso_results/
│   ├── fold_speaker_1/
│   ├── fold_speaker_2/
│   └── loso_cross_validation_summary.json
├── personalization_results/
│   ├── personalization_speaker_1_K10.json
│   ├── personalization_speaker_2_K10.json
│   └── personalization_evaluation_results.json
├── dual_track_evaluation_report.png
├── dual_track_evaluation_report.txt
└── execution_log.json
```

## 🛠️ Technical Details

### Model Architecture
- **Base Model**: Enhanced Lightweight CNN-LSTM (787K parameters)
- **Input**: 32 frames × 64×96 grayscale video
- **Classes**: 4 ICU phrases (doctor, i_need_to_move, my_mouth_is_dry, pillow)

### Training Configuration
- **LOSO**: 15-20 epochs, AdamW optimizer, Focal Loss
- **Personalization**: 5 epochs, head-only fine-tuning, CrossEntropy Loss
- **Augmentation**: Conservative (brightness ±10%, contrast 0.9-1.1x)

### Hardware Requirements
- **GPU**: Recommended (CUDA/MPS support)
- **Memory**: 8GB+ RAM
- **Storage**: 2GB+ for datasets and results

## 🔧 Troubleshooting

### Common Issues

1. **"No videos found for speaker"**
   - Check data directory structure
   - Verify video file extensions (.mp4, .mov, .avi, .mkv)

2. **"BatchNorm error with batch size 1"**
   - Automatically handled with minimum batch size of 2
   - Drop last incomplete batch for training

3. **"CUDA out of memory"**
   - Reduce batch size in configuration
   - Use CPU mode: `--device cpu`

### Performance Optimization

1. **Faster LOSO**: Reduce epochs to 10-15 for development
2. **Faster Personalization**: Use K=10 instead of K=20
3. **Quick Testing**: Use `--quick-test` flag

## 📚 References

- **LOSO Validation**: Standard practice in speaker-independent speech recognition
- **Few-Shot Learning**: Meta-learning approaches for rapid adaptation
- **Clinical Deployment**: Bedside AI system requirements and constraints

## 🤝 Contributing

This dual-track system maintains scientific integrity while addressing practical deployment needs. When extending:

1. **Preserve metric separation** - never conflate generalization and personalization
2. **Maintain validation safeguards** - LOSO and cross-adaptation validation
3. **Document clinical relevance** - explain both tracks' purposes
4. **Test thoroughly** - use quick test mode for development

---

**Author**: Augment Agent  
**Date**: 2025-09-27  
**Purpose**: Honest generalization + practical deployment for lip-reading systems
