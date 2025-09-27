# Three-Stage Training Pipeline for Lip-Reading

## Overview

This repository implements a comprehensive three-stage training pipeline for lip-reading models that provides both honest cross-speaker generalization metrics and practical bedside calibration capabilities.

### Pipeline Stages

1. **GRID Corpus Pretraining** - Learn robust visual features from viseme-matched words
2. **ICU Fine-tuning with LOSO** - Speaker-disjoint validation for honest generalization  
3. **Few-shot Personalization** - Rapid adaptation for bedside deployment

## Architecture

### Stage 1: GRID Pretraining
- **Purpose**: Learn robust visual speech features from large-scale GRID corpus
- **Strategy**: Multi-class word classification on viseme-matched GRID words
- **Model**: EnhancedLightweightCNNLSTM (787K parameters)
- **Data**: Selected GRID words with highest viseme similarity to ICU phrases
- **Output**: Pretrained encoder with strong visual feature representations

### Stage 2: ICU Fine-tuning (LOSO)
- **Purpose**: Adapt to ICU domain with honest cross-speaker evaluation
- **Strategy**: LOSO cross-validation for speaker-disjoint validation
- **Model**: GRID pretrained encoder + new 4-class ICU classifier
- **Data**: Organized speaker sets from ICU dataset
- **Output**: Base model with honest generalization metrics (36-44% expected)

### Stage 3: Few-shot Personalization
- **Purpose**: Rapid speaker-specific adaptation for bedside use
- **Strategy**: K-shot learning with head-only fine-tuning
- **Model**: Frozen encoder + personalized classifier
- **Data**: 10-20 examples per class from target speaker
- **Output**: Personalized model (>90% target accuracy)

## Key Components

### Viseme Mapping System (`utils/viseme_mapper.py`)
- Maps phonemes to 14-viseme system for visual similarity
- Calculates similarity between ICU phrases and GRID words
- Enables optimal GRID subset selection for pretraining

### Data Organization (`tools/organize_speaker_data.py`)
- Organizes existing videos into required speaker sets structure
- Extracts demographic information from filenames
- Creates balanced speaker groups for LOSO validation

### GRID Processing Tools
- `tools/build_grid_manifest.py` - Scans GRID corpus and builds manifests
- `tools/select_grid_subset.py` - Selects optimal GRID words by viseme similarity
- `train_grid_pretrain.py` - Executes GRID pretraining

### ICU Fine-tuning (`train_icu_finetune.py`)
- Loads GRID pretrained encoder
- Implements staged unfreezing (frozen → unfrozen encoder)
- Uses existing LOSO framework for speaker-disjoint validation

### Personalization (`calibrate.py`)
- Existing few-shot personalization system
- Head-only fine-tuning with frozen encoder
- Rapid adaptation in <1 minute

## Usage

### Quick Start
```bash
# Execute complete pipeline
python3 execute_three_stage_pipeline.py

# With custom configuration
python3 execute_three_stage_pipeline.py \
    --grid-epochs 30 \
    --icu-epochs 20 \
    --personalization-epochs 5 \
    --batch-size 16
```

### Individual Stages

#### Stage 1: GRID Pretraining
```bash
# Build GRID manifest (if GRID data available)
python3 tools/build_grid_manifest.py --grid-dir data/grid

# Select viseme-matched subset
python3 tools/select_grid_subset.py --words-per-class 20

# Train GRID pretraining
python3 train_grid_pretrain.py --max-epochs 30
```

#### Stage 2: ICU Fine-tuning
```bash
# Fine-tune with GRID pretrained encoder
python3 train_icu_finetune.py \
    --pretrained-encoder checkpoints/grid_pretraining/grid_pretrain_best.pth \
    --max-epochs 20

# Or train from scratch
python3 train_icu_finetune.py --no-pretrained --max-epochs 20
```

#### Stage 3: Personalization
```bash
# Few-shot personalization
python3 calibrate.py \
    --base-model checkpoints/icu_finetuning/icu_finetune_fold_1_best.pth \
    --k-shot 10
```

## Data Requirements

### ICU Dataset (Required)
- **Location**: `data/speaker sets/`
- **Structure**: `speaker X/class_name/video_files`
- **Classes**: doctor, i_need_to_move, my_mouth_is_dry, pillow
- **Format**: Videos organized by speaker and class

### GRID Corpus (Optional)
- **Location**: `data/grid/`
- **Structure**: `sX/sentence_files` (X = speaker number)
- **Purpose**: Pretraining for better visual features
- **Fallback**: Can train from scratch without GRID

## Expected Performance

### Honest Generalization (LOSO)
- **Without GRID Pretraining**: 36-44% cross-speaker accuracy
- **With GRID Pretraining**: 45-55% cross-speaker accuracy (estimated)
- **Scientific Integrity**: True speaker-disjoint validation

### Personalization Performance
- **Target Accuracy**: >90% within-speaker accuracy
- **Training Time**: <1 minute per speaker
- **Data Requirements**: 10-20 examples per class

## Scientific Integrity

### Dual-Track Approach
- **Track 1**: Honest generalization metrics via LOSO
- **Track 2**: Personalized performance for clinical deployment
- **Clear Separation**: Never conflate generalization vs personalization metrics

### Validation Strategy
- **LOSO Cross-Validation**: Leave-One-Speaker-Out for honest metrics
- **Speaker-Disjoint**: Zero contamination between train/validation
- **Realistic Expectations**: 36-44% generalization is scientifically honest

## File Structure

```
├── execute_three_stage_pipeline.py    # Master execution script
├── train_grid_pretrain.py             # Stage 1: GRID pretraining
├── train_icu_finetune.py              # Stage 2: ICU fine-tuning
├── calibrate.py                       # Stage 3: Personalization
├── utils/
│   └── viseme_mapper.py               # Viseme mapping system
├── tools/
│   ├── organize_speaker_data.py       # Data organization
│   ├── build_grid_manifest.py         # GRID manifest builder
│   └── select_grid_subset.py          # GRID subset selection
├── data/
│   ├── speaker sets/                  # ICU organized data
│   └── grid/                          # GRID corpus (optional)
├── checkpoints/
│   ├── grid_pretraining/              # Stage 1 checkpoints
│   ├── icu_finetuning/               # Stage 2 checkpoints
│   └── personalization/               # Stage 3 checkpoints
└── manifests/                         # Data manifests
```

## Configuration Options

### Pipeline Configuration
- `--grid-epochs`: GRID pretraining epochs (default: 30)
- `--icu-epochs`: ICU fine-tuning epochs (default: 20)
- `--personalization-epochs`: Personalization epochs (default: 5)
- `--batch-size`: Training batch size (default: 16)
- `--words-per-class`: GRID words per ICU class (default: 20)
- `--k-shot`: Examples per class for personalization (default: 10)

### Stage Skipping
- `--skip-grid`: Skip GRID pretraining
- `--skip-icu`: Skip ICU fine-tuning
- `--skip-personalization`: Skip personalization

## Technical Details

### Model Architecture
- **Base**: EnhancedLightweightCNNLSTM (787K parameters)
- **Input**: 32 frames × 64×48 grayscale video
- **Encoder**: 3D CNN + LSTM for temporal modeling
- **Classifier**: Lightweight head for class prediction

### Data Augmentation
- **Conservative**: Brightness ±10%, contrast 0.9-1.1x, horizontal flip
- **Temporal**: Speed variations 0.95-1.05x
- **Preservation**: Maintains lip visibility for visual features

### Training Strategy
- **GRID**: Multi-class word classification (20+ classes)
- **ICU**: 4-class phrase classification with transfer learning
- **Personalization**: Head-only fine-tuning with frozen encoder

## Monitoring and Logging

### Execution Logging
- Comprehensive step-by-step logging
- Error handling and recovery
- Performance metrics tracking
- Final execution report

### Checkpointing
- Best model saving per stage
- Training history preservation
- Resume capability for interrupted training

## Next Steps

1. **Execute Pipeline**: Run complete three-stage training
2. **Evaluate Performance**: Review LOSO and personalization metrics
3. **Deploy Models**: Use personalized models for bedside testing
4. **Collect Feedback**: Monitor real-world performance
5. **Iterate**: Refine based on clinical validation results

## Support

For questions or issues:
1. Check execution logs in respective output directories
2. Review individual stage documentation
3. Verify data organization and prerequisites
4. Monitor training metrics and convergence

---

**Note**: This pipeline maintains scientific integrity by clearly separating honest generalization metrics (36-44% LOSO) from personalized performance (>90% within-speaker). Both metrics are valuable for different aspects of clinical deployment.
