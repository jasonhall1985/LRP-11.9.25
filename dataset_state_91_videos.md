# Dataset State: 91 Videos (Reverted)

## Overview
Successfully reverted the dataset back to the exact state it was in when we achieved **57.14% validation accuracy** with the OptimizedLipReadingCNN model.

## Current Dataset Composition
- **Total Videos:** 91
- **Original Videos:** 66 (gentle_v5 preprocessing)
- **First-Round Augmented:** 25 (lighting-only augmentations)
- **Second-Round Augmented:** 0 (removed)

## Class Distribution
```
doctor  : 20 videos (15 original +  5 augmented)
glasses : 16 videos (12 original +  4 augmented)
help    : 20 videos (15 original +  5 augmented)
phone   : 16 videos (12 original +  4 augmented)
pillow  : 19 videos (12 original +  7 augmented)
```

## Actions Performed
1. ✅ Removed 18 second-round augmented videos (aug_round2_*)
2. ✅ Cleaned up second-round augmentation files:
   - `second_round_augmentation.py`
   - `second_round_augmentation_log.json`
   - `expanded_dataset_trainer.py`
   - `best_expanded_dataset_model.pth`
3. ✅ Preserved the 57.14% accuracy model: `best_augmented_lip_reading_model.pth`
4. ✅ Maintained all original and first-round augmented videos

## Preserved Model
- **File:** `best_augmented_lip_reading_model.pth`
- **Validation Accuracy:** 57.14%
- **Architecture:** OptimizedLipReadingCNN
- **Dataset:** 91 videos (66 original + 25 first-round augmented)
- **Training History:** `augmented_training_history.json`

## Dataset Ready For
- ✅ Adding new original videos from additional sources
- ✅ Expanding with fresh content rather than more augmentations
- ✅ Maintaining the proven 91-video baseline for comparison

## File Structure
```
data/training set 17.9.25/
├── [66 original videos] *_gentle_v5.npy
├── [25 augmented videos] *_aug_*.npy (first-round only)
├── preview_videos/
└── preview_videos_fixed/
```

## Next Steps
The dataset is now ready for you to:
1. Add new original videos from additional sources
2. Process them with the same gentle_v5 preprocessing pipeline
3. Expand the dataset with fresh content
4. Compare performance against the 57.14% baseline model

The dataset is in the exact state that produced our best model performance.
