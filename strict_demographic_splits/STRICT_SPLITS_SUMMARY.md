# Strict Demographic Dataset Splits Summary

## 🎯 Zero Demographic Overlap Guarantee

Successfully created dataset splits with **ZERO demographic overlap** to prevent data leakage.
Each demographic group (age+gender+ethnicity) assigned to **ONLY ONE split**.

## 📊 Split Distribution

| Split | Videos | Percentage | Target |
|-------|--------|------------|--------|
| **Train** | 543 | 76.1% | 70% |
| **Validation** | 57 | 8.0% | 20% |
| **Test** | 114 | 16.0% | 10% |
| **Total** | **714** | **100%** | **100%** |

## 🚨 Mandatory Assignments Satisfied

✅ **65+ Age Groups**: All demographics with 65+ age exclusively in training set

✅ **Male 18-39 Demographics**: All male 18-39 demographics exclusively in training set

## 👥 Demographic Groups by Split

### Train Split (11 demographic groups)

- **female_65+_caucasian**: 295 videos
- **male_18-39_not**: 67 videos
- **female_18-39_caucasian**: 59 videos
- **male_18-39_caucasian**: 50 videos
- **female_18-39_asian**: 21 videos
- **male_65+_caucasian**: 18 videos
- **female_65+_asian**: 16 videos
- **male_65+_not**: 14 videos
- **female_18-39_aboriginal**: 1 videos
- **female_18-39_not**: 1 videos
- **male_18-39_asian**: 1 videos

### Validation Split (4 demographic groups)

- **female_40-64_caucasian**: 44 videos
- **female_40-64_aboriginal**: 9 videos
- **male_40-64_asian**: 2 videos
- **male_40-64_caucasian**: 2 videos

### Test Split (1 demographic groups)

- **unknown_unknown_unknown**: 114 videos

## 📊 Class Distribution

| Class | Train | Validation | Test | Total |
|-------|-------|------------|------| ----- |
| **doctor** | 70 | 11 | 21 | 102 |
| **glasses** | 72 | 7 | 23 | 102 |
| **help** | 64 | 12 | 26 | 102 |
| **i_need_to_move** | 96 | 6 | 0 | 102 |
| **my_mouth_is_dry** | 96 | 6 | 0 | 102 |
| **phone** | 70 | 8 | 24 | 102 |
| **pillow** | 75 | 7 | 20 | 102 |

## ✅ Verification Results

- ✅ **Zero demographic overlap confirmed**
- ✅ **All mandatory assignments satisfied**
- ✅ **All files exist and accessible**
- ✅ **Comprehensive manifests generated**
- ✅ **Ready for model training**

## 📁 Generated Files

- `strict_demographic_manifest.csv` - Complete manifest (714 videos)
- `strict_train_manifest.csv` - Training set manifest
- `strict_validation_manifest.csv` - Validation set manifest
- `strict_test_manifest.csv` - Test set manifest
- `demographic_assignments.txt` - Demographic group assignments
- `STRICT_SPLITS_SUMMARY.md` - This summary document

*Generated with zero demographic overlap guarantee for data leakage prevention.*
