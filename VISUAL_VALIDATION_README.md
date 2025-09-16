# 🔍 Visual Validation Tool for Lip Reading Dataset

Comprehensive visual validation tool that samples videos across all data splits and classes, extracts representative frames, and generates an interactive HTML visualization to verify dataset quality before training.

## ✨ Features

- **📊 Proportional Sampling**: Automatically samples 40 videos across train/val/test splits and all 7 classes
- **🎯 Demographic Diversity**: Ensures representation across gender, age bands, and ethnicities
- **🖼️ Frame Extraction**: Extracts middle frame from each video to avoid edge artifacts
- **🔍 Quality Validation**: Comprehensive checks for dimensions, format, contrast, and CLAHE processing
- **📈 Interactive Visualization**: Responsive HTML grid with filtering and detailed metadata
- **🌐 Auto Browser Display**: Automatically opens results in default browser
- **📋 Comprehensive Reports**: JSON and Markdown reports with detailed statistics

## 🎯 What It Validates

### Technical Specifications
- ✅ **Exact Dimensions**: Verifies all frames are exactly 96×96 pixels
- ✅ **Grayscale Format**: Confirms single-channel grayscale images
- ✅ **Pixel Value Range**: Checks for proper dynamic range (0-255)
- ✅ **Contrast Metrics**: Analyzes RMS contrast and Michelson contrast
- ✅ **CLAHE Enhancement**: Detects if contrast enhancement was applied

### Data Quality Checks
- ✅ **Mouth ROI Visibility**: Visual inspection of lip/mouth region clarity
- ✅ **Consistent Cropping**: Ensures uniform mouth positioning across videos
- ✅ **Lighting Consistency**: Verifies CLAHE normalization effectiveness
- ✅ **Processing Version**: Shows V3 > V2 > original priority handling

### Dataset Distribution
- ✅ **Class Balance**: Shows distribution across all 7 classes
- ✅ **Split Distribution**: Validates train/val/test demographic splits
- ✅ **Demographic Coverage**: Ensures diversity in gender, age, ethnicity

## 🚀 Quick Start

### 1. Automated Validation (Recommended)
```bash
# Run complete validation pipeline
./run_visual_validation.sh

# Custom sample size
./run_visual_validation.sh manifest.csv 60
```

### 2. Manual Validation
```bash
# Basic validation with 40 samples
python visual_validation.py --manifest manifest.csv --open-browser

# Custom configuration
python visual_validation.py \
  --manifest manifest.csv \
  --samples 60 \
  --output ./custom_validation \
  --open-browser
```

### 3. Test the Tool
```bash
# Run with synthetic test data
python test_visual_validation.py
```

## 📊 Sample Selection Strategy

The tool uses intelligent sampling to ensure comprehensive coverage:

### Proportional Split Sampling
- **Training Set**: ~70% of samples (proportional to actual data size)
- **Validation Set**: ~15% of samples (male 40-64 demographic)
- **Test Set**: ~15% of samples (female 18-39 demographic)

### Class Distribution
- Samples distributed evenly across all 7 classes:
  - `help`
  - `doctor` 
  - `glasses`
  - `phone`
  - `pillow`
  - `i_need_to_move`
  - `my_mouth_is_dry`

### Demographic Diversity
- **Gender**: Male and female representation
- **Age Bands**: 18-39, 40-64, 65+ coverage
- **Ethnicity**: Caucasian, Asian, African, Hispanic diversity
- **Processing**: V3, V2, and original video versions

## 🖼️ Interactive HTML Visualization

### Grid Layout
- **Responsive Design**: Adapts to screen size (8×5 or 10×4 grid)
- **Quality Badges**: Color-coded quality scores (Excellent/Good/Poor)
- **Issue Highlighting**: Red border for videos with problems

### Filtering Options
- **Data Split**: Filter by train/val/test
- **Class**: Filter by specific lip reading class
- **Quality**: Filter by quality score ranges
- **Issues**: Show only videos with/without problems

### Detailed Metadata
Each video card displays:
- **Frame Image**: Representative middle frame (pixelated rendering)
- **File Information**: Filename, path, source dataset
- **Demographics**: Gender, age band, ethnicity
- **Technical Specs**: Dimensions, pixel statistics, contrast metrics
- **Processing Info**: CLAHE status, processed version
- **Quality Issues**: List of any detected problems
- **Pixel Histogram**: Distribution of pixel values

## 📈 Quality Scoring System

### Quality Score (0-100)
- **Dimensions Correct (30 pts)**: Frame is exactly 96×96
- **Grayscale Format (20 pts)**: Single channel image
- **Dynamic Range (20 pts)**: Pixel range > 50 (good contrast)
- **RMS Contrast (20 pts)**: Standard deviation > 20
- **Brightness Range (10 pts)**: Mean pixel value 10-245

### Quality Categories
- **🟢 Excellent (90-100)**: Meets all specifications perfectly
- **🟡 Good (70-89)**: Minor issues, acceptable for training
- **🔴 Poor (<70)**: Significant problems, needs attention

## 📋 Output Files

### Interactive Visualization
- **`visual_validation.html`**: Interactive grid with filtering and metadata
- **Responsive design** with hover effects and detailed tooltips
- **Automatic browser opening** for immediate inspection

### Comprehensive Reports
- **`validation_report.json`**: Complete technical data in JSON format
- **`VALIDATION_SUMMARY.md`**: Human-readable summary with statistics
- **Quality distribution**, **issue analysis**, and **recommendations**

### Sample Report Structure
```
📊 VALIDATION SUMMARY
- Total Videos: 40
- Average Quality: 87.3/100
- Issues Found: 3/40 (7.5%)

🎯 QUALITY DISTRIBUTION
- Excellent: 28 videos (70%)
- Good: 9 videos (22.5%)
- Poor: 3 videos (7.5%)

⚙️ TECHNICAL VALIDATION
- Correct Dimensions: 37/40 (92.5%)
- Grayscale Format: 40/40 (100%)
- CLAHE Enhancement: 35/40 (87.5%)
```

## 🔧 Common Issues and Solutions

### Dimension Problems
```
Issue: Wrong dimensions (112x112 instead of 96x96)
Solution: Check video preprocessing pipeline
```

### Poor Contrast
```
Issue: Low RMS contrast (<20)
Solution: Verify CLAHE enhancement is applied
```

### Missing Demographics
```
Issue: Unknown gender/age/ethnicity
Solution: Check manifest demographic parsing
```

### Processing Version Issues
```
Issue: Using original instead of V3 processed
Solution: Verify processed video availability
```

## 🎛️ Configuration Options

### Command Line Arguments
```bash
--manifest PATH          # Dataset manifest CSV file
--samples N              # Number of videos to sample (default: 40)
--output DIR             # Output directory (default: ./validation_output)
--open-browser           # Auto-open HTML in browser
--seed N                 # Random seed for reproducibility (default: 42)
```

### Sampling Parameters
- **Total Samples**: 1-200 videos (recommended: 40-60)
- **Random Seed**: For reproducible sampling across runs
- **Proportional Splits**: Maintains actual dataset proportions

## 📊 Expected Results

### Healthy Dataset Indicators
- ✅ **>95% Correct Dimensions**: All frames are 96×96
- ✅ **100% Grayscale**: All frames single-channel
- ✅ **>80% CLAHE Applied**: Consistent contrast enhancement
- ✅ **Average Quality >85**: High overall quality scores
- ✅ **<10% Issues**: Minimal quality problems

### Warning Signs
- ⚠️ **<90% Correct Dimensions**: Preprocessing issues
- ⚠️ **<70% CLAHE Applied**: Inconsistent enhancement
- ⚠️ **Average Quality <70**: Significant quality problems
- ⚠️ **>20% Issues**: Dataset needs attention

## 🧪 Testing

### Run Test Suite
```bash
# Test with synthetic data
python test_visual_validation.py
```

The test creates 15 synthetic videos with various quality characteristics and runs the full validation pipeline to ensure everything works correctly.

## 🎯 Integration with Training Pipeline

### Pre-Training Validation
```bash
# 1. Create manifest
python prepare_manifest.py [options]

# 2. Validate quality
./run_visual_validation.sh

# 3. Review results before training
# Check HTML visualization for any issues

# 4. Proceed with training if validation passes
./run_train.sh
```

### Quality Gates
- **Minimum 95% correct dimensions** before training
- **Average quality score >80** for optimal results
- **Visual inspection** of mouth ROI positioning
- **CLAHE consistency** across all samples

---

**🎯 Ensure dataset quality before committing to training with comprehensive visual validation!**
