# Mouth ROI Pipeline

A comprehensive Python tool for standardizing mouth ROIs in videos using **MediaPipe Face Mesh** and auto-re-cropping clips where the lips are too small. Designed for robust processing of lip-reading datasets with fast batching and detailed logging.

## Features

- **MediaPipe Face Mesh Integration**: High-precision lip landmark detection with configurable confidence thresholds
- **Intelligent Size Analysis**: Automatic flagging of videos with too-small mouth ROIs based on area, height, and width ratios
- **Smart Re-cropping**: Adaptive re-cropping strategy for flagged videos to achieve target mouth-to-frame ratios
- **EMA Smoothing**: Exponential Moving Average smoothing for stable bounding boxes across frames
- **Comprehensive Reporting**: Detailed CSV reports with processing statistics and debug visualizations
- **Multi-threaded Processing**: Fast batch processing with configurable worker threads and timeout protection
- **Dry Run Mode**: Analyze-only mode for testing parameters without writing files

## Installation

### Prerequisites

- Python 3.8+
- OpenCV 4.8+
- MediaPipe 0.10+

### Setup

#### Option 1: Python 3.11 with MediaPipe (Recommended)

MediaPipe provides the most accurate face and lip detection. However, it requires Python 3.11 or earlier.

1. **Install Python 3.11** (if not already installed):
   ```bash
   brew install python@3.11  # On macOS with Homebrew
   ```

2. **Create a Python 3.11 virtual environment**:
   ```bash
   /opt/homebrew/opt/python@3.11/bin/python3.11 -m venv .venv311
   ```

3. **Activate the environment**:
   ```bash
   source .venv311/bin/activate
   ```

   Or use the provided script:
   ```bash
   ./switch_to_mediapipe.sh
   ```

4. **Install dependencies**:
   ```bash
   pip install -U pip
   pip install mediapipe opencv-python numpy pandas tqdm
   ```

#### Option 2: Python 3.13 with OpenCV Fallback

If you must use Python 3.13, the pipeline will automatically fall back to OpenCV face detection (less accurate for cropped faces).

1. **Activate your virtual environment**:
   ```bash
   source venv/bin/activate  # Your existing Python 3.13 environment
   ```

2. **Install dependencies**:
   ```bash
   pip install opencv-python numpy pandas tqdm
   ```

The required packages are:
- `mediapipe>=0.10.0` (Python 3.11 only)
- `opencv-python>=4.8.0`
- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `tqdm>=4.65.0`

## Quick Start

### Basic Usage

**Scan + recrop flagged videos (recommended):**
```bash
python mouth_roi_pipeline.py \
  --in_dir '/Users/client/Desktop/LRP classifier 11.9.25/data/videos for training 14.9.25 not cropped completely /13.9.25top7dataset_cropped' \
  --out_dir '/Users/client/Desktop/LRP classifier 11.9.25/data/videos for training 14.9.25 not cropped completely /13.9.25top7dataset_cropped_ROI' \
  --min_area_ratio 0.30 --min_h_ratio 0.40 --min_w_ratio 0.40 \
  --target_h_ratio 0.50 --target_w_ratio 0.50 \
  --out_size 96 --fps_sample 5 --pad 0.12 --workers 6
```

**Scan only (no writes), stricter thresholds:**
```bash
python mouth_roi_pipeline.py \
  --in_dir '/Users/client/Desktop/LRP classifier 11.9.25/data/videos for training 14.9.25 not cropped completely /13.9.25top7dataset_cropped' \
  --out_dir '/Users/client/Desktop/tmp_out' \
  --dry_run --min_area_ratio 0.35 --min_h_ratio 0.45 --min_w_ratio 0.45
```

**Quick summary analysis:**
```bash
python quick_check.py \
  --in_dir '/Users/client/Desktop/LRP classifier 11.9.25/data/videos for training 14.9.25 not cropped completely /13.9.25top7dataset_cropped'
```

## Components

### 1. `mouth_roi_pipeline.py` - Main CLI Tool

The primary pipeline for comprehensive video processing:

**Key Parameters:**
- `--in_dir`: Input directory (default set to your dataset path)
- `--out_dir`: Output directory (required)
- `--min_area_ratio`: Minimum area ratio threshold (default: 0.30)
- `--min_h_ratio`: Minimum height ratio threshold (default: 0.40)
- `--min_w_ratio`: Minimum width ratio threshold (default: 0.40)
- `--target_h_ratio`: Target height ratio for recropping (default: 0.50)
- `--target_w_ratio`: Target width ratio for recropping (default: 0.50)
- `--out_size`: Output video size in pixels (default: 96)
- `--fps_sample`: Frame sampling rate for analysis (default: 5)
- `--pad`: Padding ratio around detected ROI (default: 0.12)
- `--ema`: EMA smoothing factor (default: 0.6)
- `--workers`: Number of worker processes (default: CPU count)
- `--dry_run`: Analyze only, don't write files
- `--verbose`: Enable detailed logging

### 2. `roi_utils.py` - Core Utilities

MediaPipe and geometry helper functions:

**Classes:**
- `MediaPipeLipDetector`: Face mesh-based lip landmark detection
- `ROIGeometry`: Bounding box calculation and size analysis
- `BBoxSmoother`: EMA smoothing for stable tracking
- `RecropCalculator`: Optimal recrop window calculation

**Key Functions:**
- Lip landmark detection using MediaPipe Face Mesh
- Tight bounding box calculation with configurable padding
- Size ratio analysis (area, height, width ratios)
- Debug visualization generation

### 3. `quick_check.py` - Fast Analysis Tool

Lightweight scan-only tool for quick dataset assessment:

**Features:**
- Fast MediaPipe-based analysis without file writing
- Summary statistics and distribution analysis
- Configurable quality thresholds
- Progress tracking with detailed console output

## Processing Pipeline

### 1. **Scan & Analysis**
- Recursively finds supported video files (`.mp4`, `.mov`, `.webm`, `.mkv`)
- Samples frames at specified FPS for analysis
- Detects lip landmarks using MediaPipe Face Mesh
- Calculates tight bounding boxes with padding
- Applies EMA smoothing for stable tracking
- Computes size ratios (area, height, width)

### 2. **Size Classification**
Videos are classified based on median ratios:
- **PASS**: All ratios meet minimum thresholds
- **TOO_SMALL**: One or more ratios below thresholds
- **FAILED**: High detection failure rate or processing errors

### 3. **Output Processing**
- **PASS videos**: Copied to `keep/` (or re-cropped if `--normalize_keep`)
- **TOO_SMALL videos**: Re-cropped to `recrop/` with target ratios
- **FAILED videos**: Copied to `failed/` for manual review

### 4. **Re-cropping Strategy**
For flagged videos:
- Calculate optimal crop window to achieve target ratios
- Apply EMA smoothing to crop window for stability
- Add 5% safety margin for motion compensation
- Resize to specified output size (default: 96x96)

## Output Structure

```
output_directory/
├── keep/           # Videos meeting size requirements
├── recrop/         # Re-cropped videos (originally too small)
├── failed/         # Videos with processing failures
├── debug/          # Debug visualizations (first frame of each video)
└── mouth_roi_report.csv  # Comprehensive processing report
```

## Report Format

The CSV report includes:
- `input_path`: Original video path
- `status`: Processing status (pass/too_small/failed)
- `reason`: Detailed reason for classification
- `width`, `height`, `fps`: Video properties
- `n_frames_sampled`: Number of frames analyzed
- `area_ratio_med`, `h_ratio_med`, `w_ratio_med`: Median size ratios
- `out_path`: Output file path
- `notes`: Additional processing notes

## Advanced Usage

### Custom Thresholds
```bash
# Stricter quality requirements
python mouth_roi_pipeline.py \
  --in_dir INPUT_DIR --out_dir OUTPUT_DIR \
  --min_area_ratio 0.35 --min_h_ratio 0.45 --min_w_ratio 0.45

# More aggressive recropping targets
python mouth_roi_pipeline.py \
  --in_dir INPUT_DIR --out_dir OUTPUT_DIR \
  --target_h_ratio 0.60 --target_w_ratio 0.60
```

### Performance Tuning
```bash
# High-performance processing
python mouth_roi_pipeline.py \
  --in_dir INPUT_DIR --out_dir OUTPUT_DIR \
  --workers 12 --fps_sample 3 --timeout_s 180

# Conservative processing (slower but more robust)
python mouth_roi_pipeline.py \
  --in_dir INPUT_DIR --out_dir OUTPUT_DIR \
  --workers 2 --fps_sample 10 --ema 0.8
```

### Debug Mode
```bash
# Verbose logging with debug visualizations
python mouth_roi_pipeline.py \
  --in_dir INPUT_DIR --out_dir OUTPUT_DIR \
  --verbose --fps_sample 10
```

## Troubleshooting

### Common Issues

1. **MediaPipe Compatibility (Python 3.13)**
   ```
   Warning: MediaPipe not available. Falling back to OpenCV face detection.
   ```
   **Solution**: Use Python 3.11 for full MediaPipe support:
   ```bash
   ./switch_to_mediapipe.sh
   ```
   Or manually activate the Python 3.11 environment:
   ```bash
   source .venv311/bin/activate
   ```

2. **MediaPipe Import Error (Python 3.11)**
   ```bash
   pip install --upgrade mediapipe
   ```

3. **Video Codec Issues**
   ```bash
   pip install opencv-python-headless
   ```

4. **Memory Issues with Large Datasets**
   - Reduce `--workers` count
   - Increase `--timeout_s` value
   - Use `--dry_run` for initial analysis

5. **Path Issues with Spaces**
   - Always use quoted paths: `'/path/with spaces/'`
   - Avoid special characters in directory names

6. **Low Detection Success Rate**
   - Ensure you're using MediaPipe (Python 3.11) for best results
   - OpenCV fallback works poorly with cropped face videos
   - Check that input videos contain visible faces/lips

### Performance Tips

- Use `quick_check.py` first to assess dataset quality
- Start with `--dry_run` to test parameters
- Adjust `--fps_sample` based on video length (shorter videos need higher sampling)
- Use `--workers 1` for debugging individual video issues

## License

This tool is part of the LRP Classifier project. See the main project documentation for licensing information.

## Support

For issues or questions, please refer to the main project documentation or create an issue in the project repository.
