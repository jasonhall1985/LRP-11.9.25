#!/bin/bash
"""
Visual Validation Runner Script
==============================

Automated script to run visual validation on the lip reading dataset.
This script will:
1. Check if manifest exists, create it if needed
2. Run visual validation with 40 representative samples
3. Automatically open results in browser
4. Display summary of findings

Usage:
  ./run_visual_validation.sh [manifest_path] [sample_count]

Examples:
  ./run_visual_validation.sh                    # Use default manifest.csv, 40 samples
  ./run_visual_validation.sh manifest.csv 60    # Custom manifest and sample count
"""

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MANIFEST="manifest.csv"
DEFAULT_SAMPLES=40

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
MANIFEST_PATH="${1:-$DEFAULT_MANIFEST}"
SAMPLE_COUNT="${2:-$DEFAULT_SAMPLES}"

echo "=================================================================="
echo "🔍 Visual Validation for Lip Reading Dataset"
echo "=================================================================="
log "Manifest: $MANIFEST_PATH"
log "Sample count: $SAMPLE_COUNT"
log "Working directory: $SCRIPT_DIR"

# Check if manifest exists
if [ ! -f "$MANIFEST_PATH" ]; then
    warning "Manifest not found: $MANIFEST_PATH"
    log "Creating manifest first..."
    
    # Check if prepare_manifest.py exists
    if [ ! -f "prepare_manifest.py" ]; then
        error "prepare_manifest.py not found! Please create the manifest first."
        exit 1
    fi
    
    # Run manifest preparation with default settings
    log "Running manifest preparation..."
    python3 prepare_manifest.py \
        --sources \
        "/Users/client/Desktop/LRP classifier 11.9.25/data/videos for training 14.9.25 not cropped completely /13.9.25top7dataset_cropped" \
        "/Users/client/Desktop/training set 2.9.25" \
        "/Users/client/Desktop/VAL set" \
        "/Users/client/Desktop/test set" \
        --processed_dir "fixed_temporal_output/full_processed" \
        --out "$MANIFEST_PATH" \
        --val_holdout "gender=male,age_band=40-64" \
        --test_holdout "gender=female,age_band=18-39" \
        --show_stats \
        --verify_videos
        
    if [ ! -f "$MANIFEST_PATH" ]; then
        error "Failed to create manifest"
        exit 1
    fi
    
    success "Manifest created successfully"
fi

# Check if visual_validation.py exists
if [ ! -f "visual_validation.py" ]; then
    error "visual_validation.py not found!"
    exit 1
fi

# Check Python dependencies
log "Checking Python dependencies..."
python3 -c "import cv2, pandas, numpy, matplotlib, seaborn, PIL" 2>/dev/null || {
    warning "Some Python dependencies may be missing"
    log "Installing required packages..."
    pip3 install opencv-python pandas numpy matplotlib seaborn pillow
}

# Run visual validation
log "Starting visual validation..."
log "This will sample $SAMPLE_COUNT videos across all splits and classes"

python3 visual_validation.py \
    --manifest "$MANIFEST_PATH" \
    --samples "$SAMPLE_COUNT" \
    --output "./validation_output" \
    --open-browser \
    --seed 42

# Check if validation completed successfully
if [ $? -eq 0 ]; then
    success "Visual validation completed successfully!"
    
    # Display quick summary if files exist
    if [ -f "./validation_output/VALIDATION_SUMMARY.md" ]; then
        echo ""
        echo "=================================================================="
        echo "📊 QUICK SUMMARY"
        echo "=================================================================="
        head -20 "./validation_output/VALIDATION_SUMMARY.md" | tail -15
        echo ""
        echo "Full report available at: ./validation_output/VALIDATION_SUMMARY.md"
        echo "Interactive visualization: ./validation_output/visual_validation.html"
    fi
    
    echo ""
    echo "=================================================================="
    echo "✅ VALIDATION COMPLETE!"
    echo "=================================================================="
    echo "📊 HTML Visualization: ./validation_output/visual_validation.html"
    echo "📋 JSON Report: ./validation_output/validation_report.json"
    echo "📝 Summary: ./validation_output/VALIDATION_SUMMARY.md"
    echo ""
    echo "The HTML visualization should have opened in your browser."
    echo "If not, manually open: ./validation_output/visual_validation.html"
    echo "=================================================================="
    
else
    error "Visual validation failed!"
    exit 1
fi
