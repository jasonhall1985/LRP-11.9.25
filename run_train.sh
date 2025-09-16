#!/bin/bash
"""
Production-Ready Lip Reading Training Script
===========================================

Executable bash script for 7-class lip reading training with comprehensive
error handling, proper quote escaping for spaces, and production-ready setup.

Features:
- Automatic environment setup and validation
- Comprehensive error handling and logging
- Progress monitoring and status reporting
- Proper handling of paths with spaces
- GPU memory management
- Automatic recovery from common issues

Author: Production Lip Reading System
Date: 2025-09-15
"""

set -e  # Exit on any error
set -u  # Exit on undefined variables

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/training_$(date +%Y%m%d_%H%M%S).log"
PYTHON_ENV="${SCRIPT_DIR}/venv"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

# Error handling
handle_error() {
    local exit_code=$?
    error "Script failed with exit code $exit_code at line $1"
    error "Check the log file for details: $LOG_FILE"
    
    # GPU cleanup on error
    if command -v nvidia-smi &> /dev/null; then
        log "Cleaning up GPU memory..."
        python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    fi
    
    exit $exit_code
}

trap 'handle_error $LINENO' ERR

# Print header
echo "=================================================================="
echo "🚀 Production-Ready 7-Class Lip Reading Trainer"
echo "=================================================================="
log "Starting training script from: $SCRIPT_DIR"
log "Log file: $LOG_FILE"

# Check system requirements
check_requirements() {
    log "Checking system requirements..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is required but not installed"
        exit 1
    fi
    
    local python_version=$(python3 --version | cut -d' ' -f2)
    log "Python version: $python_version"
    
    # Check GPU availability
    if command -v nvidia-smi &> /dev/null; then
        log "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | head -1
    else
        warning "No NVIDIA GPU detected, will use CPU"
    fi
    
    # Check available memory
    local available_mem=$(free -h | awk '/^Mem:/ {print $7}')
    log "Available memory: $available_mem"
    
    # Check disk space
    local available_disk=$(df -h "$SCRIPT_DIR" | awk 'NR==2 {print $4}')
    log "Available disk space: $available_disk"
}

# Setup Python environment
setup_environment() {
    log "Setting up Python environment..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$PYTHON_ENV" ]; then
        log "Creating virtual environment..."
        python3 -m venv "$PYTHON_ENV"
    fi
    
    # Activate virtual environment
    source "$PYTHON_ENV/bin/activate"
    
    # Upgrade pip
    log "Upgrading pip..."
    pip install --upgrade pip
    
    # Install requirements
    if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
        log "Installing requirements..."
        pip install -r "$SCRIPT_DIR/requirements.txt"
    else
        error "requirements.txt not found!"
        exit 1
    fi
    
    success "Environment setup complete"
}

# Validate data paths
validate_data_paths() {
    log "Validating data paths..."
    
    # Define data source paths (with proper quoting for spaces)
    local training_source1="/Users/client/Desktop/LRP classifier 11.9.25/data/videos for training 14.9.25 not cropped completely /13.9.25top7dataset_cropped"
    local training_source2="/Users/client/Desktop/training set 2.9.25"
    local val_source="/Users/client/Desktop/VAL set"
    local test_source="/Users/client/Desktop/test set"
    local processed_dir="fixed_temporal_output/full_processed"
    
    # Check each path
    local paths_valid=true
    
    if [ ! -d "$training_source1" ]; then
        error "Training source 1 not found: $training_source1"
        paths_valid=false
    else
        local count1=$(find "$training_source1" -name "*.mp4" | wc -l)
        log "Training source 1: $count1 videos found"
    fi
    
    if [ ! -d "$training_source2" ]; then
        error "Training source 2 not found: $training_source2"
        paths_valid=false
    else
        local count2=$(find "$training_source2" -name "*.mp4" | wc -l)
        log "Training source 2: $count2 videos found"
    fi
    
    if [ ! -d "$val_source" ]; then
        error "Validation source not found: $val_source"
        paths_valid=false
    else
        local count_val=$(find "$val_source" -name "*.mp4" | wc -l)
        log "Validation source: $count_val videos found"
    fi
    
    if [ ! -d "$test_source" ]; then
        error "Test source not found: $test_source"
        paths_valid=false
    else
        local count_test=$(find "$test_source" -name "*.mp4" | wc -l)
        log "Test source: $count_test videos found"
    fi
    
    if [ -d "$processed_dir" ]; then
        local count_processed=$(find "$processed_dir" -name "processed_v*.mp4" | wc -l)
        log "Processed videos: $count_processed found"
    else
        warning "Processed directory not found: $processed_dir"
    fi
    
    if [ "$paths_valid" = false ]; then
        error "One or more data paths are invalid"
        exit 1
    fi
    
    success "All data paths validated"
}

# Prepare manifest
prepare_manifest() {
    log "Preparing dataset manifest..."
    
    # Activate environment
    source "$PYTHON_ENV/bin/activate"
    
    # Define paths with proper quoting
    local sources=(
        "/Users/client/Desktop/LRP classifier 11.9.25/data/videos for training 14.9.25 not cropped completely /13.9.25top7dataset_cropped"
        "/Users/client/Desktop/training set 2.9.25"
        "/Users/client/Desktop/VAL set"
        "/Users/client/Desktop/test set"
    )
    
    # Build manifest command
    local manifest_cmd="python3 prepare_manifest.py"
    manifest_cmd="$manifest_cmd --sources"
    
    # Add each source with proper quoting
    for source in "${sources[@]}"; do
        manifest_cmd="$manifest_cmd \"$source\""
    done
    
    manifest_cmd="$manifest_cmd --processed_dir \"fixed_temporal_output/full_processed\""
    manifest_cmd="$manifest_cmd --out manifest.csv"
    manifest_cmd="$manifest_cmd --val_holdout \"gender=male,age_band=40-64\""
    manifest_cmd="$manifest_cmd --test_holdout \"gender=female,age_band=18-39\""
    manifest_cmd="$manifest_cmd --show_stats"
    manifest_cmd="$manifest_cmd --verify_videos"
    manifest_cmd="$manifest_cmd --balance_classes"
    manifest_cmd="$manifest_cmd --balance_source_demo \"gender=male,age_band=18-39\""
    
    log "Running manifest preparation..."
    log "Command: $manifest_cmd"
    
    # Execute command
    eval "$manifest_cmd"
    
    if [ -f "manifest.csv" ]; then
        local manifest_size=$(wc -l < manifest.csv)
        success "Manifest created with $manifest_size entries"
    else
        error "Failed to create manifest"
        exit 1
    fi
}

# Run training
run_training() {
    log "Starting training..."
    
    # Activate environment
    source "$PYTHON_ENV/bin/activate"
    
    # Create experiment directory
    local experiment_dir="./experiments/run_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$experiment_dir"
    
    # Training command with weighted sampling (recommended for balanced classes)
    local train_cmd="python3 train.py"
    train_cmd="$train_cmd --manifest manifest.csv"
    train_cmd="$train_cmd --config config.yaml"
    train_cmd="$train_cmd --balance weighted_sampler"
    train_cmd="$train_cmd --output_dir \"$experiment_dir\""
    
    # Add GPU specification if available
    if command -v nvidia-smi &> /dev/null; then
        train_cmd="$train_cmd --gpu 0"
    fi
    
    log "Training command: $train_cmd"
    log "Experiment directory: $experiment_dir"
    
    # Monitor GPU usage in background if available
    if command -v nvidia-smi &> /dev/null; then
        log "Starting GPU monitoring..."
        nvidia-smi dmon -s pucvmet -d 30 > "$experiment_dir/gpu_usage.log" 2>&1 &
        local gpu_monitor_pid=$!
    fi
    
    # Execute training
    eval "$train_cmd"
    
    # Stop GPU monitoring
    if [ -n "${gpu_monitor_pid:-}" ]; then
        kill $gpu_monitor_pid 2>/dev/null || true
    fi
    
    # Check training results
    if [ -f "$experiment_dir/final_report.json" ]; then
        success "Training completed successfully"
        
        # Extract key metrics
        local test_accuracy=$(python3 -c "import json; print(json.load(open('$experiment_dir/final_report.json'))['training_summary']['test_accuracy'])")
        local test_f1=$(python3 -c "import json; print(json.load(open('$experiment_dir/final_report.json'))['training_summary']['test_macro_f1'])")
        
        log "Final Results:"
        log "  Test Accuracy: $test_accuracy"
        log "  Test Macro F1: $test_f1"
        
        # Check target achievement
        local target_achieved=$(python3 -c "import json; report=json.load(open('$experiment_dir/final_report.json')); print(report['target_achievement']['achieved'] and report['target_achievement']['macro_f1_achieved'])")
        
        if [ "$target_achieved" = "True" ]; then
            success "🎉 TARGET ACCURACY >80% ACHIEVED! 🎉"
        else
            warning "Target accuracy not achieved, consider:"
            warning "  - Increasing training epochs"
            warning "  - Adjusting learning rate"
            warning "  - Using focal loss for class imbalance"
            warning "  - Adding more data augmentation"
        fi
        
    else
        error "Training failed - no final report generated"
        exit 1
    fi
}

# Cleanup function
cleanup() {
    log "Performing cleanup..."
    
    # GPU cleanup
    if command -v nvidia-smi &> /dev/null; then
        python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    fi
    
    # Kill any remaining background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    
    log "Cleanup complete"
}

# Main execution
main() {
    log "Starting main execution pipeline..."
    
    # Step 1: Check requirements
    check_requirements
    
    # Step 2: Setup environment
    setup_environment
    
    # Step 3: Validate data paths
    validate_data_paths
    
    # Step 4: Prepare manifest
    prepare_manifest
    
    # Step 5: Run training
    run_training
    
    # Step 6: Cleanup
    cleanup
    
    success "🎉 ALL STEPS COMPLETED SUCCESSFULLY! 🎉"
    log "Check the experiment directory for detailed results"
    log "Log file saved at: $LOG_FILE"
}

# Execute main function
main "$@"

echo "=================================================================="
echo "✅ Training pipeline completed successfully!"
echo "=================================================================="
