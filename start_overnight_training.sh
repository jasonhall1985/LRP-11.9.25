#!/bin/bash

# Robust Overnight Training Script for Mac
# This script will keep your Mac awake during training and handle all logging

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== LIP READING OVERNIGHT TRAINING ===${NC}"
echo -e "${BLUE}Starting robust training with comprehensive safety guardrails${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "train.py" ]; then
    echo -e "${RED}Error: train.py not found. Please run this script from the project directory.${NC}"
    exit 1
fi

# Check if manifest exists
if [ ! -f "clean_balanced_manifest.csv" ]; then
    echo -e "${RED}Error: clean_balanced_manifest.csv not found.${NC}"
    exit 1
fi

# Create experiments directory if it doesn't exist
mkdir -p experiments/run_001

# Find latest checkpoint
CHECKPOINT=""
if ls experiments/run_001/checkpoint_epoch_*.pth 1> /dev/null 2>&1; then
    CHECKPOINT=$(ls -t experiments/run_001/checkpoint_epoch_*.pth | head -n1)
    echo -e "${GREEN}Found latest checkpoint: $CHECKPOINT${NC}"
elif [ -f "experiments/run_001/best_model.pth" ]; then
    CHECKPOINT="experiments/run_001/best_model.pth"
    echo -e "${YELLOW}Using best model checkpoint: $CHECKPOINT${NC}"
else
    echo -e "${YELLOW}No checkpoint found, starting from scratch${NC}"
fi

# Build the training command
CMD="python train.py \
  --manifest clean_balanced_manifest.csv \
  --unfreeze_layers layer3,layer4 \
  --head_lr 2e-4 --backbone_lr 2e-5 \
  --weight_decay 0.01 --dropout 0.3 --label_smoothing 0.05 \
  --clip_len 32 --ema_beta 0.999 --early_stop_patience 6 \
  --balance weighted_sampler --aug temporal10,affine2,bc0.1 \
  --save_dir experiments/run_001"

# Add resume if checkpoint exists
if [ ! -z "$CHECKPOINT" ]; then
    CMD="$CMD --resume $CHECKPOINT"
fi

echo -e "${BLUE}Training Configuration:${NC}"
echo "  - Architecture: R(2+1)D-18 with layer3+layer4 unfrozen"
echo "  - Learning Rates: head=2e-4, backbone=2e-5"
echo "  - Regularization: dropout=0.3, label_smoothing=0.05, weight_decay=0.01"
echo "  - EMA: β=0.999 for weight averaging"
echo "  - Augmentations: temporal_jitter=10%, affine=2°, brightness/contrast=10%"
echo "  - Early Stopping: patience=6 on VAL macro-F1"
echo "  - Safety: NaN/Inf detection with automatic recovery"
echo "  - Adaptive: Balance switching, sequence length increase, overfitting prevention"
echo ""

echo -e "${BLUE}Safety Guardrails Active:${NC}"
echo "  ✓ NaN/Inf detection with checkpoint recovery"
echo "  ✓ Automatic learning rate reduction (10x) on instability"
echo "  ✓ Balance strategy auto-switching (WeightedSampler → Duplicate)"
echo "  ✓ Sequence length increase (32 → 40 frames) if needed"
echo "  ✓ Overfitting prevention (dropout increase, augmentation reduction)"
echo "  ✓ Comprehensive logging and checkpoint management"
echo "  ✓ Mac keep-alive with caffeinate"
echo ""

# Show the full command
echo -e "${BLUE}Full Command:${NC}"
echo "caffeinate -dimsu $CMD"
echo ""

# Ask for confirmation
read -p "Start overnight training? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Training cancelled by user${NC}"
    exit 0
fi

echo -e "${GREEN}Starting overnight training...${NC}"
echo -e "${GREEN}Logs will be saved to: experiments/run_001/train.log${NC}"
echo -e "${GREEN}Monitor progress with: tail -f experiments/run_001/train.log${NC}"
echo ""

# Start training with caffeinate to keep Mac awake
# Redirect all output to log file
exec caffeinate -dimsu nohup $CMD > experiments/run_001/train.log 2>&1 &

# Get the PID of the background process
TRAIN_PID=$!

echo -e "${GREEN}Training started successfully!${NC}"
echo -e "${GREEN}Process ID: $TRAIN_PID${NC}"
echo -e "${GREEN}Your Mac will stay awake during training${NC}"
echo ""
echo -e "${BLUE}To monitor progress:${NC}"
echo "  tail -f experiments/run_001/train.log"
echo ""
echo -e "${BLUE}To stop training:${NC}"
echo "  kill $TRAIN_PID"
echo ""
echo -e "${YELLOW}Training is now running in the background.${NC}"
echo -e "${YELLOW}You can safely close this terminal.${NC}"
echo -e "${YELLOW}Check experiments/run_001/train.log for progress updates.${NC}"
echo ""
echo -e "${GREEN}Sweet dreams! 🌙 Your model will be training overnight.${NC}"
