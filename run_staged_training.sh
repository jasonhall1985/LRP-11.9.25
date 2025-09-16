#!/bin/bash

# Staged Fine-Tuning with Overfitting Recovery
# Resume from best checkpoint with enhanced regularization

echo "🚀 Starting Staged Fine-Tuning for Lip-Reading"
echo "================================================"
echo "Target: >80% generalization accuracy"
echo "Approach: Progressive unfreezing with corrected overfitting handling"
echo ""

# Create experiment directory
EXPERIMENT_DIR="experiments/run_002_staged"
mkdir -p "$EXPERIMENT_DIR"

# Copy configuration for reproducibility
cp config_staged.yaml "$EXPERIMENT_DIR/"
cp clean_balanced_manifest.csv "$EXPERIMENT_DIR/"

echo "📁 Experiment directory: $EXPERIMENT_DIR"
echo "📊 Manifest: clean_balanced_manifest.csv (2,079 videos, 297 per class)"
echo "🔄 Resume from: experiments/run_001/best_model.pth"
echo ""

# Launch training with caffeinate to prevent sleep
echo "🎯 Launching staged training with comprehensive safety guardrails..."
echo "⏰ Training will run overnight with automatic monitoring"
echo ""

caffeinate -dimsu nohup python train.py \
  --config config_staged.yaml \
  --manifest clean_balanced_manifest.csv \
  --resume experiments/run_001/best_model.pth \
  --output_dir "$EXPERIMENT_DIR" \
  > "$EXPERIMENT_DIR/train.log" 2>&1 &

# Get the process ID
TRAIN_PID=$!
echo "🔧 Training process ID: $TRAIN_PID"
echo "📝 Log file: $EXPERIMENT_DIR/train.log"
echo ""

# Create monitoring script
cat > "$EXPERIMENT_DIR/monitor.sh" << 'EOF'
#!/bin/bash
EXPERIMENT_DIR="experiments/run_002_staged"
LOG_FILE="$EXPERIMENT_DIR/train.log"

echo "📊 Monitoring Staged Training Progress"
echo "======================================"
echo "Log file: $LOG_FILE"
echo ""

# Function to show last N lines with timestamp
show_progress() {
    if [ -f "$LOG_FILE" ]; then
        echo "🕐 $(date): Latest training progress:"
        echo "----------------------------------------"
        tail -20 "$LOG_FILE" | grep -E "(Epoch|Stage|Train|Val|✓|⚠️|🎯|🔄)" || echo "No recent progress found"
        echo ""
    else
        echo "❌ Log file not found: $LOG_FILE"
    fi
}

# Function to check for completion
check_completion() {
    if [ -f "$LOG_FILE" ]; then
        if grep -q "Training completed" "$LOG_FILE"; then
            echo "✅ Training completed successfully!"
            grep -A 10 "FINAL TEST RESULTS" "$LOG_FILE" || echo "Final results not found"
            return 0
        elif grep -q "Early stopping triggered" "$LOG_FILE"; then
            echo "🛑 Training stopped early"
            grep -A 5 "Early stopping triggered" "$LOG_FILE"
            return 0
        elif grep -q "Error\|Exception\|Failed" "$LOG_FILE" | tail -1; then
            echo "❌ Training encountered errors:"
            grep -A 3 -B 1 "Error\|Exception\|Failed" "$LOG_FILE" | tail -10
            return 1
        fi
    fi
    return 2  # Still running
}

# Show initial progress
show_progress

# Check completion status
check_completion
status=$?

if [ $status -eq 2 ]; then
    echo "🔄 Training is still running..."
    echo "💡 Use 'tail -f $LOG_FILE' to follow live progress"
    echo "💡 Use 'bash $EXPERIMENT_DIR/monitor.sh' to check status"
fi

echo ""
echo "📈 Key files to monitor:"
echo "  - Training log: $LOG_FILE"
echo "  - Best model: $EXPERIMENT_DIR/best_model.pth"
echo "  - Training history: $EXPERIMENT_DIR/training_history.json"
echo "  - Final report: $EXPERIMENT_DIR/final_report.txt"
EOF

chmod +x "$EXPERIMENT_DIR/monitor.sh"

echo "🔍 Monitoring tools created:"
echo "  - Monitor script: $EXPERIMENT_DIR/monitor.sh"
echo "  - Live log: tail -f $EXPERIMENT_DIR/train.log"
echo ""

echo "🎯 STAGED TRAINING SCHEDULE:"
echo "  Stage A - Linear Probe (1 epoch):"
echo "    ✓ Freeze entire backbone, train only head"
echo "    ✓ Learning rate: head=3e-4"
echo ""
echo "  Stage B - Partial Unfreezing (2-3 epochs):"
echo "    ✓ Unfreeze layer4 only"
echo "    ✓ Differential LR: head=2e-4, backbone=1e-5"
echo ""
echo "  Stage C - Full Fine-tuning (3-6 epochs):"
echo "    ✓ Unfreeze layer3 + layer4"
echo "    ✓ Continue with head=2e-4, backbone=1e-5"
echo ""

echo "🛡️ CORRECTED OVERFITTING RESPONSE:"
echo "  ✓ INCREASE dropout: 0.4 → 0.5"
echo "  ✓ MAINTAIN augmentation levels"
echo "  ✓ HALVE head learning rate"
echo "  ✓ KEEP backbone LR at 1e-5"
echo ""

echo "📊 ENHANCED INPUT PROCESSING:"
echo "  ✓ Grayscale conversion with cv2.COLOR_BGR2GRAY"
echo "  ✓ Channel replication to 3 channels"
echo "  ✓ Spatial resize: 96×96 → 112×112"
echo "  ✓ Kinetics-400 normalization"
echo ""

echo "🎯 TARGET: >80% generalization accuracy"
echo "⏱️  Expected duration: 2-4 hours"
echo ""

# Show initial status
sleep 2
echo "📊 Initial status check:"
bash "$EXPERIMENT_DIR/monitor.sh"

echo ""
echo "🚀 Staged training launched successfully!"
echo "💡 Check progress with: bash $EXPERIMENT_DIR/monitor.sh"
echo "💡 Follow live log with: tail -f $EXPERIMENT_DIR/train.log"
