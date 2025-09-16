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
