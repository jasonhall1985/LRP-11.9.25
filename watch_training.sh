#!/bin/bash

# Simple script to watch training progress in real-time

echo "🔍 TRAINING PROGRESS MONITOR"
echo "=============================="
echo "Press Ctrl+C to stop monitoring"
echo ""

# Find the latest experiment directory
EXPERIMENT_DIR=$(ls -t training_experiment_* 2>/dev/null | head -1)

if [ -z "$EXPERIMENT_DIR" ]; then
    echo "❌ No training experiment found"
    exit 1
fi

echo "📁 Monitoring: $EXPERIMENT_DIR"
echo "📄 Log file: $EXPERIMENT_DIR/training.log"
echo ""

# Watch the log file in real-time
if [ -f "$EXPERIMENT_DIR/training.log" ]; then
    echo "📊 LIVE TRAINING LOG:"
    echo "===================="
    tail -f "$EXPERIMENT_DIR/training.log"
else
    echo "⏳ Waiting for training to start..."
    while [ ! -f "$EXPERIMENT_DIR/training.log" ]; do
        sleep 2
    done
    echo "📊 LIVE TRAINING LOG:"
    echo "===================="
    tail -f "$EXPERIMENT_DIR/training.log"
fi
