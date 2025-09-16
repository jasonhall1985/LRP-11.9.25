#!/bin/bash

# Switch to Python 3.11 environment with MediaPipe
# This script helps you activate the correct environment for the mouth ROI pipeline

echo "Switching to Python 3.11 environment with MediaPipe..."

# Deactivate any current virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Deactivating current environment: $VIRTUAL_ENV"
    deactivate
fi

# Activate the Python 3.11 environment
source .venv311/bin/activate

# Verify the setup
echo "Environment activated!"
echo "Python version: $(python --version)"
echo "MediaPipe status:"
python -c "
try:
    import mediapipe as mp
    print(f'✓ MediaPipe {mp.__version__} is available')
except ImportError:
    print('✗ MediaPipe is not available')
"

echo ""
echo "You can now run the mouth ROI pipeline with MediaPipe support:"
echo "python mouth_roi_pipeline.py --help"
