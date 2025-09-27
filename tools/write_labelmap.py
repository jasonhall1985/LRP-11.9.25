#!/usr/bin/env python3
"""
Create a global label map to ensure consistent label indexing across all splits.
"""
import json
import os

def main():
    # Fixed canonical labels
    labels = sorted(["doctor", "i_need_to_move", "my_mouth_is_dry", "pillow"])
    
    # Create label to index mapping
    label2idx = {lab: i for i, lab in enumerate(labels)}
    
    # Ensure checkpoints directory exists
    os.makedirs("checkpoints", exist_ok=True)
    
    # Save global label map
    with open("checkpoints/label2idx.json", "w") as f:
        json.dump(label2idx, f, indent=2)
    
    print("Global label map created:")
    for label, idx in label2idx.items():
        print(f"  {idx}: {label}")
    
    print(f"\nSaved to: checkpoints/label2idx.json")

if __name__ == "__main__":
    main()
