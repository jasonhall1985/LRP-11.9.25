#!/usr/bin/env python3
"""
Create label encoder for the lipreading app
"""

import pickle
import json
import os
import numpy as np

# Create directories
os.makedirs('processed_data', exist_ok=True)

# Target words
target_words = ["doctor", "glasses", "help", "pillow", "phone"]

# Simple label encoder class
class SimpleLabelEncoder:
    def __init__(self, classes):
        self.classes_ = np.array(classes)
    
    def transform(self, labels):
        return [list(self.classes_).index(label) for label in labels]
    
    def inverse_transform(self, indices):
        return [self.classes_[idx] for idx in indices]

# Create and save label encoder
label_encoder = SimpleLabelEncoder(target_words)

with open('processed_data/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Save label mapping as JSON
label_mapping = {str(i): word for i, word in enumerate(target_words)}
with open('processed_data/label_mapping.json', 'w') as f:
    json.dump(label_mapping, f, indent=2)

print("✅ Label encoder created and saved")
print(f"Target words: {target_words}")
print("Files created:")
print("  - processed_data/label_encoder.pkl")
print("  - processed_data/label_mapping.json")
