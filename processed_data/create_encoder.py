import pickle
import numpy as np

class SimpleLabelEncoder:
    def __init__(self, classes):
        self.classes_ = np.array(classes)

target_words = ["doctor", "glasses", "help", "pillow", "phone"]
encoder = SimpleLabelEncoder(target_words)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

print("Label encoder created")
