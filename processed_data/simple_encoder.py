import numpy as np

class SimpleLabelEncoder:
    def __init__(self, classes):
        self.classes_ = np.array(classes)
    
    def transform(self, labels):
        return [list(self.classes_).index(label) for label in labels]
    
    def inverse_transform(self, indices):
        return [self.classes_[idx] for idx in indices]
