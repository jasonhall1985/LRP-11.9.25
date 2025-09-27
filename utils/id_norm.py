"""
ID normalization utilities to fix speaker leakage and label inconsistencies.
"""
import re

def norm_speaker_id(s):
    """Normalize speaker ID by removing extra spaces and converting to lowercase."""
    return re.sub(r"\s+", " ", s.strip().lower())

def norm_label(s):
    """Normalize class label by removing spaces and converting to lowercase."""
    return s.strip().lower().replace(" ", "_")

def validate_no_speaker_overlap(train_speakers, val_speakers):
    """Ensure no speaker overlap between train and validation sets."""
    train_norm = set(norm_speaker_id(s) for s in train_speakers)
    val_norm = set(norm_speaker_id(s) for s in val_speakers)
    overlap = train_norm & val_norm
    if overlap:
        raise ValueError(f"Speaker overlap detected after normalization: {overlap}")
    return True

def validate_label_consistency(train_labels, val_labels, global_labels):
    """Ensure label consistency across train/val and global label map."""
    train_norm = set(norm_label(l) for l in train_labels)
    val_norm = set(norm_label(l) for l in val_labels)
    global_norm = set(norm_label(l) for l in global_labels)
    
    if train_norm != val_norm:
        raise ValueError(f"Train/val label mismatch: train={train_norm}, val={val_norm}")
    if train_norm != global_norm:
        raise ValueError(f"Labels don't match global map: found={train_norm}, expected={global_norm}")
    return True
