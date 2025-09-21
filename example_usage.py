#!/usr/bin/env python3
"""
📋 EXAMPLE USAGE - 75.9% Checkpoint
Simple example showing how to use the restored checkpoint
"""

import torch
import numpy as np
from load_75_9_checkpoint import load_checkpoint

def main():
    print("🚀 EXAMPLE: Using the 75.9% Checkpoint")
    print("=" * 50)
    
    # Load the checkpoint
    model, class_to_idx, idx_to_class, checkpoint = load_checkpoint()
    
    print(f"\n📊 Model Information:")
    print(f"   Architecture: DoctorFocusedModel")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Classes: {len(class_to_idx)}")
    print(f"   Input shape: (batch, 1, 32, 64, 96)")
    print(f"   Output shape: (batch, 4)")
    
    print(f"\n🎯 Class Mapping:")
    for class_name, idx in class_to_idx.items():
        print(f"   {idx}: {class_name}")
    
    # Example 1: Single prediction
    print(f"\n🧪 Example 1: Single Video Prediction")
    print("-" * 30)
    
    # Simulate a video input (32 frames, 64x96 grayscale)
    video_input = torch.randn(1, 1, 32, 64, 96)  # Random video for demo
    
    model.eval()
    with torch.no_grad():
        output = model(video_input)
        probabilities = torch.softmax(output, dim=1)
        predicted_idx = torch.argmax(output, dim=1).item()
        predicted_class = idx_to_class[predicted_idx]
        confidence = probabilities[0, predicted_idx].item()
    
    print(f"   Input shape: {video_input.shape}")
    print(f"   Predicted class: {predicted_class}")
    print(f"   Confidence: {confidence:.3f}")
    print(f"   All probabilities:")
    for i, prob in enumerate(probabilities[0]):
        print(f"     {idx_to_class[i]}: {prob:.3f}")
    
    # Example 2: Batch prediction
    print(f"\n🧪 Example 2: Batch Prediction")
    print("-" * 30)
    
    batch_size = 3
    batch_input = torch.randn(batch_size, 1, 32, 64, 96)
    
    with torch.no_grad():
        batch_output = model(batch_input)
        batch_probabilities = torch.softmax(batch_output, dim=1)
        batch_predictions = torch.argmax(batch_output, dim=1)
    
    print(f"   Batch size: {batch_size}")
    print(f"   Batch predictions:")
    for i in range(batch_size):
        pred_class = idx_to_class[batch_predictions[i].item()]
        confidence = batch_probabilities[i, batch_predictions[i]].item()
        print(f"     Video {i+1}: {pred_class} (confidence: {confidence:.3f})")
    
    # Example 3: Model information
    print(f"\n📋 Example 3: Checkpoint Information")
    print("-" * 30)
    
    if 'epoch' in checkpoint:
        print(f"   Training epoch: {checkpoint['epoch']}")
    if 'val_accuracy' in checkpoint:
        print(f"   Validation accuracy: {checkpoint['val_accuracy']:.2f}%")
    
    print(f"   Model state keys: {len(checkpoint['model_state_dict'])}")
    print(f"   Available checkpoint keys: {list(checkpoint.keys())}")
    
    print(f"\n✅ Examples completed successfully!")
    print(f"🎯 The model is ready for:")
    print(f"   • Further training/fine-tuning")
    print(f"   • Real video inference")
    print(f"   • Extension to more classes")
    print(f"   • Production deployment")

if __name__ == "__main__":
    main()
