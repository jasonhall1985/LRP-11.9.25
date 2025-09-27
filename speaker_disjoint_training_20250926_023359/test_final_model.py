import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import sys
import os

# Define model architecture directly
class LightweightCNNLSTM(nn.Module):
    def __init__(self, num_classes=4, dropout=0.4):
        super(LightweightCNNLSTM, self).__init__()

        self.conv3d1 = nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=1)
        self.bn3d1 = nn.BatchNorm3d(16)
        self.pool3d1 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.conv3d2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1)
        self.bn3d2 = nn.BatchNorm3d(32)
        self.pool3d2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3d3 = nn.Conv3d(32, 48, kernel_size=(3, 3, 3), padding=1)
        self.bn3d3 = nn.BatchNorm3d(48)
        self.pool3d3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 6))
        self.lstm_input_size = 48 * 4 * 6
        self.lstm_hidden_size = 128
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.lstm_hidden_size,
                           num_layers=1, batch_first=True, dropout=0.0)

        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.lstm_hidden_size, 64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout * 0.75)
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.bn3d1(self.conv3d1(x)))
        x = self.pool3d1(x)
        x = torch.relu(self.bn3d2(self.conv3d2(x)))
        x = self.pool3d2(x)
        x = torch.relu(self.bn3d3(self.conv3d3(x)))
        x = self.pool3d3(x)
        x = self.adaptive_pool(x)

        batch_size = x.size(0)
        timesteps = x.size(2)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.contiguous().view(batch_size, timesteps, -1)

        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.dropout1(x)
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout2(x)
        x = self.fc_out(x)
        return x

def test_model():
    # Load test manifest
    test_df = pd.read_csv('test_manifest.csv')

    # Load model
    model = LightweightCNNLSTM()
    checkpoint = torch.load('best_model.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Class mapping
    class_to_idx = {'doctor': 0, 'i_need_to_move': 1, 'my_mouth_is_dry': 2, 'pillow': 3}
    
    predictions = []
    labels = []
    
    print(f"Testing on {len(test_df)} videos...")
    
    with torch.no_grad():
        for _, row in test_df.iterrows():
            try:
                # Load preprocessed video
                video_data = np.load(row['file_path'])
                video_tensor = torch.FloatTensor(video_data).unsqueeze(0).unsqueeze(0)
                
                # Get prediction
                outputs = model(video_tensor)
                _, predicted = torch.max(outputs, 1)
                
                predictions.append(predicted.item())
                labels.append(class_to_idx[row['class']])
                
            except Exception as e:
                print(f"❌ Error processing {row['file_path']}: {e}")
    
    if len(predictions) > 0:
        accuracy = accuracy_score(labels, predictions) * 100
        print(f"\n📈 FINAL TEST RESULTS:")
        print(f"   Test Accuracy: {accuracy:.2f}%")
        print(f"   Videos tested: {len(predictions)}")
        
        # Classification report
        target_names = ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']
        print(f"\n📋 Classification Report:")
        print(classification_report(labels, predictions, target_names=target_names))
        
        return accuracy
    else:
        print("❌ No valid predictions made")
        return 0.0

if __name__ == "__main__":
    test_accuracy = test_model()
    
    if test_accuracy >= 70.0:
        print(f"\n🎉 SUCCESS: Test accuracy {test_accuracy:.2f}% meets deployment criteria!")
    else:
        print(f"\n⚠️  WARNING: Test accuracy {test_accuracy:.2f}% below deployment threshold")
