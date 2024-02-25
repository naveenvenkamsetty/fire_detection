import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from PIL import Image

# Define your LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Initialize hidden state and cell state
        batch_size, seq_len, input_size = x.size()
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        # Forward pass
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Assuming you want to use the last output for classification
        return out

# Function to extract features from video frames
def extract_features_from_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    features = []
    labels = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Extract features from frame using your model (Faster R-CNN, MobileNet, etc.)
        feature_vector = extract_features(frame)  # Replace extract_features with your feature extraction function
        features.append(feature_vector)
        # Assuming you have a function to determine if the frame contains fire or not
        label = determine_label(frame)  # Replace determine_label with your label determination function
        labels.append(label)
    cap.release()
    return np.array(features), np.array(labels)

# Function to save features and labels to numpy files
def save_data(features, labels, features_path, labels_path):
    np.save(features_path, features)
    np.save(labels_path, labels)

# Function to load features and labels from numpy files
def load_data(features_path, labels_path):
    features = np.load(features_path)
    labels = np.load(labels_path)
    return features, labels

# Define paths to save features and labels
features_path = 'features.npy'
labels_path = 'labels.npy'

# Extract features and labels from video frames
video_path = 'fire1.mp4'
features, labels = extract_features_from_frames(video_path)

# Save features and labels to numpy files
save_data(features, labels, features_path, labels_path)

# Load features and labels from numpy files
X_train, y_train = load_data(features_path, labels_path)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Use torch.long for classification

# Create a TensorDataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

# Define batch size for training
batch_size = 32

# Create DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Define model parameters
input_size = X_train.shape[2]  # Feature size
hidden_size = 128
num_layers = 2
num_classes = 2  # Binary classification (fire or no fire)

# Initialize your LSTM model
lstm_model = LSTMModel(input_size, hidden_size, num_layers, num_classes)

# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        # Forward pass
        outputs = lstm_model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
if not os.path.exists('models'):
    os.makedirs('models')
torch.save(lstm_model.state_dict(), 'models/trained_lstm_model.pth')
