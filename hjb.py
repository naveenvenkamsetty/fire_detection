import cv2
import torch
import numpy as np
import threading
import pygame
import smtplib
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Initialize pygame mixer
pygame.mixer.init()
Email_Status=False

def send_mail_function():
    recipientEmail="firedetection.ucen@gmail.com"
    recipientEmail=recipientEmail.lower()
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login("venkamsetty.naveen@gmail.com","naveen@2002")
        server.sendmail("venkamsetty.naveen@gmail.com",recipientEmail,"Warning A Fire Accident has been reported on company ABC")
        print("sent to {}".format(recipientEmail))
        server.close
    except Exception as e:
        print (e)

# Load pre-trained Faster R-CNN model
faster_rcnn_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
faster_rcnn_model.eval()

# Load pre-trained ResNet model for feature extraction
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
resnet.eval()

# Define preprocessing transform for the input frame
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((800, 800)),  # Resize to match the expected input size of the model
    transforms.ToTensor(),
])

# Load the trained LSTM model
class LSTMClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

lstm_model = LSTMClassifier(input_size=2048, hidden_size=128, num_layers=2, num_classes=2)
lstm_model.load_state_dict(torch.load('trained_lstm_model.pth'))  # Load the trained model
lstm_model.eval()

# Function to detect suspected fire regions using Faster R-CNN
def detect_fire_regions(frame):
    input_image = preprocess(frame)
    input_image = input_image.unsqueeze(0)
    with torch.no_grad():
        predictions = faster_rcnn_model(input_image)
    # Extract bounding boxes and labels from predictions
    boxes = predictions[0]['boxes'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    # Filter out suspected fire regions
    fire_boxes = boxes[labels == 1]  # Assuming fire class label is 1
    return fire_boxes

# Function to extract features from detected fire regions using ResNets
def extract_features(frame, fire_boxes):
    fire_features = []
    for box in fire_boxes:
        x1, y1, x2, y2 = map(int, box)
        
        # Resize the bounding box coordinates to match the resized frame
        scale_x = frame.shape[1] / 800  # Original width / Resized width
        scale_y = frame.shape[0] / 800  # Original height / Resized height
        x1_resized = int(x1 * scale_x)
        y1_resized = int(y1 * scale_y)
        x2_resized = int(x2 * scale_x)
        y2_resized = int(y2 * scale_y)
        
        fire_region = frame[y1_resized:y2_resized, x1_resized:x2_resized]
        fire_region = cv2.resize(fire_region, (224, 224))  # Resize to fit ResNet input size
        fire_region = torch.tensor(fire_region, dtype=torch.float32).permute(2, 0, 1) / 255.0
        # Detach the tensor before converting it to a numpy array
        fire_features.append(resnet(fire_region.unsqueeze(0)).squeeze().detach().numpy())
    return fire_features

# Function to classify fire regions using the trained LSTM model
def classify_fire_regions(fire_features):
    fire_features_array = np.array(fire_features)  # Convert the list of numpy arrays into a single numpy array
    fire_features_tensor = torch.tensor(fire_features_array, dtype=torch.float32)
    # Ensure the input tensor has the correct shape (batch_size, seq_len, input_size)
    fire_features_tensor = fire_features_tensor.unsqueeze(0)  # Add a batch dimension
    with torch.no_grad():
        outputs = lstm_model(fire_features_tensor)
    _, predicted = torch.max(outputs, 1)
    return predicted.numpy()

# Function to perform majority voting on fire region classifications
def majority_voting(classifications):
    # Perform majority voting to determine the final fire detection result
    # You can customize the voting strategy based on your requirements
    return np.argmax(np.bincount(classifications))

# Function to play the emergency alarm sound
def play_emergency_alarm_sound():
    pygame.mixer.music.load("alarm-sound.mp3")
    pygame.mixer.music.play()

# Function to draw bounding boxes around detected fire regions
def draw_fire_boxes(frame, fire_boxes):
    for box in fire_boxes:
        x1, y1, x2, y2 = map(int, box)
        
        # Draw bounding box around the fire region
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, "Fire Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Function to process video frames
def process_video(video_path):
    global Email_Status
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        fire_boxes = detect_fire_regions(frame)
        if len(fire_boxes) > 0:
            fire_features = extract_features(frame, fire_boxes)
            classifications = classify_fire_regions(fire_features)
            fire_detection_result = majority_voting(classifications)
            if fire_detection_result == 1:
                # Fire detected, trigger emergency alarm
                threading.Thread(target=play_emergency_alarm_sound).start()

                if Email_Status == False:
                    threading.Thread(target=send_mail_function).start()
                    Email_Status = True
            
            # Draw bounding boxes around fire regions
            draw_fire_boxes(frame, fire_boxes)
            
        # Display the processed frame
        cv2.imshow('Processed Frame', frame)
        
        # Check if 'q' key is pressed to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_path = 'fire3.mp4'  # Replace with the path to your video
Email_Status=False
process_video(video_path)
