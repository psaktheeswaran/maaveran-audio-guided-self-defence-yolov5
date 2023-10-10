import cv2
import numpy as np
import torch
import tkinter as tk
import pygame
from pathlib import Path

# Initialize Tkinter
root = tk.Tk()
root.title("YOLOv5 Object Detection")

# Initialize pygame for audio playback
pygame.mixer.init()

# Load YOLOv5 model
weights_path = Path("best5s.pt")  # Replace with the path to your downloaded weights
model = torch.load(weights_path, map_location='cpu')['model'].float()
model.eval()

# Define class labels
class_labels = ["class1", "class2", "class3"]  # Replace with your class labels

# Define MP3 files for each class
mp3_files = {
    "class1": "d.mp3",
    "class2": "re.mp3",
    "class3": "nd.mp3"
}

# Create a label for displaying detection results
label = tk.Label(root, text="", font=("Helvetica", 16))
label.pack()

# Open the webcam (you can replace this with your video source)
cap = cv2.VideoCapture(0)

# Initialize flags for each class
class_flags = {class_label: False for class_label in class_labels}

# Define confidence threshold and NMS threshold
confidence_threshold = 0.5  # Adjust as needed
nms_threshold = 0.4  # Adjust as needed

def update_label(text):
    label.config(text=text)

def play_mp3(class_name):
    mp3_file = mp3_files[class_name]
    pygame.mixer.music.load(mp3_file)
    pygame.mixer.music.play()

def detect_objects():
    ret, frame = cap.read()

    # Perform object detection
    results = model(frame)

    # Filter detections based on class labels and confidence threshold
    detections = results.pred[0]
    mask = (detections[:, 4] > confidence_threshold) & (detections[:, 5].long() < len(class_labels))
    filtered_detections = detections[mask]

    if len(filtered_detections) > 0:
        # Apply non-maximum suppression
        filtered_detections = filtered_detections[filtered_detections[:, 4].argsort(descending=True)]
        keep = torch.ops.torchvision.nms(filtered_detections[:, :4], filtered_detections[:, 4], nms_threshold)
        filtered_detections = filtered_detections[keep]

        # Check if any of the filtered detections match a specific class
        for detection in filtered_detections:
            class_index = int(detection[5])
            class_name = class_labels[class_index]
            if not class_flags[class_name]:
                play_mp3(class_name)
                class_flags[class_name] = True

    # Draw bounding boxes on the frame
    results.render()[0]

    root.after(10, detect_objects)  # Continuously detect objects

detect_objects()

root.mainloop()

