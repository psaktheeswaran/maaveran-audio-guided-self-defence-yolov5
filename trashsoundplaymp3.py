import os
from keras.models import load_model
import cv2
import numpy as np
import pygame
import time

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Initialize pygame for playing sounds
pygame.mixer.init()

# Load your sound files for each class (adjust the filenames accordingly)
sound_class1 = pygame.mixer.Sound("re.mp3")  # Example sound file
sound_class2 = pygame.mixer.Sound("nd.mp3")  # Example sound file
sound_class3 = pygame.mixer.Sound("d.mp3")   # Example sound file

# Specify the folders where the images are located
image_folder1 = "/home/josva/olo trashh/yolov5trash-master/runs/detect/exp12/crops/Carton"
image_folder2 = os.getcwd()  # Current working directory

sound_play_time = 3.0  # Time to play the sound (in seconds)
sound_pause_time = 0.3  # Time to pause before playing the sound again (in seconds)
last_sound_time = 0

while True:
    # List all JPG files in image_folder1
    image_files1 = [f for f in os.listdir(image_folder1) if f.endswith(".jpg")]

    # List all JPG files in image_folder2
    image_files2 = [f for f in os.listdir(image_folder2) if f.endswith(".jpg")]

    # Sort the image files by modification time and get the most recently modified one
    if image_files1:
        latest_image1 = max(image_files1, key=lambda x: os.path.getmtime(os.path.join(image_folder1, x)))
        image_path1 = os.path.join(image_folder1, latest_image1)

        # Load the most recently modified image from image_folder1
        image1 = cv2.imread(image_path1)

        # Resize the image into (224-height,224-width) pixels
        image1 = cv2.resize(image1, (224, 224), interpolation=cv2.INTER_AREA)

        # Make the image a numpy array and reshape it to the model's input shape.
        image1 = np.asarray(image1, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        image1 = (image1 / 127.5) - 1

        # Predict the model
        prediction1 = model.predict(image1)
        index1 = np.argmax(prediction1)

    if image_files2:
        latest_image2 = max(image_files2, key=lambda x: os.path.getmtime(os.path.join(image_folder2, x)))
        image_path2 = os.path.join(image_folder2, latest_image2)

        # Load the most recently modified image from image_folder2
        image2 = cv2.imread(image_path2)

        # Resize the image into (224-height,224-width) pixels
        image2 = cv2.resize(image2, (224, 224), interpolation=cv2.INTER_AREA)

        # Make the image a numpy array and reshape it to the model's input shape.
        image2 = np.asarray(image2, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        image2 = (image2 / 127.5) - 1

        # Predict the model
        prediction2 = model.predict(image2)
        index2 = np.argmax(prediction2)

    current_time = time.time()

    # Check if it's time to play the sound again
    if current_time - last_sound_time >= sound_play_time:
        # Play a sound based on the detected class for each folder
        if image_files1 and index1 == 0:
            sound_class1.play()
        elif image_files1 and index1 == 1:
            sound_class2.play()
        elif image_files1 and index1 == 2:
            sound_class3.play()

        if image_files2 and index2 == 0:
            sound_class1.play()
        elif image_files2 and index2 == 1:
            sound_class2.play()
        elif image_files2 and index2 == 2:
            sound_class3.play()

        # Update the last sound time
        last_sound_time = current_time

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

cv2.destroyAllWindows()

