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

# Initialize pygame for playing sounds
pygame.mixer.init()

# Load your sound files for each class (adjust the filenames accordingly)
sound_files = [
    pygame.mixer.Sound("re.mp3"),
    pygame.mixer.Sound("d.mp3"),
    pygame.mixer.Sound("nd.mp3")
]

# Specify the folders where the images are located
image_folders = [
    "/home/josva/olo trashh/yolov5trash-master/runs/detect/exp14/crops/Metal",
    "/home/josva/olo trashh/yolov5trash-master/runs/detect/exp12/crops/Carton",
    "/home/josva/olo trashh/yolov5trash-master/runs/detect/exp12/crops/Plastico"
]

sound_play_time = 3.0  # Time to play the sound (in seconds)
sound_pause_time = 0.3  # Time to pause before playing the sound again (in seconds)
last_sound_time = 0

while True:
    detected_class = None

    for i, image_folder in enumerate(image_folders):
        # List all JPG files in the current image folder
        image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]

        # Sort the image files by modification time and get the most recently modified one
        if image_files:
            latest_image = max(image_files, key=lambda x: os.path.getmtime(os.path.join(image_folder, x)))
            image_path = os.path.join(image_folder, latest_image)

            # Load the most recently modified image
            image = cv2.imread(image_path)

            # Resize the image into (224-height,224-width) pixels
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

            # Make the image a numpy array and reshape it to the model's input shape.
            image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

            # Normalize the image array
            image = (image / 127.5) - 1

            # Predict the model
            prediction = model.predict(image)
            index = np.argmax(prediction)

            if index in [0, 1, 2]:
                detected_class = index  # Map class index to the corresponding folder

    current_time = time.time()

    # Check if it's time to play the sound again
    if current_time - last_sound_time >= sound_play_time and detected_class is not None:
        sound_files[detected_class].play()  # Play the corresponding sound

        # Update the last sound time
        last_sound_time = current_time

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

cv2.destroyAllWindows()

