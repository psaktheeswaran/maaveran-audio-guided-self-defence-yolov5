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
sound_class1 = pygame.mixer.Sound("re.mp3")
sound_class2 = pygame.mixer.Sound("nd.mp3")
sound_class3 = pygame.mixer.Sound("d.mp3")

# CAMERA can be 0 or 1 based on the default camera of your computer
camera = cv2.VideoCapture(0)

sound_play_time = 3.0  # Time to play the sound (in seconds)
sound_pause_time = 0.3  # Time to pause before playing the sound again (in seconds)
last_sound_time = 0

while True:
    # Grab the webcam's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the model's input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predict the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    current_time = time.time()

    # Check if it's time to play the sound again
    if current_time - last_sound_time >= sound_play_time:
        # Play a sound based on the detected class
        if index == 0:
            sound_class1.play()
        elif index == 1:
            sound_class2.play()
        elif index == 2:
            sound_class3.play()

        # Update the last sound time
        last_sound_time = current_time

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()

