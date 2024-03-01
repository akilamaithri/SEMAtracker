import cv2
import numpy as np
import pygame
import time

# Load the pre-trained face detection model
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
fullbody_cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")

pygame.init()

# Load audio tracks
audio_left = pygame.mixer.Sound("fire-1.mp3")
audio_right = pygame.mixer.Sound("water-1.mp3") 

# Capture video from webcam
cap = cv2.VideoCapture(0)

# State variables for tracking face position and audio playback
current_position = None
last_audio_played = None

# Check if webcam is opened successfully
if not cap.isOpened():
    print("Error opening video capture object")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale (optional)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    faces = fullbody_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Focus on the first detected face (if any)
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Take only the first face
        center_x = int(x + w / 2)
        half_width = frame.shape[1] / 2

        new_position = 'right' if center_x > half_width else 'left'

        # Audio playback based on position change and playback state
        if new_position != current_position and last_audio_played != new_position:
            current_position = new_position
            last_audio_played = new_position

             # Stop any currently playing audio
            pygame.mixer.stop()
            
            if new_position == 'right':
                audio_right.play()  # Play only once for new position
                print("right")
            else:
                audio_left.play()  # Play only once for new position
                print("left")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# # Loop through each detected face
# for (x, y, w, h) in faces:
#     # Calculate center point coordinates
#     center_x = int(x + w / 2)
#     center_y = int(y + h / 2)

#     # Draw a circle at the center point (optional)
#     cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

# # Loop through each detected face
# for (x, y, w, h) in faces:
#     # Calculate center point
#     center_x = int(x + w / 2)
#     half_width = frame.shape[1] / 2

#     # Play audio based on position
#     if center_x > half_width:
#         audio_right.play()
#     else:
#         audio_left.play()

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
pygame.quit()