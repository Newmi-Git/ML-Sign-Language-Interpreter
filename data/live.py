import cv2
import mediapipe as mp
import time
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision 
import os
import numpy as np
from features import extract_features
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task")
print(MODEL_PATH, os.path.exists(MODEL_PATH))


camera = cv2.VideoCapture(0)
base = BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options = base,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = vision.HandLandmarker.create_from_options(options)

while True:
    success, frame = camera.read()

    if not success:
        print("couldnt access camera")
        break
    
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame )
    timestamp = int(time.time() * 1000)
    result = hands.detect_for_video(mp_image, timestamp)
    if result.hand_landmarks:
        features = extract_features(result.hand_landmarks[0])
        print("hands:", len(result.hand_landmarks), "features:", len(features))
        for hand in result.hand_landmarks:
            for point in hand:
                px = int(point.x * w)
                py = int(point.y * h)
                cv2.circle(frame, (px, py), 5, (0, 255, 0),1)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xff == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()
hands.close()

