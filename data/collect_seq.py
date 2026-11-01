import csv
import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from features import frame_features, window_to_array

T = 30
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")
CSV_PATH = os.path.join(BASE_DIR, "seq_data.csv")
KEYS = {ord("j"): "J", ord("z"): "Z", ord("n"): "N"}

base = BaseOptions(model_asset_path=HAND_MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
hands = vision.HandLandmarker.create_from_options(options)
camera = cv2.VideoCapture(0)
f = open(CSV_PATH, "a", newline="")
writer = csv.writer(f)

counts = {"J": 0, "Z": 0, "N": 0}
rec_label, window = None, []

while True:
    ok, frame = camera.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = hands.detect_for_video(img, int(time.time() * 1000))

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        for p in hand:
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 5, (0, 255, 0), 1)
        if rec_label:
            window.append(frame_features(hand))
            if len(window) == T:
                writer.writerow([rec_label, *window_to_array(window).flatten()])
                counts[rec_label] += 1
                rec_label, window = None, []
    elif rec_label:
        rec_label, window = None, []  # hand lost, discard clip

    status = f"REC {rec_label} {len(window)}/{T}" if rec_label else "idle"
    cv2.putText(frame, f"{status}  {counts}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Collect sequences", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("1"):
        break
    if key in KEYS and not rec_label:
        rec_label, window = KEYS[key], []

f.close()
print(counts)
camera.release()
cv2.destroyAllWindows()
hands.close()