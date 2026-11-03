import csv
import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from features import frame_features, window_to_arr

T = 30
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")
CSV_PATH = os.path.join(BASE_DIR, "seq_data.csv")
KEYS = {ord("j"): "J", ord("z"): "Z", ord("0"): "NONE", ord("h"): "HELLO"}

base = BaseOptions(model_asset_path=HAND_MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)

fbase = BaseOptions(model_asset_path=FACE_MODEL_PATH)
pbase = BaseOptions(model_asset_path=POSE_MODEL_PATH)
foptions = vision.FaceLandmarkerOptions(
    base_options=fbase,
    running_mode=vision.RunningMode.VIDEO,
    num_faces = 1
)

poptions = vision.PoseLandmarkerOptions(
    base_options=pbase,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)

hands = vision.HandLandmarker.create_from_options(options)
face = vision.FaceLandmarker.create_from_options(foptions)
pose = vision.PoseLandmarker.create_from_options(poptions)

camera = cv2.VideoCapture(0)
f = open(CSV_PATH, "a", newline="")
writer = csv.writer(f)

counts = {"J": 0, "Z": 0, "NONE": 0, "HELLO": 0}
rec_label, window = None, []
MAX_MISS = 5
miss = 0

while True:
    ok, frame = camera.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = hands.detect_for_video(img, int(time.time() * 1000))

    item = None
    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        for p in hand:
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 5, (0, 255, 0), 1)
        miss = 0
        item = frame_features(hand)
    elif rec_label and window and miss < MAX_MISS:
        miss += 1
        item = window[-1]          # hand lost briefly: repeat last good frame
    elif rec_label:
        rec_label, window, miss = None, [], 0   # lost too long: discard clip

    if rec_label and item is not None:
        window.append(item)
        if len(window) == T:
            writer.writerow([rec_label, *window_to_arr(window).flatten()])
            counts[rec_label] += 1
            rec_label, window, miss = None, [], 0

    status = f"REC {rec_label} {len(window)}/{T}" if rec_label else "idle"
    cv2.putText(frame, f"{status}  {counts}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Collect sequences", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("1"):
        break
    if key in KEYS and not rec_label:
        rec_label, window, miss = KEYS[key], [], 0

f.close()
print(counts)
camera.release()
cv2.destroyAllWindows()
hands.close()