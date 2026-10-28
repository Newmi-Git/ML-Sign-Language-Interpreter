import cv2
import mediapipe as mp
import time
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision 
import os
import torch
from collections import Counter, deque
from features import extract_features
from model import build_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")
CKPT_PATH = os.path.join(BASE_DIR, "models", "letters.pt")

CONF_THRESHOLD = 0.8
HISTORY = 8
MIN_AGREE = 6

ckpt = torch.load(CKPT_PATH)
labels = ckpt["labels"]
clf = build_model(len(labels))
clf.load_state_dict(ckpt["state"])
clf.eval()

camera = cv2.VideoCapture(0)
base = BaseOptions(model_asset_path=HAND_MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options = base,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = vision.HandLandmarker.create_from_options(options)

recent = deque(maxlen=HISTORY)
shown = ""

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
    conf_text = ""
    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        features = extract_features(hand)

        with torch.no_grad():
            x = torch.tensor([features], dtype=torch.float32)
            probs = torch.softmax(clf(x), dim=1)[0]
            conf, idx = probs.max(0)

        if conf.item() >= CONF_THRESHOLD:
            recent.append(labels[idx.item()])
        else:
            recent.append(None)

        letter, count = Counter(recent).most_common(1)[0]
        shown = letter if (letter is not None and count >= MIN_AGREE) else ""
        conf_text = f"{conf.item():.2f}"

        for point in hand:
            cv2.circle(frame, (int(point.x * w), int(point.y * h)), 5, (0, 255, 0), 1)
    else:
        recent.clear()
        shown = ""
 
    cv2.putText(frame, f"{shown}  {conf_text}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xff == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()
hands.close()

