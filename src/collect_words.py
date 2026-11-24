import csv
import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from features import frame_vec, resample

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "words.csv")

REC_SECONDS = 2.0     # must match WINDOW_MS in App.vue

# Edit this list. Only add signs you have checked with a SASL source / Deaf signer.
VOCAB = ["HELLO", "HOW-ARE-YOU", "FINE", "THANK-YOU", "PLEASE", "SORRY",
         "HELP", "WATER", "TOILET", "YES", "NO", "NONE"]

SESSION = input("Session / signer name (e.g. newmi_1): ").strip() or "s1"

hand_lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=os.path.join(HERE, "models", "hand_landmarker.task")),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
))
pose_lm = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=os.path.join(HERE, "models", "pose_landmarker.task")),
    running_mode=vision.RunningMode.VIDEO,
))

camera = cv2.VideoCapture(0)
out = open(CSV_PATH, "a", newline="")
writer = csv.writer(out)

current, recording, clip, rec_start = 0, False, [], 0
saved = {v: 0 for v in VOCAB}

while True:
    ok, frame = camera.read()
    if not ok:
        print("couldn't access camera")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts = int(time.time() * 1000)
    hr = hand_lm.detect_for_video(img, ts)
    pr = pose_lm.detect_for_video(img, ts)

    left = right = None
    for lm, side in zip(hr.hand_landmarks, hr.handedness):
        if side[0].category_name == "Left":
            left = lm
        else:
            right = lm
    pose = pr.pose_landmarks[0] if pr.pose_landmarks else None

    label = VOCAB[current]

    if recording:                              # keep every frame for exactly 1 second
        clip.append(frame_vec(left, right, pose))
        if time.time() - rec_start >= REC_SECONDS:
            recording = False
            writer.writerow([label, SESSION] + resample(clip).round(5).flatten().tolist())
            out.flush()
            saved[label] += 1
    h, w = frame.shape[:2]
    for hand, dot_colour in ((left, (255, 255, 0)), (right, (0, 255, 255))):
        for p in hand or []:                   # all 21 points of each hand
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 3, dot_colour, -1)
    if pose:                                   # nose, mouth, shoulders, elbows, wrists
        for i in (0, 9, 10, 11, 12, 13, 14, 15, 16):
            cv2.circle(frame, (int(pose[i].x * w), int(pose[i].y * h)), 5, (0, 255, 0), -1)

    colour = (0, 0, 255) if recording else (255, 255, 255)
    text = f"{label}  saved:{saved[label]}  {'REC ' + str(len(clip)) if recording else ''}"
    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
    cv2.putText(frame, "r=record 1s  n=next  p=prev  q=quit", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("collect words", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("n"):
        current = (current + 1) % len(VOCAB)
    elif key == ord("p"):
        current = (current - 1) % len(VOCAB)
    elif key == ord("r") and not recording:
        recording, clip, rec_start = True, [], time.time()

camera.release()
cv2.destroyAllWindows()
out.close()