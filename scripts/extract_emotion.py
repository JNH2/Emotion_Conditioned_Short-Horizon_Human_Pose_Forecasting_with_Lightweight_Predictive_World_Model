import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
import os

VIDEO_PATH = "data/sample7.mp4"
OUTPUT_PATH = "data/emotion_seq7.npy"

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False)

cap = cv2.VideoCapture(VIDEO_PATH)

emotion_sequence = []

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total frames:", frame_count)

for _ in tqdm(range(frame_count)):

    ret, frame = cap.read()

    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:

        landmarks = results.multi_face_landmarks[0]

        features = []

        for lm in landmarks.landmark[0:10]:
            features.append(lm.x)
            features.append(lm.y)

        emotion_sequence.append(features)

    else:

        emotion_sequence.append(np.zeros(20))

cap.release()

emotion_sequence = np.array(emotion_sequence)

print("Emotion shape:", emotion_sequence.shape)

np.save(OUTPUT_PATH, emotion_sequence)

print("Saved to:", OUTPUT_PATH)