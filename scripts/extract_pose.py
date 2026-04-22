import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

VIDEO_PATH = "sample_anger.mp4"
OUTPUT_PATH = "sample.npy"

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

cap = cv2.VideoCapture(VIDEO_PATH)

pose_sequence = []

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total frames:", frame_count)

for _ in tqdm(range(frame_count)):
    ret, frame = cap.read()

    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(frame_rgb)

    if results.pose_landmarks:

        joints = []

        for lm in results.pose_landmarks.landmark:
            joints.append([lm.x, lm.y])

        pose_sequence.append(joints)

    else:
        pose_sequence.append(np.zeros((33, 2)))

cap.release()

pose_sequence = np.array(pose_sequence)

print("Pose shape:", pose_sequence.shape)

np.save(OUTPUT_PATH, pose_sequence)

print("Saved to:", OUTPUT_PATH)