import cv2
import mediapipe as mp
import numpy as np
import os

DATASET_PATH = "dataset/ISL/"
SAVE_PATH = "dataset_landmarks/"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

os.makedirs(SAVE_PATH, exist_ok=True)

GESTURES = os.listdir(DATASET_PATH)

for gesture in GESTURES:
    print("Processing:", gesture)

    gesture_folder = os.path.join(DATASET_PATH, gesture)
    save_folder = os.path.join(SAVE_PATH, gesture)

    os.makedirs(save_folder, exist_ok=True)

    for img_name in os.listdir(gesture_folder):
        img_path = os.path.join(gesture_folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            landmarks = []

            for lm in result.multi_hand_landmarks[0].landmark:
                landmarks.append([lm.x, lm.y, lm.z])

            landmarks = np.array(landmarks).flatten()
            save_name = img_name.replace(".jpg", ".npy").replace(".png", ".npy")

            np.save(os.path.join(save_folder, save_name), landmarks)

print("DONE! All landmarks extracted.")
