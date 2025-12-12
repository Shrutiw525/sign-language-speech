import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from gtts import gTTS
import playsound
import threading
from collections import deque, Counter
import os

# ------------------- Load Model -------------------
model = tf.keras.models.load_model("sign_model.h5")
label_map = pickle.load(open("label_map.pkl", "rb"))

# ------------------- MediaPipe Hands -------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ------------------- Webcam -------------------
cap = cv2.VideoCapture(0)

# ------------------- Prediction Buffer -------------------
frame_buffer = deque(maxlen=5)
prev_sign = None

# ------------------- TTS Function -------------------
def speak_gtts(text):
    """Convert text to speech and play it in a thread."""
    tts = gTTS(text=text, lang='en')
    filename = "temp_sign.mp3"
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)

# ------------------- Main Loop -------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)  # mirror for left/right hand
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        # Extract landmarks
        landmarks = []
        for lm in result.multi_hand_landmarks[0].landmark:
            landmarks.append([lm.x, lm.y, lm.z])
        landmarks = np.array(landmarks).flatten()

        # Predict sign
        prediction = model.predict(np.array([landmarks]))
        sign = label_map.inverse_transform([np.argmax(prediction)])[0]

        # Add to buffer and get most common (smooth prediction)
        frame_buffer.append(sign)
        most_common_sign = Counter(frame_buffer).most_common(1)[0][0]

        # Draw landmarks
        mp_draw.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        # Display predicted sign
        cv2.putText(frame, most_common_sign.upper(), (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        # Speak the sign if it changed
        if most_common_sign != prev_sign:
            threading.Thread(target=speak_gtts, args=(most_common_sign,), daemon=True).start()
            prev_sign = most_common_sign

    cv2.imshow("Sign Recognition", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------- Cleanup -------------------
cap.release()
cv2.destroyAllWindows()
