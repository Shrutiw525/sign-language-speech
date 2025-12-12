import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import models, layers
import pickle

DATASET_PATH = "dataset_landmarks/"

X = []
y = []

gestures = os.listdir(DATASET_PATH)

for gesture in gestures:
    gesture_folder = os.path.join(DATASET_PATH, gesture)
    for file in os.listdir(gesture_folder):
        file_path = os.path.join(gesture_folder, file)
        landmarks = np.load(file_path)
        X.append(landmarks)
        y.append(gesture)

X = np.array(X)
y = np.array(y)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

model = models.Sequential([
    layers.Dense(128, activation="relu", input_shape=(X.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dense(len(gestures), activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

model.save("sign_model.h5")

with open("label_map.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Training complete. Model saved successfully!")
