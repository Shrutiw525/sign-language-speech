import cv2
import os

# List of sign folders to flip
sign_folders = [r"D:\signai\dataset\ISL\hello",
    r"D:\signai\dataset\ISL\please",
    r"D:\signai\dataset\ISL\thankyou",
    r"D:\signai\dataset\ISL\ok",
    r"D:\signai\dataset\ISL\yes",
    r"D:\signai\dataset\ISL\no"]  # add more as needed

for folder in sign_folders:
    for file in os.listdir(folder):
        if file.endswith(".jpg") or file.endswith(".png"):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)
            flipped = cv2.flip(img, 1)  # horizontal flip
            flipped_filename = f"flipped_{file}"
            cv2.imwrite(os.path.join(folder, flipped_filename), flipped)
            print(f"Saved {flipped_filename} in {folder}")
