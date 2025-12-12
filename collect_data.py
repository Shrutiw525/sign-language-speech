import cv2
import os

GESTURE = input("Enter gesture name (hello/thankyou/yes/no/ok/please): ")

SAVE_PATH = f"dataset/ISL/{GESTURE}"

if not os.path.exists(SAVE_PATH):
    print("Folder does not exist!")
    exit()

cap = cv2.VideoCapture(0)
count = 0

print("Press 's' to start saving frames, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(frame, f"Frames saved: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Recording ISL Gesture", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):  # save frames
        filename = os.path.join(SAVE_PATH, f"{count}.jpg")
        cv2.imwrite(filename, frame)
        count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Done! Total frames saved:", count)
