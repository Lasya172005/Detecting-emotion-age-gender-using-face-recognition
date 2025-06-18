import cv2
import os
import numpy as np
from deepface import DeepFace
from datetime import datetime

# Setup CSV logging
log_file = "emotion_logs.csv"
if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        f.write("timestamp,emotion,gender,age\n")

# Start webcam
cap = cv2.VideoCapture(0)
frame_count = 0  # To reduce lag, analyze every 5th frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for DeepFace
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = DeepFace.extract_faces(frame_rgb, enforce_detection=False)

    for face in faces:
        facial_area = face['facial_area']
        x = facial_area['x']
        y = facial_area['y']
        w = facial_area['w']
        h = facial_area['h']

        face_img = frame_rgb[y:y+h, x:x+w]

        # Analyze every 5th frame to reduce lag
        if frame_count % 5 == 0:
            result = DeepFace.analyze(
                face_img,
                actions=['emotion', 'age', 'gender'],
                enforce_detection=False
            )[0]

            emotion = result['dominant_emotion']
            age = result['age']
            gender = result['gender']

            label = f"{emotion}, {gender}, {age}"
            print(f"Logged: {label}")

            # Log to CSV
            with open(log_file, "a") as f:
                f.write(f"{datetime.now()},{emotion},{gender},{age}\n")

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    frame_count += 1
    cv2.imshow("Real-Time Emotion, Age & Gender Detection", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
