# data_collector_yoga_poses.py
import cv2
import mediapipe as mp
import numpy as np
import csv
import datetime
import os

# --- Initialize MediaPipe Pose solution ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- Webcam Input Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream. Please ensure your webcam is connected and not in use.")
    exit()

# --- CSV File Setup for Data Collection ---
csv_file_path = 'yoga_pose_data.csv'
column_names = ['label']
for i in range(33):
    column_names.extend([f'x_lm_{i}', f'y_lm_{i}', f'z_lm_{i}', f'v_lm_{i}'])

file_exists = os.path.isfile(csv_file_path)
csv_file = open(csv_file_path, 'a' if file_exists else 'w', newline='')
writer = csv.writer(csv_file)

if not file_exists:
    writer.writerow(column_names)
    print(f"CSV file '{csv_file_path}' created with headers.")
else:
    print(f"Appending to existing CSV file '{csv_file_path}'.")

# --- Pose Estimation ---
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    print("\n--- Yoga Pose Data Collection Mode ---")
    print("Instructions:")
    print("  - Position yourself in front of the webcam.")
    print("  - Press 'D' to save a 'Downward Dog' sample.")
    print("  - Press 'T' to save a 'Tree Pose' sample.")
    print("  - Press 'W' to save a 'Warrior II' sample.")
    print("  - Press 'Q' to quit the data collection.")
    print("------------------------------------\n")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        cv2.putText(image, "Press 'D' for Downward Dog", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, "Press 'T' for Tree Pose", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, "Press 'W' for Warrior II", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        cv2.putText(image, "Press 'Q' to quit", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        current_landmark_data = []
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            for landmark in landmarks:
                current_landmark_data.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord('d'):
                writer.writerow(['Downward Dog'] + current_landmark_data)
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Saved 'Downward Dog' sample.")
            elif key == ord('t'):
                writer.writerow(['Tree Pose'] + current_landmark_data)
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Saved 'Tree Pose' sample.")
            elif key == ord('w'):
                writer.writerow(['Warrior II'] + current_landmark_data)
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Saved 'Warrior II' sample.")
            elif key == ord('q'):
                print("Quitting data collection.")
                break

        else:
            cv2.putText(image, "No Pose Detected", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting data collection.")
                break

        cv2.imshow('Just Yoga - Data Collection Mode', image)

cap.release()
cv2.destroyAllWindows()
csv_file.close()
print(f"Data collection finished. Data saved to '{csv_file_path}'.")

