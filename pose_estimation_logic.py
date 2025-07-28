# pose_estimation_logic.py
import cv2
import mediapipe as mp
import numpy as np


# Initialize MediaPipe Pose and Drawing utilities globally for this module
# This ensures the 'pose' object is created once and reused across calls to process_frame_for_pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# Configure the Pose model:
# min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for the pose detection to be considered successful.
# min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the pose landmarks to be considered tracked successfully.
# model_complexity: Complexity of the pose landmark model: 0, 1, or 2. Higher complexity generally means more accurate but slower.
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)


def process_frame_for_pose(frame):
    """
    Processes a single frame to detect human pose landmarks using MediaPipe.
    Args:
        frame (numpy.ndarray): The input image frame (BGR format from OpenCV).
    Returns:
        tuple: A tuple containing:
            - frame_with_landmarks (numpy.ndarray): The frame with drawn landmarks.
            - landmarks (list): A flattened list of [x, y, z, visibility] for each of the 33 MediaPipe landmarks,
                                or an empty list if no pose is detected.
    """
    # Convert the frame from BGR to RGB, as MediaPipe requires RGB input
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   
    # To improve performance, optionally mark the image as not writeable to pass by reference.
    # This prevents MediaPipe from making a copy of the image.
    image_rgb.flags.writeable = False


    # Process the image and get pose landmarks
    results = pose.process(image_rgb)


    # Mark the image as writeable again before drawing, as OpenCV needs to modify it.
    image_rgb.flags.writeable = True
   
    # Convert the image back to BGR for OpenCV to display.
    frame_with_landmarks = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


    landmarks_data = []
    if results.pose_landmarks:
        # Draw pose landmarks on the frame
        mp_drawing.draw_landmarks(
            frame_with_landmarks,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            # Customize drawing specifications for points and connections
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), # Color for landmarks (orange)
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)  # Color for connections (pink)
        )
       
        # Extract landmark coordinates for classification
        for landmark in results.pose_landmarks.landmark:
            # Each landmark has x, y, z (relative to hips), and visibility (confidence)
            landmarks_data.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
   
    return frame_with_landmarks, landmarks_data


if __name__ == "__main__":
    # This block is for testing the pose estimation logic independently.
    # It will open your webcam and display the pose detection.
    print("Running pose_estimation_logic.py for testing. Press 'q' to quit.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam for testing.")
        exit()


    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame during test.")
            break
       
        # Flip for selfie view
        frame = cv2.flip(frame, 1)


        # Process the frame
        processed_frame, _ = process_frame_for_pose(frame)
       
        # Display the result
        cv2.imshow('Pose Estimation Test', processed_frame)
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()
    print("Pose estimation test finished.")
