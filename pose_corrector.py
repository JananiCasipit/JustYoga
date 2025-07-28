# pose_corrector.py
import numpy as np

def calculate_angle(a, b, c):
    # a, b, c are landmark coordinates (x, y, z)
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

def provide_feedback(current_landmarks, predicted_pose):
    feedback = []
    # Define ideal angles for each pose (example for 'tree_pose')
    if predicted_pose == "tree_pose":
        # Assuming current_landmarks is a flat list of (x,y,z,visibility) for 33 landmarks
        # Convert to a more usable format, e.g., a dictionary mapping landmark name to (x,y,z)
        # For simplicity, let's assume we know the indices of relevant landmarks
        # Example: hip_left_idx, knee_left_idx, ankle_left_idx based on MediaPipe

        # This is simplified. You'd need a robust mapping and knowledge of MediaPipe's landmark order.
        # Example indices (refer to MediaPipe documentation for exact mapping):
        # NOSE = 0, LEFT_EYDER_INNER = 1, LEFT_EYE = 2, LEFT_EYE_OUTER = 3, RIGHT_EYE_INNER = 4, RIGHT_EYE = 5, RIGHT_EYE_OUTER = 6, LEFT_EAR = 7, RIGHT_EAR = 8, MOUTH_LEFT = 9, MOUTH_RIGHT = 10, LEFT_SHOULDER = 11, RIGHT_SHOULDER = 12, LEFT_ELBOW = 13, RIGHT_ELBOW = 14, LEFT_WRIST = 15, RIGHT_WRIST = 16, LEFT_PINKY = 17, RIGHT_PINKY = 18, LEFT_INDEX = 19, RIGHT_INDEX = 20, LEFT_THUMB = 21, RIGHT_THUMB = 22, LEFT_HIP = 23, RIGHT_HIP = 24, LEFT_KNEE = 25, RIGHT_KNEE = 26, LEFT_ANKLE = 27, RIGHT_ANKLE = 28, LEFT_HEEL = 29, RIGHT_HEEL = 30, LEFT_FOOT_INDEX = 31, RIGHT_FOOT_INDEX = 32

        # Example: Tree Pose - check left knee angle if right leg is supporting
        # If using right leg, then right hip (24), right knee (26), right ankle (28) are key for straight leg
        # Left hip (23), left knee (25), left ankle (27) for bent leg

        # Simplified example: check if knee is too bent for a standing pose
        # Assuming a straight leg for mountain pose
        if len(current_landmarks) == 33 * 4: # ensure all landmarks are present
            right_hip = current_landmarks[24*4 : 24*4+3] # x,y,z
            right_knee = current_landmarks[26*4 : 26*4+3]
            right_ankle = current_landmarks[28*4 : 28*4+3]

            if all(val is not None for val in right_hip + right_knee + right_ankle):
                right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
                if right_leg_angle < 165: # Example threshold for a straight leg
                    feedback.append("Straighten your standing leg more.")

            # For tree pose, typically the lifted foot is on the inner thigh of the standing leg
            # This would involve checking the relative position of the left foot to the right thigh.
            # More complex geometry is needed here.
            # Example: Check if the left knee is pointing outwards (for tree pose)
            left_hip = current_landmarks[23*4 : 23*4+3]
            left_knee = current_landmarks[25*4 : 25*4+3]

            # A very basic check: Is the left knee relatively high?
            if all(val is not None for val in left_hip + left_knee) and left_knee[1] > left_hip[1]: # y-coordinate higher means lower on image
                 feedback.append("Lift your lifted knee higher towards your hip.")


    # Add more pose-specific logic here
    elif predicted_pose == "mountain_pose":
        # Example: Check for straight back
        # You'd need to define ideal relationships between shoulder, hip, and ankle landmarks
        feedback.append("Ensure your spine is aligned and shoulders are relaxed.")

    if not feedback:
        feedback.append(f"Great work on your {predicted_pose}!")

    return feedback
