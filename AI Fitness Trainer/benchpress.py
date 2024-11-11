import cv2
import numpy as np
import mediapipe as mp

# Initialize video capture objects for both cameras
cap1 = cv2.VideoCapture(0)  # Frontal view camera
cap2 = cv2.VideoCapture(1)  # Side view camera

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Provide feedback based on angles
def provide_feedback(knee_angle_side, knee_angle_front, ideal_knee_angle=90):
    if abs(knee_angle_side - ideal_knee_angle) < 5 and abs(knee_angle_front - ideal_knee_angle) < 5:
        return "Perfect squat!"
    elif knee_angle_side < ideal_knee_angle:
        return "Bend your knees more!"
    elif knee_angle_front < ideal_knee_angle:
        return "Align your knees better!"
    return "Adjust your form!"

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2:
        break
    
    # Convert frames to RGB
    frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    
    # Perform pose estimation
    results1 = pose.process(frame1_rgb)
    results2 = pose.process(frame2_rgb)
    
    if results1.pose_landmarks and results2.pose_landmarks:
        landmarks1 = results1.pose_landmarks.landmark
        landmarks2 = results2.pose_landmarks.landmark
        
        # Define key points
        hip = mp_pose.PoseLandmark.LEFT_HIP.value
        knee = mp_pose.PoseLandmark.LEFT_KNEE.value
        ankle = mp_pose.PoseLandmark.LEFT_ANKLE.value
        
        # Calculate angles
        knee_angle_side = calculate_angle(
            [landmarks1[hip].x, landmarks1[hip].y],
            [landmarks1[knee].x, landmarks1[knee].y],
            [landmarks1[ankle].x, landmarks1[ankle].y]
        )
        knee_angle_front = calculate_angle(
            [landmarks2[hip].x, landmarks2[hip].y],
            [landmarks2[knee].x, landmarks2[knee].y],
            [landmarks2[ankle].x, landmarks2[ankle].y]
        )
        
        # Provide feedback
        feedback = provide_feedback(knee_angle_side, knee_angle_front)
        print(feedback)
        
    # Display frames from both cameras
    cv2.imshow('Frontal View', frame1)
    cv2.imshow('Side View', frame2)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture objects
cap1.release()
cap2.release()
cv2.destroyAllWindows()
