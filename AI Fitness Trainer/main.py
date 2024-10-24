# import libraries
import cv2
import math
import mediapipe as mp
import matplotlib.pyplot as plt
from plyer import notification
from threshold import get_thresholds_beginner


# threshold count
STATE_THRESHOLD = 5
FEEDBACK_THRESHOLD = 3
OFFSET_THRESHOLD = 100
INACTIVE_THRESHOLD = 15


# initialization
# class counter:
#     def init(self):
#         self.CORRECT = 0
#         self.INCORRECT = 0

#     def increment_counter(self):
#         self.CORRECT += 1
    
#     def decrement_counter(self):
#         self.INCORRECT += 1

#     def get_counts(self):
#         return self.CORRECT, self.INCORRECT



# offset angle (angle between nose and shoulder)
# offset distance is always calculated between two points(any).
def offsetAngle(x1, y1, x2, y2):
    # theta = math.acos((y2 - y1) * (-y1) / (math.sqrt((x2 - x1)**2 + (y2 - y1)**2) * y1))
    # degree = int(180/math.pi) * theta
    return  math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# calculate angle hip_shldr
def Angle_hip_shldr(x1, y1, x2, y2):
    theta = math.acos( (y2 -y1)*(-y1) / (math.sqrt(
        (x2 - x1)**2 + (y2 - y1)**2 ) * y1) )
    degree = int(180/math.pi)*theta
    return degree

# calculate angle hip_shldr
def Angle_hip_knee(a1, b1, a2, b2):
    theta = math.acos( (b2 -b1)*(-b1) / (math.sqrt(
        (a2 - a1)**2 + (b2 - b1)**2 ) * b1) )
    degree = int(180/math.pi)*theta
    return degree

# calculate angle hip_shldr
def Angle_knee_ankle(c1, d1, c2, d2):
    theta = math.acos( (d2 -d1)*(-d1) / (math.sqrt(
        (c2 - c1)**2 + (d2 - d1)**2 ) * d1) )
    degree = int(180/math.pi)*theta
    return degree


def sendWarning(x):
    notification.notify(
        title = "Squat Trainer",
        message = f"Perfect Squats done today. {x} angle!",
        timeout = 10
    )


font = cv2.FONT_HERSHEY_SIMPLEX
green = (127, 255, 0)
red = (50, 50, 255)
yellow = (0, 255, 255)
pink = (255, 0, 255)


mp_squat = mp.solutions.pose
pose = mp_squat.Pose()

filename = "3.mp4"
cap = cv2.VideoCapture(filename)

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

height = 900
width= 1900

(cap.set(cv2.CAP_PROP_FRAME_WIDTH, width))
int(cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height))

frame_size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_output = cv2.VideoWriter("Tested.mp4", fourcc, fps, frame_size)

def draw_dotted_line(image, start, end, color, thickness=1):
    dist = int(math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2))
    for i in range(0, dist, 5):  # Adjust the step to control dot spacing
        x = int(start[0] + (end[0] - start[0]) * i / dist)
        y = int(start[1] + (end[1] - start[1]) * i / dist)
        cv2.circle(image, (x, y), thickness, color, -1)

while cap.isOpened():
    success, video = cap.read()
    if not success:
        print("Reached the end of the video.")
        break
    print(f"Frame reached sucessfully: {success}, Frame shape: {video.shape}")


    # convert to RGB
    video_rgb = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)
    keypoints = pose.process(video_rgb)
    lm = keypoints.pose_landmarks


    # detect landmarks
    if lm:
        lmPose = mp_squat.PoseLandmark

        # Extract key points
        l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * width) 
        l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * height)
        r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * width)   # to calculate offset distance
        r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * height)  # to calculate offset distance
        l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * width)
        l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * height)
        l_knee_x = int(lm.landmark[lmPose.LEFT_KNEE].x * width)
        l_knee_y = int(lm.landmark[lmPose.LEFT_KNEE].y * height)
        l_ankle_x = int(lm.landmark[lmPose.LEFT_ANKLE].x * width)
        l_ankle_y = int(lm.landmark[lmPose.LEFT_ANKLE].y * height)

        # calculate offset distance
        offset = offsetAngle(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
        hip_inclination = Angle_hip_shldr(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)
        knee_inclination = Angle_hip_knee(l_knee_x, l_knee_y, l_hip_x, l_hip_y)
        ankle_inclination = Angle_knee_ankle(l_ankle_x, l_ankle_y, l_knee_x, l_knee_y)

        # draw landmarks
        # cv2.circle(video, (l_shldr_x, l_shldr_y), 10, yellow, -1)
        # cv2.circle(video, (r_shldr_x, r_shldr_y), 10, pink, -1)
        # cv2.circle(video, (l_hip_x, l_hip_y), 10, red, -1)
        # cv2.circle(video, (l_knee_x, l_knee_y), 10, green, -1)
        # cv2.circle(video, (l_ankle_x, l_ankle_y), 10, pink, -1)

        # Draw dotted lines between the keypoints
        # draw_dotted_line(video, (l_shldr_x, l_shldr_y), (l_hip_x, l_hip_y), green)
        # draw_dotted_line(video, (l_hip_x, l_hip_y), (l_knee_x, l_knee_y), green)
        # draw_dotted_line(video, (l_knee_x, l_knee_y), (l_ankle_x, l_ankle_y), green)

        # Display angles
        cv2.putText(video, f"{int(hip_inclination)}", (int((l_shldr_x + l_hip_x) / 2), int((l_shldr_y + l_hip_y) / 2)), font, 0.6, red, 1)
        cv2.putText(video, f"{int(knee_inclination)}", (int((l_hip_x + l_knee_x) / 2), int((l_hip_y + l_knee_y) / 2)), font, 0.6, red, 1)
        cv2.putText(video, f"{int(ankle_inclination)}", (int((l_knee_x + l_ankle_x) / 2), int((l_knee_y + l_ankle_y) / 2)), font, 0.6, red, 1)
        
        # check posture status
        posture_status = "Perfect squat"
        color =  green
        

        # LOGIC
        if offset < OFFSET_THRESHOLD:
            cv2.putText(video, f"Aligned Properly: {int(offset)}", (width - 130, 30), font, 0.8, green, 5)
            

            # hip

            if 0 == hip_inclination == 7:
                posture_status = "Start Squat"
            
            elif 45 <= hip_inclination <= 52:
                posture_status = "Too Deep torso bend."
                color = red
            
            elif hip_inclination <= 44:
                posture_status = "Not Deep Enough bend(Torso)."
                color = yellow

            else:
                posture_status = "Deep Enough."
                color = green
            cv2.putText(video, f"Hip inclination: {int(hip_inclination)} - {posture_status}", (7, 60), font, 0.9, color, 5)

            
            # knee
            if knee_inclination <= 43:
                 posture_status = "Start Squat"
            
            elif 80 <= knee_inclination <= 96:
                posture_status = "Too much Knee bend."
                color = red
            
            elif 56 == knee_inclination == 66 :
                posture_status = "Not enough Knee bend."
                color = yellow
            
            elif 89 <= knee_inclination == 94:
                posture_status = "Perfect Knee band."
                color = green
            cv2.putText(video, f"knee inclination: {int(knee_inclination)} - {posture_status}", (7, 120), font, 0.9, color, 5)

           
            # ankle
            if ankle_inclination < 23:
                posture_status = "Start Squat"
            
            elif 25 < ankle_inclination < 27:
                posture_status = "Too much pressure on ankle."
                color = red
            
            elif 25 <= ankle_inclination < 40:
                posture_status = "Not enough ankle bend."
                color = yellow
            
            else:
                posture_status = "Perfect Ankle bend."
                color = green
            cv2.putText(video, f"Ankle inclination: {int(ankle_inclination)} - {posture_status}", (7, 180), font, 0.9, color, 5)


      


        # else:
        #     cv2.putText(video, f"Not Aligned Properly: {int(offset)}", (width - 150, 30), font, 0.6, green, 2)

        
        #show posture
        # cv2.putText(video, f"Aligned: , {int(offset)}", (width - 150, 30), font, 0.9, green, 2)
        # if posture_status == "Deep Enough."  and posture_status == "Perfect Knee band." and posture_status == "Perfect Ankle bend.":
        #     sendWarning (f"Waring : Perfect Squat Done")
        # elif posture_status == "Not Deep Enough.":
        #     sendWarning (f"Waring : {posture_status}")
        # elif posture_status == "Not enough Knee bend.":
        #     sendWarning (f"Waring : {posture_status}")
        # elif posture_status == "Not enough ankle bend.":
        #     sendWarning(f"Waring : {posture_status}")
        # else:
        #     posture_status = "Squat's done improper way!" 


    # resizing video 
    resized_frame = cv2.resize(video, (width, height))
    cv2.imshow("Squat Analysis",resized_frame)
    video_output.write(resized_frame)
    print("Resized frame shape:", resized_frame.shape)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()