# import libraries
import cv2
import math
import mediapipe as mp
import matplotlib.pyplot as plt
from plyer import notification
import time
import numpy as np
# from twilio.rest import Client
# from threshold import get_thresholds_beginner


# threshold count
# STATE_THRESHOLD = 5
# FEEDBACK_THRESHOLD = 3
OFFSET_THRESHOLD = 100
# INACTIVE_THRESHOLD = 15

hip_threshold = 10  # Angle below which the hip inclination indicates start of squat
knee_threshold = 90  # Knee inclination threshold for is_squatting
ankle_threshold = 20  # Ankle inclination threshold

filename = "3.mp4"
cap = cv2.VideoCapture(filename)

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 

mp_squat = mp.solutions.pose
pose = mp_squat.Pose()

display_scale = 0.5

# initialization

CORRECT = 0
INCORRECT = 0
is_squatting = False


# class Person:
#     def __init__(self, name, gender, age, weight, height):
#         self.name = name
#         self.age = age
#         self.gender = gender
#         self.weight = weight
#         self.height = height

#     def name_descr(self):
#         return f"Name is {self.name}."

#     def weight_descr(self):
#         return f"Weight is {self.weight} Kg."
        
#     def height_descr(self):
#         return f"Height is {self.height} Cm."
        
#     def age_descr(self):
#         return f"Age is {self.age} Yrs."
    
#     def validate_gender_descr(self):
#         if self.gender not in ["Male", "Female"]:
#             print("Enter Male or Female.")
#             raise ValueError("Invalid gender entered.")

#     def gender_descr(self):
#         return f"Gender is {self.gender}."
       
# print()

# # INPUT
# name = str(input("Enter Name (Full Name):"))
# age = int(input(f"Enter Age (Yrs): "))
# height =float(input(f"Enter Height (Cm): "))
# weight =float(input(f"Enter Weight (Kg): "))
# gender =str(input(f"Enter Gender (Male or Female): "))

# try:
#     person = Person(name, gender, age, weight, height)
#     print("All data inserted successfully!!")
#     print(person.gender_descr())
# except ValueError as e:
#     print(e)

# print()


# # DATAFRAME
# data = {
#     "Full Name": [name],
#     "Gender": [gender],
#     "Age": [age],
#     "Weight": [weight],
#     "Height": [height]
# }
# df = pd.DataFrame(data)


# # EXCLE FILE
# try:
#     existing_df = pd.read_excel('User_data.xlsx')
# except FileNotFoundError:
#     existing_df = pd.DataFrame(columns=["Full Name", "Gender", "Age", "Weight", "Height"])

# combined_df = pd.concat([existing_df, df], ignore_index=True)

# combined_df.to_excel('User_data.xlsx', index=False)

# try:
#     person = Person(name, gender, age, weight, height)
#     print("All data inserted successfully!!")
#     print(person.gender_descr())
# except ValueError as e:
#     print(e)
# print()

# person = Person(name, gender, age, weight, height)

# print(person.name_descr())
# print(person.gender_descr())
# print(person.age_descr())
# print(person.weight_descr())
# print(person.height_descr())

# if person.name_descr() and person.gender_descr() and person.age_descr() and person.weight_descr() and person.height_descr() != None:

# offset angle (angle between nose and shoulder)\
# offset distance is always calculated between two points(any).
def offsetAngle(x1, y1, x2, y2):
    # theta = math.acos((y2 - y1) * (-y1) / (math.sqrt((x2 - x1)**2 + (y2 - y1)**2) * y1))
    # degree = int(180/math.pi) * theta
    return  math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# calculate angle hip_shldr
def Angle_hip_shldr(x1, y1, x2, y2):
        theta = math.acos( (y2 -y1)*(-y1) / (math.sqrt((x2 - x1)**2 + (y2 - y1)**2 ) * y1) )
        degree = int(180/math.pi)*theta
        return degree

def Angle_elbow_shldr(e1, f1, e2, f2):
        theta = math.acos( (f2 - f1)*(-f1) / (math.sqrt((e2 - e1)**2 + (f2 - f1)**2 ) * f1) )
        degree = int(180/math.pi)*theta
        return degree

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b # vectors ba and bc
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

# calculate angle hip_knee
def Angle_hip_knee(a1, b1, a2, b2):
        theta = math.acos( (b2 -b1)*(-b1) / (math.sqrt((a2 - a1)**2 + (b2 - b1)**2 ) * b1))
        degree = int(180/math.pi)*theta
        return degree

# calculate angle knee_ankle
def Angle_knee_ankle(c1, d1, c2, d2):
        theta = math.acos( (d2 -d1)*(-d1) / (math.sqrt((c2 - c1)**2 + (d2 - d1)**2 ) * d1) )
        degree = int(180/math.pi)*theta
        return degree


def sendWarning(x):
        notification.notify(
            title = "Squat Trainer",
        #     message = f"Perfect Squats done today. {x} angle!",
            timeout = 10
        )


font = cv2.FONT_HERSHEY_SIMPLEX
green = (127, 255, 0)
red = (50, 50, 255)
yellow = (0, 255, 255)
pink = (255, 0, 255)


duration_video = frame_count / fps

# min = int(duration_video / 60)
# sec = int(duration_video % 60)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

start_time = time.time()

# print(f"Video length: {min} and {sec} seconds.")

# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


height = 750
width= 1500

(cap.set(cv2.CAP_PROP_FRAME_WIDTH, width))
int(cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height))

frame_size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_output = cv2.VideoWriter("Tested.mp4", fourcc, fps, frame_size)

# time = min, sec


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


        #TIME Calculation
        # Calculate elapsed time based on frames
        eclapsed_frames = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        eclapsed_seconds = eclapsed_frames / fps
        eclapsed_microseconds = (eclapsed_frames / fps) * 1e6
        time_text = f'Time: {int(eclapsed_seconds // 60):02}:{int(eclapsed_seconds % 60):02}:{int(eclapsed_microseconds % 1e6):06}'
        remaining_seconds = (frame_count - eclapsed_frames) / fps

        # Calculate elapsed time based on frames
        # elapsed_frames = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # elapsed_seconds = elapsed_frames / fps
        # remaining_seconds = (frame_count - elapsed_frames) / fps

        # Send message after 2 or fewer seconds remaining
        if remaining_seconds <= 2:
                sendWarning("You have done BodyWeight squat!")

        # Create time text
        # elapsed_microseconds = (elapsed_frames / fps) * 1e6
        # time_text = f'Time: {int(elapsed_seconds // 60):02}:{int(elapsed_seconds % 60):02}:{int(elapsed_microseconds % 1e6):06}'

        # Add the time counter to the frame
        cv2.putText(video, time_text, (width - 350, 30), font, 0.8, (0, 255, 0), 2)

        # Landmarks on the original frame
        if keypoints.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                        video, keypoints.pose_landmarks, mp_squat.POSE_CONNECTIONS
                )

        # Write the frame with the time counter
        video_output.write(video)


        # Add the time counter to the frame6
        # (width - 200, 100)
        cv2.putText(video, time_text, (7, 350), font, 1.2, red, 5)
 

        # Write the frame_size with the time counter
        out.write(frame_size)

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
            l_elbow_x = int(lm.landmark[lmPose.LEFT_ELBOW].x * width)
            l_elbow_y = int(lm.landmark[lmPose.LEFT_ELBOW].y * height)
            l_wrist_x = int(lm.landmark[lmPose.LEFT_WRIST].x * width)
            l_wrist_y = int(lm.landmark[lmPose.LEFT_WRIST].y * height)
            l_shldr = (l_shldr_x, l_shldr_y)
            l_elbow = (l_elbow_x, l_elbow_y)
            l_wrist = (l_wrist_x, l_wrist_y)

            # calculate offset distance
            offset = offsetAngle(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
            hip_angle = Angle_hip_shldr(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)
            knee_angle = Angle_hip_knee(l_knee_x, l_knee_y, l_hip_x, l_hip_y)
            ankle_angle = Angle_knee_ankle(l_ankle_x, l_ankle_y, l_knee_x, l_knee_y)
        #     shldr_elbow_angle = Angle_elbow_shldr(l_elbow_x, l_elbow_y, l_shldr_x, l_shldr_y)
            wrist_elbow_angle = Angle_elbow_shldr(l_wrist_x, l_wrist_y, l_elbow_x, l_elbow_y)
            elbow_angle = calculate_angle(l_shldr, l_elbow, l_wrist)

        #     shldr_wrist_angle = Angle_shldr_wrist(l_shldr_x, l_shldr_y, l_wrist_x, l_wrist_y)
            

            # draw points
            cv2.circle(video, (l_shldr_x, l_shldr_y), 10, yellow, -1)
            cv2.circle(video, (r_shldr_x, r_shldr_y), 10, pink, -1)
            cv2.circle(video, (l_hip_x, l_hip_y), 10, red, -1)
            cv2.circle(video, (l_knee_x, l_knee_y), 10, green, -1)
            cv2.circle(video, (l_ankle_x, l_ankle_y), 10, pink, -1)
            cv2.circle(video, (l_elbow_x, l_elbow_y), 10, pink, -1)
            cv2.circle(video, (l_wrist_x, l_wrist_y), 10, pink, -1)
        #     cv2.circle(video, (l_shldr_x, l_shldr_y), (l_wrist_x, l_wrist_y), 10, pink, -1)

            # Draw dotted lines between the keypoints
            draw_dotted_line(video, (l_shldr_x, l_shldr_y), (l_hip_x, l_hip_y), green)
            draw_dotted_line(video, (l_hip_x, l_hip_y), (l_knee_x, l_knee_y), green)
            draw_dotted_line(video, (l_knee_x, l_knee_y), (l_ankle_x, l_ankle_y), green)
            draw_dotted_line(video, (l_shldr_x, l_shldr_y), (l_elbow_x, l_elbow_y), green)
            draw_dotted_line(video, (l_elbow_x, l_elbow_y), (l_wrist_x, l_wrist_y), green)

            # Display angles
            cv2.putText(video, f"{int(hip_angle)}", (int((l_shldr_x + l_hip_x) / 2), int((l_shldr_y + l_hip_y) / 2)), font, 1.0, red, 5)
            cv2.putText(video, f"{int(knee_angle)}", (int((l_hip_x + l_knee_x) / 2), int((l_hip_y + l_knee_y) / 2)), font, 1.0, red, 5)
            cv2.putText(video, f"{int(ankle_angle)}", (int((l_knee_x + l_ankle_x) / 2), int((l_knee_y + l_ankle_y) / 2)), font, 1.0, red, 5)
        #     cv2.putText(video, f"{int(shldr_elbow_angle)}", (int((l_shldr_x + l_shldr_x) / 2), int((l_elbow_y + l_elbow_y) / 2)), font, 1.0, red, 5)
        #     cv2.putText(video, f"{int(wrist_elbow_angle)}", (int((l_elbow_x + l_elbow_x) / 2), int((l_wrist_y + l_wrist_y) / 2)), font, 1.0, red, 5)

            cv2.putText(video, f"{int(elbow_angle)}", (l_elbow_x + 20, l_elbow_y - 20), font, 0.8, red, 2)

            
            # check posture status
            posture_status = "Perfect squat"
            color =  green
            

            # if offset < OFFSET_THRESHOLD:
            #     cv2.putText(video, f"Aligned Properly: {int(offset)}", (width - 130, 30), font, 0.8, green, 5)
                
        # LOGIC
        # Hip
        if hip_angle <= 10:
                posture_status_hip = "Start Squat"

        elif hip_angle <= 44:
                posture_status_hip = "Lean Forward(Torso)."
                color = yellow
                
        elif 45 <= hip_angle <= 60:
                posture_status_hip = "Perfect Torso bend."
                color = green

        elif hip_angle > 60:
                posture_status_hip = "Lean backword."
                color = red

        cv2.putText(video, f"Hip inclination: {int(hip_angle)} - {posture_status_hip}", (7, 60), font, 1.2, color, 5)

                
        # Knee
        if knee_angle <= 10:
                posture_status_knee = "Start Squat"
                
        elif 11 <= knee_angle < 89 :
                posture_status_knee = "Bend You knees."
                color = yellow

        elif knee_angle > 120:
                posture_status_knee = "Too much Knee bend."
                color = red
                
        elif 90 <= knee_angle <= 120:
                posture_status_knee = "Perfect Knee band."
                color = green
        cv2.putText(video, f"knee inclination: {int(knee_angle)} - {posture_status_knee}", (7, 120), font, 1.2, color, 5)


        # Ankle
        if ankle_angle < 8:
                posture_status_ankle = "Start Squat"
                
        elif ankle_angle > 30:
                posture_status_ankle = "Too much pressure on ankle."
                color = red
                
        elif ankle_angle < 20:
                posture_status_ankle = "Not enough ankle bend."
                color = yellow
                
                
        elif 20 <= ankle_angle <= 30:
                posture_status_ankle = "Perfect Ankle bend."
                color = green
        cv2.putText(video, f"Ankle inclination: {int(ankle_angle)} - {posture_status_ankle}", (7, 180), font, 1.2, color, 5)


        # Elbow
        if 70 <= wrist_elbow_angle <= 92:
                posture_status_elbow = "Perfect elbow angle"
                color = green
        elif wrist_elbow_angle < 70:
                posture_status_elbow = "Straighten your elbow"
                color = red
        # cv2.putText(video, f"Elbow bend: {int(wrist_elbow_angle)} - {posture_status_elbow}", (7, 240), font, 1.2, color, 5)        


        # Message
        # if posture_status_hip == "Perfect Torso bend." and posture_status_knee == "Perfect Knee band." and posture_status_ankle == "Perfect Ankle bend.":
        #         sendWarning (f"Waring : Perfect Squat Done")

               
        # Detect squat position transition
        if posture_status_hip == "Perfect Torso bend." and posture_status_knee == "Perfect Knee band." and posture_status_ankle == "Perfect Ankle bend.":
                if not is_squatting:  # Up to down
                        is_squatting = True
        else:
                if is_squatting:  # Down to up
                        if posture_status_hip == "Perfect Torso bend." and posture_status_knee == "Perfect Knee band." and posture_status_ankle == "Perfect Ankle bend.":
                                INCORRECT += 1
                                print(f"Correct Squats: {INCORRECT}")
                        else:
                                CORRECT += 1
                                print(f"Incorrect Squats: {CORRECT}")
                is_squatting = False

        # Display the counters
        cv2.putText(video, f"Correct Squats: {CORRECT}", (7, 520), font, 1.2, (0, 255, 0), 5)
        cv2.putText(video, f"Incorrect Squats: {INCORRECT}", (7, 580), font, 1.2, (0, 0, 255), 5)


        # if posture_status_hip == "Perfect Torso bend." and posture_status_knee == "Perfect Knee band." and posture_status_ankle == "Perfect Ankle bend." and posture_status_elbow == "Perfect elbow angle":
        #         sendWarning("You have done BodyWeight Squat")
                                                                                        # OR
        if posture_status_elbow == "Perfect elbow angle":
                print("You have done BodyWeight squat")



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

# else:
#     print("Please Enter all details..")