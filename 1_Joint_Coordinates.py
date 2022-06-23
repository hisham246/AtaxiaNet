import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from datetime import datetime

cwd = os.getcwd()  # Get the current working directory

for i in range(21, 24):
    path_videos = cwd + '\\Test LSTM\\Test Videos\\' + f'test{i}.mp4'

    path_data = cwd + '\\Joint Coordinates\\Pre Augmentation\\Test\\' + f'test_coordinates_{i}.csv'

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    TimeStamp = []  # Time stamps list

    X_left_shoulder = []
    X_right_shoulder = []
    X_left_elbow = []
    X_right_elbow = []
    X_left_wrist = []
    X_right_wrist = []
    X_left_hip = []
    X_right_hip = []
    X_left_knee = []
    X_right_knee = []
    X_left_ankle = []
    X_right_ankle = []

    Y_left_shoulder = []
    Y_right_shoulder = []
    Y_left_elbow = []
    Y_right_elbow = []
    Y_left_wrist = []
    Y_right_wrist = []
    Y_left_hip = []
    Y_right_hip = []
    Y_left_knee = []
    Y_right_knee = []
    Y_left_ankle = []
    Y_right_ankle = []

    Z_left_shoulder = []
    Z_right_shoulder = []
    Z_left_elbow = []
    Z_right_elbow = []
    Z_left_wrist = []
    Z_right_wrist = []
    Z_left_hip = []
    Z_right_hip = []
    Z_left_knee = []
    Z_right_knee = []
    Z_left_ankle = []
    Z_right_ankle = []

    cap = cv2.VideoCapture(path_videos)

    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print(f'Finished on test{i}')
                # If loading a video, use 'break' instead of 'continue'.
                break

            # Convert the BGR image to RGB.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

            # Get the current time
            time = datetime.now()
            TimeStamp.append(time)

            X_LShoulder = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x
            Y_LShoulder = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            Z_LShoulder = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z

            X_RShoulder = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x
            Y_RShoulder = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            Z_RShoulder = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].z

            X_LElbow = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x
            Y_LElbow = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y
            Z_LElbow = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].z

            X_RElbow = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x
            Y_RElbow = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y
            Z_RElbow = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].z

            X_LWrist = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x
            Y_LWrist = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
            Z_LWrist = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].z

            X_RWrist = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x
            Y_RWrist = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
            Z_RWrist = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].z

            X_LHip = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x
            Y_LHip = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y
            Z_LHip = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].z

            X_RHip = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x
            Y_RHip = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y
            Z_RHip = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].z

            X_LKnee = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x
            Y_LKnee = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y
            Z_LKnee = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].z

            X_RKnee = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x
            Y_RKnee = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y
            Z_RKnee = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].z

            X_LAnkle = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x
            Y_LAnkle = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y
            Z_LAnkle = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].z

            X_RAnkle = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x
            Y_RAnkle = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y
            Z_RAnkle = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].z

            X_left_shoulder.append(X_LShoulder)
            X_right_shoulder.append(X_RShoulder)
            X_left_elbow.append(X_LElbow)
            X_right_elbow.append(X_RElbow)
            X_left_wrist.append(X_LWrist)
            X_right_wrist.append(X_RWrist)
            X_left_hip.append(X_LHip)
            X_right_hip.append(X_RHip)
            X_left_knee.append(X_LKnee)
            X_right_knee.append(X_RKnee)
            X_left_ankle.append(X_LAnkle)
            X_right_ankle.append(X_RAnkle)

            Y_left_shoulder.append(X_LShoulder)
            Y_right_shoulder.append(X_RShoulder)
            Y_left_elbow.append(X_LElbow)
            Y_right_elbow.append(X_RElbow)
            Y_left_wrist.append(X_LWrist)
            Y_right_wrist.append(X_RWrist)
            Y_left_hip.append(Y_LHip)
            Y_right_hip.append(Y_RHip)
            Y_left_knee.append(Y_LKnee)
            Y_right_knee.append(Y_RKnee)
            Y_left_ankle.append(Y_LAnkle)
            Y_right_ankle.append(Y_RAnkle)

            Z_left_shoulder.append(X_LShoulder)
            Z_right_shoulder.append(X_RShoulder)
            Z_left_elbow.append(X_LElbow)
            Z_right_elbow.append(X_RElbow)
            Z_left_wrist.append(X_LWrist)
            Z_right_wrist.append(X_RWrist)
            Z_left_hip.append(Z_LHip)
            Z_right_hip.append(Z_RHip)
            Z_left_knee.append(Z_LKnee)
            Z_right_knee.append(Z_RKnee)
            Z_left_ankle.append(Z_LAnkle)
            Z_right_ankle.append(Z_RAnkle)

    cap.release()

    # Total no. of seconds elapsed in execution
    delta = TimeStamp[-1] - TimeStamp[0]
    sec = delta.total_seconds()
    num = sec / (len(TimeStamp))
    timestamps = []
    for i in range(len(TimeStamp)):
        s = (num * i) + num
        timestamps.append(s)


    cols = ['XLeftShoulder', 'YLeftShoulder', 'ZLeftShoulder', 'XRightShoulder', 'YRightShoulder',
            'ZRightShoulder', 'XLeftElbow', 'YLeftElbow', 'ZLeftElbow', 'XRightElbow', 'YRightElbow', 'ZRightElbow',
            'XLeftWrist', 'YLeftWrist', 'ZLeftWrist', 'XRightWrist', 'YRightWrist', 'ZRightWrist', 'XLeftHip',
            'YLeftHip', 'ZLeftHip', 'XRightHip', 'YRightHip', 'ZRightHip', 'XLeftKnee', 'YLeftKnee', 'ZLeftKnee',
            'XRightKnee', 'YRightKnee', 'ZRightKnee', 'XLeftAnkle', 'YLeftAnkle', 'ZLeftAnkle', 'XRightAnkle',
            'YRightAnkle', 'ZRightAnkle']

    data = np.column_stack(
        [X_left_shoulder, Y_left_shoulder, Z_left_shoulder, X_right_shoulder, Y_right_shoulder, Z_right_shoulder,
         X_left_elbow, Y_left_elbow, Z_left_elbow, X_right_elbow, Y_right_elbow, Z_right_elbow,
         X_left_wrist, Y_left_wrist, Z_left_wrist, X_right_wrist, Y_right_wrist, Z_right_wrist,
         X_left_hip, Y_left_hip, Z_left_hip, X_right_hip, Y_right_hip, Z_right_hip,
         X_left_knee, Y_left_knee, Z_left_knee, X_right_knee, Y_right_knee, Z_right_knee,
         X_left_ankle, Y_left_ankle, Z_left_ankle, X_right_ankle, Y_right_ankle, Z_right_ankle])

    kinematics = pd.DataFrame(data, columns=cols)
    kinematics.insert(0, 'Timestamp', timestamps, True)

    kinematics.to_csv(path_data, index=False)

    # joints = ['LeftShoulder', 'RightShoulder', 'LeftElbow', 'RightElbow', 'LeftWrist', 'RightWrist',
    #           'LeftHip', 'RightHip', 'LeftKnee', 'RightKnee', 'LeftAnkle', 'RightAnkle']
    #
    # cols = pd.MultiIndex.from_product([joints, ['x', 'y', 'z']])


