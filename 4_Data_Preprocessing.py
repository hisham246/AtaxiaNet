# Import libraries
import os
import random
import numpy as np
import pandas as pd
from operator import add, sub

ops = (add, sub)

# Set paths to file directories
cwd = os.getcwd()
limbs_wd = cwd + '\\Limb Lengths\\Test\\'
augmented_data_wd = cwd + '\\Data Augmentation\\Augmented Datasets\\Test\\'
coordinate_data_wd = cwd + '\\Joint Coordinates\\Pre Augmentation\\Test\\'
final_data_wd = cwd + '\\Joint Coordinates\\Test\\'
augmented_data_list = []
coordinate_data_list = []
limbs_list_df = []

# Columns in the joint coordinate dataset
cols = ['Timestamp', 'XLeftShoulder', 'YLeftShoulder', 'ZLeftShoulder', 'XRightShoulder', 'YRightShoulder',
        'ZRightShoulder', 'XLeftElbow', 'YLeftElbow', 'ZLeftElbow', 'XRightElbow', 'YRightElbow', 'ZRightElbow',
        'XLeftWrist', 'YLeftWrist', 'ZLeftWrist', 'XRightWrist', 'YRightWrist', 'ZRightWrist', 'XLeftHip',
        'YLeftHip', 'ZLeftHip', 'XRightHip', 'YRightHip', 'ZRightHip', 'XLeftKnee', 'YLeftKnee', 'ZLeftKnee',
        'XRightKnee', 'YRightKnee', 'ZRightKnee', 'XLeftAnkle', 'YLeftAnkle', 'ZLeftAnkle', 'XRightAnkle',
        'YRightAnkle', 'ZRightAnkle']

# Assign datasets in lists of dataframes
for i in range(1, 24):
    augmented_data_list.append(pd.read_csv(augmented_data_wd + f'test_augmented_{i}.csv'))
    coordinate_data_list.append(pd.read_csv(coordinate_data_wd + f'test_coordinates_{i}.csv'))
    limbs_list_df.append(pd.read_csv(limbs_wd + f'test_limb_lengths_{i}.csv'))

# Assign segment length datasets as NumPy arrays
limbs_list = []
for k in range(len(limbs_list_df)):
    limbs_list_df[k] = limbs_list_df[k].drop(columns=limbs_list_df[k].columns[0], axis=1)
    limbs_list.append(limbs_list_df[k].to_numpy())

JointNames = ['LeftShoulder', 'RightShoulder', 'LeftElbow', 'RightElbow', 'LeftWrist', 'RightWrist',
              'LeftHip', 'RightHip', 'LeftKnee', 'RightKnee', 'LeftAnkle', 'RightAnkle']

# Create a grouped dictionary for joint coordinates
coordinates = []
for data in coordinate_data_list:
    Joints = {}
    for name in JointNames:
        Joints[name] = np.column_stack([list(np.float_(data[f'X{name}'].tolist())),
                                        list(np.float_(data[f'Y{name}'].tolist())),
                                        list(np.float_(data[f'Z{name}'].tolist()))])
    coordinates.append(Joints)

# Sort video sequences
videos = []
for i in range(len(augmented_data_list)):
    sequence_list = []
    for j in range(32):
        k = j
        motion = []
        for k in range(int(augmented_data_list[i].shape[0] / 32)):
            motion.append(augmented_data_list[i].iloc[k].tolist())
            k += 32
        sequence = np.array(motion)
        sequence_list.append(sequence)
    videos.append(np.concatenate(sequence_list, axis=0))

# Separate original sequence from augmented ones
original_videos = []
for video in videos:
    origin = video[0:int(video.shape[0] / 32), :]
    original_videos.append(origin)

original_coordinates = []
for i in range(len(coordinates)):
    stacked_coordinates = []
    for name in JointNames:
        coordinate_array = coordinates[i][name]
        stacked_coordinates.append(coordinate_array)

    coordinates_tuple = tuple(stacked_coordinates)
    stacked_array = np.hstack(coordinates_tuple)
    original_coordinates.append(stacked_array)


# Add random noise to the vector
def vector_noise(limb, limb_lengths):
    global data
    axes = ['x', 'y', 'z']
    data_dict = {}
    for i in range(3):
        op = random.choice(ops)
        data_dict['limb_new_{0}'.format(axes[i])] = op(limb[i], random.uniform(0, np.std(limb_lengths) / 4))
    limb_new = np.array([data_dict['limb_new_x'], data_dict['limb_new_y'], data_dict['limb_new_z']])
    return limb_new


# Scale the vector
def scale_vector(original_vector, new_vector_length):
    original_vector_length = np.linalg.norm(original_vector)
    scaled_vector = (new_vector_length / original_vector_length) * original_vector
    return scaled_vector


# Calculate new joint coordinates
def calculate_elbows(left_shoulder, right_shoulder, left_elbow, right_elbow,
                     left_upper_arm_new, right_upper_arm_new, index):
    left_upper_arm = left_elbow - left_shoulder
    right_upper_arm = right_elbow - right_shoulder

    LUpperArm = scale_vector(left_upper_arm, left_upper_arm_new)
    RUpperArm = scale_vector(right_upper_arm, right_upper_arm_new)

    LUpperArm_new = vector_noise(LUpperArm, limbs_list[index][:, 0])
    RUpperArm_new = vector_noise(RUpperArm, limbs_list[index][:, 5])

    LElbow = LUpperArm_new + left_shoulder
    RElbow = RUpperArm_new + right_shoulder

    return LElbow, RElbow


def calculate_wrists(left_elbow, right_elbow, left_wrist, right_wrist,
                     left_lower_arm_new, right_lower_arm_new, index):
    left_lower_arm = left_wrist - left_elbow
    right_lower_arm = right_wrist - right_elbow

    LLowerArm = scale_vector(left_lower_arm, left_lower_arm_new)
    RLowerArm = scale_vector(right_lower_arm, right_lower_arm_new)

    LLowerArm_new = vector_noise(LLowerArm, limbs_list[index][:, 1])
    RLowerArm_new = vector_noise(RLowerArm, limbs_list[index][:, 6])

    LWrist = LLowerArm_new + left_elbow
    RWrist = RLowerArm_new + right_elbow

    return LWrist, RWrist


def calculate_hips(left_shoulder, right_shoulder, left_hip, right_hip,
                   left_torso_new, right_torso_new, index):
    left_torso = left_hip - left_shoulder
    right_torso = right_hip - right_shoulder

    LTorso = scale_vector(left_torso, left_torso_new)
    RTorso = scale_vector(right_torso, right_torso_new)

    LTorso_new = vector_noise(LTorso, limbs_list[index][:, 2])
    RTorso_new = vector_noise(RTorso, limbs_list[index][:, 7])

    LHip = LTorso_new + left_shoulder
    RHip = RTorso_new + right_shoulder

    return LHip, RHip


def calculate_knees(left_hip, right_hip, left_knee, right_knee,
                    left_upper_leg_new, right_upper_leg_new, index):
    left_upper_leg = left_knee - left_hip
    right_upper_leg = right_knee - right_hip

    LUpperLeg = scale_vector(left_upper_leg, left_upper_leg_new)
    RUpperLeg = scale_vector(right_upper_leg, right_upper_leg_new)

    LUpperLeg_new = vector_noise(LUpperLeg, limbs_list[index][:, 3])
    RUpperLeg_new = vector_noise(RUpperLeg, limbs_list[index][:, 8])

    LKnee = LUpperLeg_new + left_hip
    RKnee = RUpperLeg_new + right_hip

    return LKnee, RKnee


def calculate_ankles(left_knee, right_knee, left_ankle, right_ankle,
                     left_lower_leg_new, right_lower_leg_new, index):
    left_lower_leg = left_ankle - left_knee
    right_lower_leg = right_ankle - right_knee

    LLowerLeg = scale_vector(left_lower_leg, left_lower_leg_new)
    RLowerLeg = scale_vector(right_lower_leg, right_lower_leg_new)

    LLowerLeg_new = vector_noise(LLowerLeg, limbs_list[index][:, 4])
    RLowerLeg_new = vector_noise(RLowerLeg, limbs_list[index][:, 9])

    LAnkle = LLowerLeg_new + left_knee
    RAnkle = RLowerLeg_new + right_knee

    return LAnkle, RAnkle


# Computation of all the coordinates in sequence
gait_videos = []
for i in range(len(videos)):
    gait_sequences = []
    counter = 0
    for j in range(32):
        data = []
        for k in range(int(videos[i].shape[0] / 32) * counter, int(videos[i].shape[0] / 32) * (counter + 1)):
            m = k - int(videos[i].shape[0] / 32) * counter
            shoulders = (coordinates[i]['LeftShoulder'][m, :], coordinates[i]['RightShoulder'][m, :])

            elbows = calculate_elbows(shoulders[0], shoulders[1],
                                      coordinates[i]['LeftElbow'][m, :],
                                      coordinates[i]['RightElbow'][m, :],
                                      videos[i][k, 0], videos[i][k, 5], i)
            wrists = calculate_wrists(elbows[0], elbows[1],
                                      coordinates[i]['LeftWrist'][m, :],
                                      coordinates[i]['RightWrist'][m, :],
                                      videos[i][k, 1], videos[i][k, 6], i)
            hips = calculate_hips(wrists[0], wrists[1],
                                  coordinates[i]['LeftHip'][m, :],
                                  coordinates[i]['RightHip'][m, :],
                                  videos[i][k, 2], videos[i][k, 7], i)
            knees = calculate_knees(hips[0], hips[1],
                                    coordinates[i]['LeftKnee'][m, :],
                                    coordinates[i]['RightKnee'][m, :],
                                    videos[i][k, 3], videos[i][k, 8], i)
            ankles = calculate_ankles(hips[0], hips[1],
                                      coordinates[i]['LeftAnkle'][m, :],
                                      coordinates[i]['RightAnkle'][m, :],
                                      videos[i][k, 4], videos[i][k, 9], i)

            data_row = np.hstack((shoulders[0], shoulders[1], elbows[0], elbows[1], wrists[0], wrists[1],
                                  hips[0], hips[1], knees[0], knees[1], ankles[0], ankles[1]))
            data.append(data_row)
        one_sequence = np.vstack(data)
        gait_sequences.append(one_sequence)
        counter += 1

    gait_array = np.vstack((original_coordinates[i], np.vstack(gait_sequences)))
    gait_videos.append(gait_array)

# Store the generated data to a list of dataframes
final_videos = []
for i in range(len(gait_videos)):
    gait_dataframe = pd.DataFrame(gait_videos[i], columns=cols[1:])
    final_videos.append(gait_dataframe)

for i in range(len(gait_videos)):
    final_videos[i].to_csv(final_data_wd + f'test_augmented_joints_{i+1}.csv', index=False)