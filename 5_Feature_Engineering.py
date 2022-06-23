import os
import numpy as np
import pandas as pd

for i in range(21, 24):
    cwd = os.getcwd()  # Get the current working directory
    original_path_ataxia = cwd + '\\Joint Coordinates\\Pre Augmentation\\Test\\' + f'test_coordinates_{i}.csv'
    # original_path_normal = cwd + '\\Joint Coordinates\\Pre Augmentation\\Test\\' + f'test_coordinates_{i}.csv'

    path_ataxia_data = cwd + '\\Joint Coordinates\\Test\\' + f'test_augmented_joints_{i}.csv'
    # path_normal_data = cwd + '\\Joint Coordinates\\Test\\' + f'test_augmented_joints_{i}.csv'

    final_path_ataxia = cwd + '\\Final Data\\Test\\Ataxia\\' + f'ataxia_features_{i}.csv'
    # final_path_normal = cwd + '\\Final Data\\Test\\Normal\\' + f'normal_features_{i}.csv'

    ataxia_data = pd.read_csv(path_ataxia_data).values
    # normal_data = pd.read_csv(path_normal_data).values

    Ataxia_Data = pd.read_csv(path_ataxia_data)
    # Normal_Data = pd.read_csv(path_normal_data)

    original_data_ataxia = pd.read_csv(original_path_ataxia)
    # original_data_normal = pd.read_csv(original_path_normal)

    original_data_ataxia = original_data_ataxia.drop(labels=0, axis=0)
    # original_data_normal = original_data_normal.drop(labels=0, axis=0)

    cols = ['Timestamp', 'XLeftShoulder', 'YLeftShoulder', 'ZLeftShoulder', 'XRightShoulder', 'YRightShoulder',
            'ZRightShoulder', 'XLeftElbow', 'YLeftElbow', 'ZLeftElbow', 'XRightElbow', 'YRightElbow','ZRightElbow',
            'XLeftWrist', 'YLeftWrist', 'ZLeftWrist', 'XRightWrist', 'YRightWrist', 'ZRightWrist', 'XLeftHip',
            'YLeftHip', 'ZLeftHip', 'XRightHip', 'YRightHip', 'ZRightHip', 'XLeftKnee', 'YLeftKnee', 'ZLeftKnee',
            'XRightKnee', 'YRightKnee', 'ZRightKnee', 'XLeftAnkle', 'YLeftAnkle', 'ZLeftAnkle', 'XRightAnkle',
            'YRightAnkle', 'ZRightAnkle']

    original_data_ataxia.columns = cols
    # original_data_normal.columns = cols

    timestamps_ataxia = original_data_ataxia['Timestamp'].tolist()
    # timestamps_normal = original_data_normal['Timestamp'].tolist()

    num1 = timestamps_ataxia[1] - timestamps_ataxia[0]
    TimeStamps_Ataxia = []
    for i in range(ataxia_data.shape[0]):
        s = (num1 * i) + num1
        TimeStamps_Ataxia.append(s)

    # num2 = timestamps_normal[1] - timestamps_normal[0]
    # TimeStamps_Normal = []
    # for i in range(normal_data.shape[0]):
    #     s = (num2 * i) + num2
    #     TimeStamps_Normal.append(s)

    columns = ['LeftShoulder', 'RightShoulder', 'LeftElbow', 'RightElbow', 'LeftWrist', 'RightWrist', 'LeftHip',
               'RightHip', 'LeftKnee', 'RightKnee', 'LeftAnkle', 'RightAnkle']

    ataxia_data_dict = {}
    for i, m in enumerate(columns):
        index = [j for j in range((i + 1) * 3 - 3, (i + 1) * 3)]
        ataxia_data_dict[m] = ataxia_data[:, index]

    # normal_data_dict = {}
    # for i, m in enumerate(columns):
    #     index = [j for j in range((i + 1) * 3 - 3, (i + 1) * 3)]
    #     normal_data_dict[m] = normal_data[:, index]

    left_upper_arm_ataxia = ataxia_data_dict['LeftElbow'] - ataxia_data_dict['LeftShoulder']
    right_upper_arm_ataxia = ataxia_data_dict['RightElbow'] - ataxia_data_dict['RightShoulder']
    left_lower_arm_ataxia = ataxia_data_dict['LeftWrist'] - ataxia_data_dict['LeftElbow']
    right_lower_arm_ataxia = ataxia_data_dict['RightWrist'] - ataxia_data_dict['RightElbow']
    left_torso_ataxia = ataxia_data_dict['LeftHip'] - ataxia_data_dict['LeftShoulder']
    right_torso_ataxia = ataxia_data_dict['RightHip'] - ataxia_data_dict['RightShoulder']
    left_upper_leg_ataxia = ataxia_data_dict['LeftKnee'] - ataxia_data_dict['LeftHip']
    right_upper_leg_ataxia = ataxia_data_dict['RightKnee'] - ataxia_data_dict['RightHip']
    left_lower_leg_ataxia = ataxia_data_dict['LeftAnkle'] - ataxia_data_dict['LeftKnee']
    right_lower_leg_ataxia = ataxia_data_dict['RightAnkle'] - ataxia_data_dict['RightKnee']

    # left_upper_arm_normal = normal_data_dict['LeftElbow'] - normal_data_dict['LeftShoulder']
    # right_upper_arm_normal = normal_data_dict['RightElbow'] - normal_data_dict['RightShoulder']
    # left_lower_arm_normal = normal_data_dict['LeftWrist'] - normal_data_dict['LeftElbow']
    # right_lower_arm_normal = normal_data_dict['RightWrist'] - normal_data_dict['RightElbow']
    # left_torso_normal = normal_data_dict['LeftHip'] - normal_data_dict['LeftShoulder']
    # right_torso_normal = normal_data_dict['RightHip'] - normal_data_dict['RightShoulder']
    # left_upper_leg_normal = normal_data_dict['LeftKnee'] - normal_data_dict['LeftHip']
    # right_upper_leg_normal = normal_data_dict['RightKnee'] - normal_data_dict['RightHip']
    # left_lower_leg_normal = normal_data_dict['LeftAnkle'] - normal_data_dict['LeftKnee']
    # right_lower_leg_normal = normal_data_dict['RightAnkle'] - normal_data_dict['RightKnee']

    theta_left_shoulder_ataxia = []
    theta_right_shoulder_ataxia = []
    theta_left_hip_ataxia = []
    theta_right_hip_ataxia = []
    theta_left_knee_ataxia = []
    theta_right_knee_ataxia = []

    # theta_left_shoulder_normal = []
    # theta_right_shoulder_normal = []
    # theta_left_hip_normal = []
    # theta_right_hip_normal = []
    # theta_left_knee_normal = []
    # theta_right_knee_normal = []

    # Ataxic Gait
    for i in range(left_upper_arm_ataxia.shape[0]):
        temp1 = (np.dot(left_upper_arm_ataxia[i, :], left_torso_ataxia[i, :])) / (np.linalg.norm(left_upper_arm_ataxia[i, :]) * np.linalg.norm(left_torso_ataxia[i, :]))
        temp = np.arccos(temp1)
        theta_left_shoulder_ataxia.append(temp)

    for i in range(left_upper_arm_ataxia.shape[0]):
        temp1 = (np.dot(right_upper_arm_ataxia[i, :], right_torso_ataxia[i, :])) / (np.linalg.norm(right_upper_arm_ataxia[i, :]) * np.linalg.norm(right_torso_ataxia[i, :]))
        temp = np.arccos(temp1)
        theta_right_shoulder_ataxia.append(temp)

    for i in range(left_upper_arm_ataxia.shape[0]):
        temp1 = (np.dot(left_torso_ataxia[i, :], left_upper_leg_ataxia[i, :])) / (np.linalg.norm(left_torso_ataxia[i, :]) * np.linalg.norm(left_upper_leg_ataxia[i, :]))
        temp = np.arccos(temp1)
        theta_left_hip_ataxia.append(temp)

    for i in range(left_upper_arm_ataxia.shape[0]):
        temp1 = (np.dot(right_torso_ataxia[i, :], right_upper_leg_ataxia[i, :])) / (np.linalg.norm(right_torso_ataxia[i, :]) * np.linalg.norm(right_upper_leg_ataxia[i, :]))
        temp = np.arccos(temp1)
        theta_right_hip_ataxia.append(temp)

    for i in range(left_upper_arm_ataxia.shape[0]):
        temp1 = (np.dot(left_upper_leg_ataxia[i, :], left_lower_leg_ataxia[i, :])) / (np.linalg.norm(left_upper_leg_ataxia[i, :]) * np.linalg.norm(left_lower_leg_ataxia[i, :]))
        temp = np.arccos(temp1)
        theta_left_knee_ataxia.append(temp)

    for i in range(right_upper_arm_ataxia.shape[0]):
        temp1 = (np.dot(right_upper_leg_ataxia[i, :], right_lower_leg_ataxia[i, :])) / (np.linalg.norm(right_upper_leg_ataxia[i, :]) * np.linalg.norm(right_lower_leg_ataxia[i, :]))
        temp = np.arccos(temp1)
        theta_right_knee_ataxia.append(temp)

    # # Normal Gait
    # for i in range(left_upper_arm_normal.shape[0]):
    #     temp1 = (np.dot(left_upper_arm_normal[i, :], left_torso_normal[i, :])) / (
    #                 np.linalg.norm(left_upper_arm_normal[i, :]) * np.linalg.norm(left_torso_normal[i, :]))
    #     temp = np.arccos(temp1)
    #     theta_left_shoulder_normal.append(temp)
    #
    # for i in range(left_upper_arm_normal.shape[0]):
    #     temp1 = (np.dot(right_upper_arm_normal[i, :], right_torso_normal[i, :])) / (
    #                 np.linalg.norm(right_upper_arm_normal[i, :]) * np.linalg.norm(right_torso_normal[i, :]))
    #     temp = np.arccos(temp1)
    #     theta_right_shoulder_normal.append(temp)
    #
    # for i in range(left_upper_arm_normal.shape[0]):
    #     temp1 = (np.dot(left_torso_normal[i, :], left_upper_leg_normal[i, :])) / (
    #                 np.linalg.norm(left_torso_normal[i, :]) * np.linalg.norm(left_upper_leg_normal[i, :]))
    #     temp = np.arccos(temp1)
    #     theta_left_hip_normal.append(temp)
    #
    # for i in range(left_upper_arm_normal.shape[0]):
    #     temp1 = (np.dot(right_torso_normal[i, :], right_upper_leg_normal[i, :])) / (
    #                 np.linalg.norm(right_torso_normal[i, :]) * np.linalg.norm(right_upper_leg_normal[i, :]))
    #     temp = np.arccos(temp1)
    #     theta_right_hip_normal.append(temp)
    #
    # for i in range(left_upper_arm_normal.shape[0]):
    #     temp1 = (np.dot(left_upper_leg_normal[i, :], left_lower_leg_normal[i, :])) / (
    #                 np.linalg.norm(left_upper_leg_normal[i, :]) * np.linalg.norm(left_lower_leg_normal[i, :]))
    #     temp = np.arccos(temp1)
    #     theta_left_knee_normal.append(temp)
    #
    # for i in range(right_upper_arm_normal.shape[0]):
    #     temp1 = (np.dot(right_upper_leg_normal[i, :], right_lower_leg_normal[i, :])) / (
    #             np.linalg.norm(right_upper_leg_normal[i, :]) * np.linalg.norm(right_lower_leg_normal[i, :]))
    #     temp = np.arccos(temp1)
    #     theta_right_knee_normal.append(temp)
    #
    joint_angles_ataxia = [theta_left_shoulder_ataxia, theta_right_shoulder_ataxia, theta_left_hip_ataxia,
                           theta_right_hip_ataxia, theta_left_knee_ataxia, theta_right_knee_ataxia]
    # joint_angles_normal = [theta_left_shoulder_normal, theta_right_shoulder_normal, theta_left_hip_normal,
    #                        theta_right_hip_normal, theta_left_knee_normal, theta_right_knee_normal]

    # 'Normal': np.column_stack(joint_angles_normal)
    JointAngles = {'Ataxia': np.column_stack(joint_angles_ataxia)}

    # Spatio-temporal parameters
    step_length_ataxia = []
    step_width_ataxia = []
    feet_clearance_ataxia = []
    left_stride_speed_ataxia = []
    right_stride_speed_ataxia = []

    for i in range(ataxia_data_dict['LeftAnkle'].shape[0]):
        sw = abs(ataxia_data_dict['LeftAnkle'][i, 0] - ataxia_data_dict['RightAnkle'][i, 0])  # X axis
        fc = abs(ataxia_data_dict['LeftAnkle'][i, 1] - ataxia_data_dict['RightAnkle'][i, 1])  # Y axis
        sl = abs(ataxia_data_dict['LeftAnkle'][i, 2] - ataxia_data_dict['RightAnkle'][i, 2])  # Z axis
        if i == 0:
            lss = (np.linalg.norm(ataxia_data_dict['LeftAnkle'][i, :])) / (TimeStamps_Ataxia[0])
            rss = (np.linalg.norm(ataxia_data_dict['RightAnkle'][i, :])) / (TimeStamps_Ataxia[0])
        else:
            lss = (np.linalg.norm(ataxia_data_dict['LeftAnkle'][i, :])) / (TimeStamps_Ataxia[i] - TimeStamps_Ataxia[i-1])
            rss = (np.linalg.norm(ataxia_data_dict['RightAnkle'][i, :])) / (TimeStamps_Ataxia[i] - TimeStamps_Ataxia[i-1])

        step_width_ataxia.append(sw)
        feet_clearance_ataxia.append(fc)
        step_length_ataxia.append(sl)
        left_stride_speed_ataxia.append(lss)
        right_stride_speed_ataxia.append(rss)

    # step_length_normal = []
    # step_width_normal = []
    # feet_clearance_normal = []
    # left_stride_speed_normal = []
    # right_stride_speed_normal = []
    #
    # for i in range(normal_data_dict['LeftAnkle'].shape[0]):
    #     sw = abs(normal_data_dict['LeftAnkle'][i, 0] - normal_data_dict['RightAnkle'][i, 0])  # X axis
    #     fc = abs(normal_data_dict['LeftAnkle'][i, 1] - normal_data_dict['RightAnkle'][i, 1])  # Y axis
    #     sl = abs(normal_data_dict['LeftAnkle'][i, 2] - normal_data_dict['RightAnkle'][i, 2])  # Z axis
    #     if i == 0:
    #         lss = (np.linalg.norm(normal_data_dict['LeftAnkle'][i, :])) / (TimeStamps_Normal[0])
    #         rss = (np.linalg.norm(normal_data_dict['RightAnkle'][i, :])) / (TimeStamps_Normal[0])
    #     else:
    #         lss = (np.linalg.norm(normal_data_dict['LeftAnkle'][i, :])) / (TimeStamps_Normal[i] - TimeStamps_Normal[i - 1])
    #         rss = (np.linalg.norm(normal_data_dict['RightAnkle'][i, :])) / (TimeStamps_Normal[i] - TimeStamps_Normal[i - 1])
    #
    #     step_width_normal.append(sw)
    #     feet_clearance_normal.append(fc)
    #     step_length_normal.append(sl)
    #     left_stride_speed_normal.append(lss)
    #     right_stride_speed_normal.append(rss)

    spatio_temporal_parameters_ataxia = [step_length_ataxia, step_width_ataxia, feet_clearance_ataxia,
                                         left_stride_speed_ataxia, right_stride_speed_ataxia]

    # spatio_temporal_parameters_normal = [step_length_normal, step_width_normal, feet_clearance_normal,
    #                                      left_stride_speed_normal, right_stride_speed_normal]
    # 'Normal': np.column_stack(spatio_temporal_parameters_normal)
    SpatioTemporal_Parameters = {'Ataxia': np.column_stack(spatio_temporal_parameters_ataxia)}
    # 'Normal': np.hstack((JointAngles['Normal'], SpatioTemporal_Parameters['Normal']))
    Features = {'Ataxia': np.hstack((JointAngles['Ataxia'], SpatioTemporal_Parameters['Ataxia']))}

    feature_columns = ['ThetaLeftShoulder', 'ThetaRightShoulder', 'ThetaLeftHip', 'ThetaRightHip', 'ThetaLeftKnee',
                       'ThetaRightKnee', 'StepLength', 'StepWidth', 'FeetClearance', 'LeftStrideSpeed',
                       'RightStrideSpeed']

    Feature_Data_Ataxia = pd.DataFrame(Features['Ataxia'], columns=feature_columns)
    # Feature_Data_Normal = pd.DataFrame(Features['Normal'], columns=feature_columns)

    Ataxia_DataSet = pd.concat([Ataxia_Data, Feature_Data_Ataxia], axis=1)
    # Normal_DataSet = pd.concat([Normal_Data, Feature_Data_Normal], axis=1)

    Ataxia_DataSet.to_csv(final_path_ataxia, index=False)
    # Normal_DataSet.to_csv(final_path_normal, index=False)


