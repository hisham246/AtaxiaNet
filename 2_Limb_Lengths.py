import os
import pandas as pd
import numpy as np

for i in range(1, 24):
    cwd = os.getcwd()  # Get the current working directory
    read_data = cwd + '\\Joint Coordinates\\Pre Augmentation\\Test\\' + f'test_coordinates_{i}.csv'
    data = pd.read_csv(read_data)
    # data = data.drop(labels=0, axis=0)
    path_data = cwd + '\\Limb Lengths\\Test\\' + f'test_limb_lengths_{i}.csv'

    joints = {'LeftShoulder': np.column_stack([list(np.float_(data['XLeftShoulder'].tolist())),
                                               list(np.float_(data['YLeftShoulder'].tolist())),
                                               list(np.float_(data['ZLeftShoulder'].tolist()))]),
              'RightShoulder': np.column_stack([list(np.float_(data['XRightShoulder'].tolist())),
                                                list(np.float_(data['XRightShoulder'].tolist())),
                                                list(np.float_(data['XRightShoulder'].tolist()))]),
              'LeftElbow': np.column_stack([list(np.float_(data['XLeftElbow'].tolist())),
                                            list(np.float_(data['YLeftElbow'].tolist())),
                                            list(np.float_(data['ZLeftElbow'].tolist()))]),
              'RightElbow': np.column_stack([list(np.float_(data['XRightElbow'].tolist())),
                                             list(np.float_(data['YRightElbow'].tolist())),
                                             list(np.float_(data['ZRightElbow'].tolist()))]),
              'LeftWrist': np.column_stack([list(np.float_(data['XLeftWrist'].tolist())),
                                            list(np.float_(data['YLeftWrist'].tolist())),
                                            list(np.float_(data['ZLeftWrist'].tolist()))]),
              'RightWrist': np.column_stack([list(np.float_(data['XRightWrist'].tolist())),
                                             list(np.float_(data['YRightWrist'].tolist())),
                                             list(np.float_(data['ZRightWrist'].tolist()))]),
              'LeftHip': np.column_stack([list(np.float_(data['XLeftHip'].tolist())),
                                          list(np.float_(data['YLeftHip'].tolist())),
                                          list(np.float_(data['ZLeftHip'].tolist()))]),
              'RightHip': np.column_stack([list(np.float_(data['XRightHip'].tolist())),
                                           list(np.float_(data['YRightHip'].tolist())),
                                           list(np.float_(data['ZRightHip'].tolist()))]),
              'LeftKnee': np.column_stack([list(np.float_(data['XLeftKnee'].tolist())),
                                           list(np.float_(data['YLeftKnee'].tolist())),
                                           list(np.float_(data['ZLeftKnee'].tolist()))]),
              'RightKnee': np.column_stack([list(np.float_(data['XRightKnee'].tolist())),
                                            list(np.float_(data['YRightKnee'].tolist())),
                                            list(np.float_(data['ZRightKnee'].tolist()))]),
              'LeftAnkle': np.column_stack([list(np.float_(data['XLeftAnkle'].tolist())),
                                            list(np.float_(data['YLeftAnkle'].tolist())),
                                            list(np.float_(data['ZLeftAnkle'].tolist()))]),
              'RightAnkle': np.column_stack([list(np.float_(data['XRightAnkle'].tolist())),
                                             list(np.float_(data['YRightAnkle'].tolist())),
                                             list(np.float_(data['ZRightAnkle'].tolist()))])}

    left_torso = joints["LeftHip"] - joints["LeftShoulder"]
    right_torso = joints["RightHip"] - joints["RightShoulder"]
    left_upper_arm = joints["LeftElbow"] - joints["LeftShoulder"]
    right_upper_arm = joints["RightElbow"] - joints["RightShoulder"]
    left_lower_arm = joints["LeftWrist"] - joints["LeftElbow"]
    right_lower_arm = joints["RightWrist"] - joints["RightElbow"]
    left_upper_leg = joints["LeftKnee"] - joints["LeftHip"]
    right_upper_leg = joints["RightKnee"] - joints["RightHip"]
    left_lower_leg = joints["LeftAnkle"] - joints["LeftKnee"]
    right_lower_leg = joints["RightAnkle"] - joints["RightKnee"]

    LTorso = []
    RTorso = []
    LUpperArm = []
    RUpperArm = []
    LLowerArm = []
    RLowerArm = []
    LUpperLeg = []
    RUpperLeg = []
    LLowerLeg = []
    RLowerLeg = []

    limb_vectors = [left_torso, right_torso, left_upper_arm, right_upper_arm,
                    left_lower_arm, right_lower_arm, left_upper_leg, right_upper_leg,
                    left_lower_leg, right_lower_leg]
    limb_lengths = [LTorso, RTorso, LUpperArm, RUpperArm, LLowerArm, RLowerArm,
                    LUpperLeg, RUpperLeg, LLowerLeg, RLowerLeg]

    for i in range(len(limb_vectors)):
        for j in range(len(data['Timestamp'])):
            temp = np.linalg.norm(limb_vectors[i][j, :])
            limb_lengths[i].append(temp)

    limb_data = {'Timestamp': data['Timestamp'].to_list(), 'LeftUpperArm': LUpperArm, 'LeftLowerArm': LLowerArm,
                 'LeftTorso': LTorso, 'LeftUpperLeg': LUpperLeg, 'LeftLowerLeg': LLowerLeg,
                 'RightUpperArm': RUpperArm, 'RightLowerArm': RLowerArm, 'RightTorso': RTorso,
                 'RightUpperLeg': RUpperLeg, 'RightLowerLeg': RLowerLeg}

    limbs = pd.DataFrame.from_dict(limb_data)
    limbs.to_csv(path_data, index=False)
