import math
import torch
import numpy as np


def read_skeleton(file):
    with open(file, 'r') as f:
        skeleton_sequence = {'numFrame': int(f.readline()), 'frameInfo': []}
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {'numBody': int(f.readline()), 'bodyInfo': []}
            for m in range(frame_info['numBody']):
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence


def read_xyz(file, max_body=2, num_joint=25, plan = "synergy_matrix"):
    seq_info = read_skeleton(file)
    if plan == "synergy_matrix":
        data = torch.zeros((10, seq_info['numFrame'], num_joint, max_body))
    if plan == "transformer":
        data = torch.zeros((7, seq_info['numFrame'], num_joint, max_body))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[0, n, j, m] = torch.tensor(v['x'])
                    data[1, n, j, m] = torch.tensor(v['y'])
                    data[2, n, j, m] = torch.tensor(v['z'])
                    if n > 0:
                        motionVector = data[:, n - 1, j, m] - data[:, n, j, m]
                        x = motionVector[0]
                        y = motionVector[1]
                        z = motionVector[2]
                        magnitude = math.sqrt(x ** 2 + y ** 2 + z ** 2)

                        if magnitude > 0:
                            xyAngle = math.acos(z / magnitude)
                            yzAngle = math.acos(x / magnitude)
                            xzAngle = math.acos(y / magnitude)
                        if magnitude == 0:
                            xyAngle = 0
                            yzAngle = 0
                            xzAngle = 0
                        if plan == "synergy_matrix":
                            # data[3:, n - 1, j, m] = [xyAngle, yzAngle, xzAngle, magnitude, x, y, z]
                            data[3, n - 1, j, m] = torch.tensor(xyAngle)
                            data[4, n - 1, j, m] = torch.tensor(yzAngle)
                            data[5, n - 1, j, m] = torch.tensor(xzAngle)
                            data[6, n - 1, j, m] = torch.tensor(magnitude)
                            data[7, n - 1, j, m] = x
                            data[8, n - 1, j, m] = y
                            data[9, n - 1, j, m] = z

                        if plan == "transformer":    
                            # data[3:, n - 1, j, m] = [xyAngle, yzAngle, xzAngle, magnitude]
                            data[3, n - 1, j, m] = torch.tensor(xyAngle)
                            data[4, n - 1, j, m] = torch.tensor(yzAngle)
                            data[5, n - 1, j, m] = torch.tensor(xzAngle)
                            data[6, n - 1, j, m] = torch.tensor(magnitude)

                else:
                    pass
    return data
