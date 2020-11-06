import argparse
import os

import numpy as np
import torch
from utility.nturgbd.pairs import Pairs
from utility.nturgbd.read_skeleton import *
from torch_geometric.data import Data
from tqdm import tqdm

edge_index = torch.tensor([(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                           (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                           (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                           (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)], dtype=torch.long).transpose(1, 0) - 1

training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]
max_body = 2
num_joint = 25
max_frame = 300


def gendata(data_path,
            out_path,
            plan="synergy_matrix",
            ignored_sample_path=None,
            benchmark='cv',
            part='val'):
    sample_name, sample_label = data_sample(
        data_path,
        out_path,
        ignored_sample_path,
        benchmark,
        part)
    if plan == "synergy_matrix":
        pyg_x = torch.zeros((num_joint, 10, max_frame, max_body))
    if plan == "transformer":
        pyg_x = torch.zeros((num_joint, 7, max_frame, max_body))

    for i in tqdm(range(len(sample_name))):
        s = sample_name[i]
        data = read_xyz(os.path.join(data_path, s), plan=plan, max_body=max_body, num_joint=num_joint)
        modified_data = data.permute(2, 0, 1, 3)  # [max_frame,num_joint,num_features,max_body,clips]
        pyg_x[:, :, 0:data.shape[1], : ] = modified_data  # data

        if plan == "synergy_matrix":
            get_pair = Pairs()
            synergy_matrix_1 = torch.zeros(0, 300)
            synergy_matrix_2 = torch.zeros(0, 300)
            for joint_1, joint_2 in get_pair.total_collection:
                # pair : (joint 1, joint 2)
                pair_synergy = torch.mul(pyg_x[joint_1][7], pyg_x[joint_2][7]) + torch.mul(pyg_x[joint_1][8],
                                                                                        pyg_x[joint_2][8]) + torch.mul(
                    pyg_x[joint_1][9], pyg_x[joint_2][9])
                pair_synergy_person_1 = torch.unsqueeze(pair_synergy.permute(1, 0)[0], 0)
                pair_synergy_person_2 = torch.unsqueeze(pair_synergy.permute(1, 0)[1], 0)
                synergy_matrix_1 = torch.cat((synergy_matrix_1, pair_synergy_person_1))
                synergy_matrix_2 = torch.cat((synergy_matrix_2, pair_synergy_person_2))

        if plan != "synergy_matrix":
            pyg_data = Data(x=pyg_x, edge_index=edge_index.t().contiguous(), y=sample_label[i])

        if plan == "synergy_matrix":
            pyg_data = Data(x=torch.unsqueeze(torch.stack((synergy_matrix_1, synergy_matrix_2)), 0), edge_index=edge_index.t().contiguous(),
                            y=sample_label[i])
        torch.save(pyg_data, os.path.join(out_path, 'data_{}.pt'.format(i)))


def data_sample(data_path,
                out_path,
                ignored_sample_path=None,
                benchmark='cv',
                part='val'):
    if ignored_sample_path is not None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []
        sample_name = []
        sample_label = []

    for filename in os.listdir(data_path):
        if filename in ignored_samples:
            continue
        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(
            filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])

        if benchmark == 'cv':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'cs':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)
    return [sample_name, sample_label]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NTURGB+D Data Converter.')
    parser.add_argument(
        '--data_path', default='/home/lawbuntu/Downloads/pytorch_geometric-master/docker/raw')
    parser.add_argument(
        '--ignored_sample_path',
        default=None)
    parser.add_argument('--out_folder', default='/home/lawbuntu/Downloads/pytorch_geometric-master/docker/processed')

    benchmark = ['cs', 'cv']
    part = ['train', 'val']
    arg = parser.parse_args()

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            gendata(
                arg.data_path,
                out_path,
                arg.ignored_sample_path,
                benchmark=b,
                part=p,
                plan="synergy_matrix")
