import os
from abc import ABC
import sys
from utility.nturgbd.prepare_ntu import gendata, edge_index, data_sample
import torch
import torch.utils.data
from torch_geometric.data import DataLoader
from torch_geometric.data import Dataset

"""NTURGBD dataset joints connection list
 1-base of the spine
 2-middle of the spine
 3-neck
 4-head
 5-left shoulder
 6-left elbow
 7-left wrist
 8-left hand
 9-right shoulder
 10-right elbow
 11-right wrist
 12-right hand
 13-left hip
 14-left knee
 15-left ankle
 16-left foot
 17-right hip
 18-right knee
 19-right ankle
 20-right foot
 21-spine
 22-tip of the left hand
 23-left thumb
 24-tip of the right hand
 25-right thumb
"""

num_joint = 25


class NTUDataset(Dataset, ABC):
    def __init__(self,
                 root,
                 batch_size,
                 transform=None,
                 pre_transform=None,
                 benchmark='cv',
                 part='val',
                 ignored_sample_path=None,
                 plan="synergy_matrix"):
        self.batch_size = batch_size
        self.raw_path = root + "/raw"
        self.ignored_sample_path = ignored_sample_path
        self.benchmark = benchmark
        self.part = part
        self.plan = plan
        self.out_path = os.path.join(root, 'processed', self.plan, self.benchmark, self.part)

        self.rawFileNames = data_sample(
            self.raw_path,
            self.out_path,
            ignored_sample_path=self.ignored_sample_path,
            benchmark=self.benchmark,
            part=self.part)

        super(NTUDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.rawFileNames[0]

    @property
    def processed_file_names(self):
        processed_data = []
        for i in range(len(self.rawFileNames[0])):
            processed_data.append('data_{}.pt'.format(i))
        return processed_data

    def process(self):
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
            gendata(
                self.raw_path,
                self.out_path,
                ignored_sample_path=self.ignored_sample_path,
                benchmark=self.benchmark,
                part=self.part,
                plan=self.plan)

    def len(self):
        return len(self.raw_paths)

    def get(self, idx):
        data = torch.load(os.path.join(self.out_path, 'data_{}.pt'.format(idx)))
        data.num_nodes = num_joint
        return data


if __name__ == '__main__':

    n_batch_size = 2
    ntu_dataset = NTUDataset(
        "/Users/yangzhiping/Documents/deepl/Apb-gcn/data/utility/nturgbd",
        batch_size=n_batch_size,
        benchmark='cv',
        part='val',
        ignored_sample_path=None,
        plan="synergy_matrix")

    ntu_dataloader = DataLoader(ntu_dataset, batch_size=n_batch_size, shuffle=True)

    count = 0
    i = 0
    batch = None
    for b in ntu_dataloader:
        batch = b
        print('Index', count)
        print(len(batch))
        print(batch)
        count += 1
    print(count)
