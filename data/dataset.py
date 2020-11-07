from abc import ABC

import os
import os.path as osp
from time import sleep

import torch
from torch_geometric.data import Dataset, Data
from tqdm import tqdm
from data.skeleton import process_skeleton, skeleton_parts


class SkeletonDataset(Dataset, ABC):
    def __init__(self,
                 root,
                 name,
                 use_motion_vector=True,
                 transform=None,
                 pre_transform=None,
                 benchmark='cv',
                 sample='train'):
        self.name = name
        self.benchmark = benchmark
        self.training_subjects = {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38}
        self.training_cameras = {2, 3}
        self.sample = sample

        self.num_joints = 25
        self.skeleton_ = skeleton_parts()
        self.use_motion_vector = use_motion_vector
        self.cached_missing_files = None

        if not osp.exists(osp.join(root, "raw")):
            os.mkdir(osp.join(root, "raw"))
        super(SkeletonDataset, self).__init__(root, transform, pre_transform)
        path = osp.join(self.processed_dir, '{}.pt'.format(self.name))
        self.data, self.labels = torch.load(path)

    @property
    def missing_skeleton_file_names(self):
        if self.cached_missing_files is not None:
            return self.cached_missing_files
        f = open(osp.join(self.root, 'samples_with_missing_skeletons.txt'), 'r')
        lines = f.readlines()
        self.cached_missing_files = set([ln[:-1] for ln in lines])
        f.close()
        return self.cached_missing_files

    @property
    def raw_file_names(self):
        fp = lambda x: osp.join(self.root, 'raw', x)
        return [fp(f) for f in os.listdir(self.raw_dir)]  # if osp.isfile(fp(f))]

    @property
    def processed_file_names(self):
        return '{}.pt'.format(self.name)

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        progress_bar = tqdm(self.raw_file_names)
        skeletons, labels = [], []
        i = 0
        for f in progress_bar:
            if 'ntu' in self.name:
                fl = f[-29:-9]
                if fl in self.missing_skeleton_file_names:
                    print('Skip file: ', fl)
                    continue

                # action_class = int(fl[fl.find('A') + 1: fl.find('A') + 4])
                subject_id = int(fl[fl.find('P') + 1: fl.find('P') + 4])
                camera_id = int(fl[fl.find('C') + 1: fl.find('C') + 4])

                if self.benchmark == 'cv':
                    if self.sample == 'train':
                        if camera_id not in self.training_cameras:
                            continue
                    else:
                        if camera_id in self.training_cameras:
                            continue

                if self.benchmark == 'cs':
                    if self.sample == 'train':
                        if subject_id not in self.training_subjects:
                            continue
                    else:
                        if subject_id in self.training_subjects:
                            continue
            elif 'kinetics' in self.name:
                import json
                with open(osp.join(self.root,
                                   'kinetics_{}_label.json'.format(self.sample)), 'r') as b:
                    _labels = json.load(b)
            # Read data from `raw_path`.
            sleep(1e-4)
            progress_bar.set_description("processing %s" % f)
            data, label = process_skeleton(f,
                                           num_joints=self.num_joints,
                                           use_motion_vector=self.use_motion_vector)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data = Data(x=data)  # , edge_index=self.skeleton_)
            skeletons.append(data)
            labels.append(label)
            i += 1

        torch.save([skeletons, torch.FloatTensor(labels)],
                   osp.join(self.processed_dir,
                            self.processed_file_names))

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]


def test():
    from torch_geometric.data import DataLoader
    ds = SkeletonDataset(root='../dataset',
                         name='ntu')
    loader = DataLoader(ds, batch_size=4)
    for b in loader:
        print(b.batch)


if __name__ == "__main__":
    test()
