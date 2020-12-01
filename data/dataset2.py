import os
import os.path as osp
from abc import ABC
from functools import partial
from multiprocessing import Pool

import torch
from torch_geometric.data import Dataset, Data
from tqdm import tqdm

# from data.skeleton2 import process_skeleton, skeleton_parts
from data.skeleton2 import process_skeleton, skeleton_parts

torch.multiprocessing.set_sharing_strategy('file_system')


def num_processes():
    return os.cpu_count() - 4


class SkeletonDataset(Dataset, ABC):
    def __init__(self,
                 root,
                 name,
                 use_motion_vector=True,
                 transform=None,
                 pre_transform=None,
                 benchmark=None,
                 sample='train'):
        self.name = name
        self.benchmark = benchmark
        self.sample = sample

        self.num_joints = 25 if 'ntu' in self.name else 18
        self.skeleton_ = skeleton_parts(num_joints=self.num_joints,
                                        dataset=self.name)
        print('processed the adjacency matrices of skeleton')
        self.use_motion_vector = use_motion_vector
        # raw_path = osp.join(os.getcwd(), root, "raw")
        # if not osp.exists(raw_path):
        #     os.mkdir(raw_path)
        super(SkeletonDataset, self).__init__(root, transform, pre_transform)
        if 'ntu' in self.name:
            path = osp.join(self.processed_dir, '{}.pt'.format(self.name))
            self.data, self.labels = torch.load(path)
        # else:
        #     self.labels = torch.load(osp.join(self.root, 'kinetics_labels.pt'))

    @property
    def raw_file_names(self):
        if 'kinetics' in self.name:
            return [f for f in os.listdir(self.raw_dir)]
        fp = lambda x: osp.join(self.root, 'raw', x)
        return [fp(f) for f in os.listdir(self.raw_dir)]  # if osp.isfile(fp(f))]

    @property
    def processed_file_names(self):
        if 'kinetics' in self.name:
            return [f for f in os.listdir(self.processed_dir)]
        else:
            return '{}.pt'.format(self.name)

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        pool = Pool(processes=num_processes())
        partial_func = partial(process_skeleton,
                               num_joints=self.num_joints,
                               dataset_name=self.name,
                               num_features=3,
                               root=self.root,
                               benchmark=self.benchmark,
                               sample=self.sample)
        progress_bar = tqdm(pool.imap(func=partial_func, iterable=self.raw_file_names),
                            total=len(self.raw_file_names))
        skeletons, labels = [], []
        for (data, label, uid) in progress_bar:
            if data is None:
                continue

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data = Data(x=data, y=label)  # , edge_index=self.skeleton_)
            if 'kinetics' in self.name:
                torch.save(data, osp.join(self.processed_dir,
                                          '{}.pt'.format(uid)))
            else:
                skeletons.append(data)
                labels.append(label)

        if 'ntu' in self.name:
            torch.save([skeletons, torch.FloatTensor(labels)],
                       osp.join(self.processed_dir,
                                self.processed_file_names))

    def len(self):
        if 'kinetics' in self.name:
            return len(self.processed_file_names)
        else:
            return len(self.data)

    def get(self, idx):
        if 'kinetics' in self.name:
            if isinstance(idx, int):
                return torch.load(osp.join(self.processed_dir,
                                           '{}.pt'.format(self.processed_file_names[idx])))
            return [torch.load(osp.join(self.processed_dir,
                                        '{}.pt'.format(self.processed_file_names[i]))) for i in idx]
        return self.data[idx]


def test():
    from argparse import ArgumentParser
    from torch_geometric.data import DataLoader
    parser = ArgumentParser()
    parser.add_argument('--root', dest='root', default='../dataset/ntu_60',
                        type=str, help='Dataset')
    parser.add_argument('--dataset', dest='dataset', default='ntu_60',
                        type=str, help='Dataset')
    args = parser.parse_args()
    ds = SkeletonDataset(root=args.root,
                         name=args.dataset)
    loader = DataLoader(ds[0: 8], batch_size=4)
    for b in loader:
        print(b.batch)


if __name__ == "__main__":
    test()
