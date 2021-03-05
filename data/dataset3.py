import os
import os.path as osp
from abc import ABC
from functools import partial
from multiprocessing import Pool

import numpy as np
import torch
from einops import rearrange
from torch_geometric.data import Dataset, Data
from torch_sparse import spspmm
from tqdm import tqdm


torch.multiprocessing.set_sharing_strategy('file_system')


def gen_bone_data(torch_data, paris, benchmark):
    T, N = torch_data.shape[0], torch_data.shape[1]
    bone_data = torch.zeros((T, N, 3), dtype=torch.float32)
    for v1, v2 in paris[benchmark]:
        if benchmark != 'kinetics':
            v1 -= 1
            v2 -= 1
        bone_data[:, v1, :] = torch_data[:, v1, :] - torch_data[:, v2, :]

    torch_data = torch.cat((torch_data, bone_data), 2)
    return torch_data


def torch_unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / torch.linalg.norm(vector)


def torch_rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    if torch.abs(axis).sum() < 1e-6 or torch.abs(theta) < 1e-6:
        return torch.eye(3)
    # axis = axis.tolist()
    axis = axis / torch.sqrt(torch.dot(axis, axis))
    a = torch.cos(theta / 2.0)
    b, c, d = -axis * torch.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return torch.tensor([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def torch_angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2':
        >>> angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        >>> angle_between((1, 0, 0), (1, 0, 0))
        0.0
        >>> angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793
    """
    if torch.abs(v1).sum() < 1e-6 or torch.abs(v2).sum() < 1e-6:
        return 0
    v1_u = torch_unit_vector(v1)
    v2_u = torch_unit_vector(v2)
    return torch.arccos(torch.clip(torch.dot(v1_u, v2_u), -1.0, 1.0))


def rotate_joints(data, joint_i, joint_j):
    axis = torch.cross(joint_i - joint_j, torch.tensor([0, 0, 1], dtype=torch.float))
    angle = torch_angle_between(joint_i - joint_j,
                                torch.tensor([0, 0, 1],
                                             dtype=torch.float))
    rotate_mat = torch_rotation_matrix(axis, angle)

    for i_f, frame in enumerate(data):
        for i_j, joint in enumerate(frame):
            try:
                data[i_f, i_j] = torch.matmul(rotate_mat.float(), joint)
            except RuntimeError:
                print("double")


def pre_normalization(data, z_axis=None, x_axis=None):
    # `index` of frames that have non-zero nodes
    if x_axis is None:
        x_axis = [8, 4]
    if z_axis is None:
        z_axis = [0, 1]
    debug_data = data.clone()
    index = (data.sum(-1).sum(-1) != 0)

    if data.sum() == 0:
        print('empty video without skeleton information')

    data = data[index]
    v1_frame = index[:(len(index) // 2)].sum().item()
    v2_frame = index[(len(index) // 2):].sum().item()

    # print('sub the center joint #1 (spine joint in ntu and neck joint in kinetics)')
    # Use the first person's body center (`1:2` along the nodes dimension)
    main_body_center = data[:v1_frame, 1:2, :].clone()
    # For all `person`, compute the `mask` which is the non-zero channel dimension
    mask = rearrange((data.sum(-1) != 0), 'f n -> f n 1')

    data[:v1_frame] = (data[:v1_frame] - main_body_center) * mask[:v1_frame]
    if v1_frame > v2_frame:
        data[v1_frame:] = (data[v1_frame:] - main_body_center[:v2_frame]) * mask[v1_frame:]
    elif v1_frame < v2_frame:
        reps = int(np.ceil(v2_frame / v1_frame))
        pad = torch.cat([main_body_center for _ in range(reps)])[:v2_frame]
        data[v1_frame:] = (data[v1_frame:] - pad) * mask[v1_frame:]
    else:
        data[v1_frame:] = (data[v1_frame:] - main_body_center) * mask[v1_frame:]

    # print('parallel the bone between hip(jpt 0) and spine(jpt 1) of the first person to the z axis')
    joint_bottom = data[0, z_axis[0]]
    joint_top = data[0, z_axis[1]]
    rotate_joints(data, joint_top, joint_bottom)

    # print('parallel the bone between right shoulder(jpt 8) and
    # left shoulder(jpt 4) of the first person to the x axis')
    joint_r_shoulder = data[0, x_axis[0]]
    joint_l_shoulder = data[0, x_axis[1]]
    rotate_joints(data, joint_r_shoulder, joint_l_shoulder)

    return data


def read_skeleton_filter(file):
    with open(file, 'r') as f:
        skeleton_sequence = {'numFrame': int(f.readline()), 'frameInfo': []}
        # num_body = 0
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {'numBody': int(f.readline()), 'bodyInfo': []}

            for m in range(frame_info['numBody']):
                body_info = {}
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


def get_nonzero_std(s):
    # `s` has shape (T, V, C)
    # Select valid frames where sum of all nodes is nonzero
    s = s[s.sum((1, 2)) != 0]
    if len(s) != 0:
        # Compute sum of standard deviation for all 3 channels as `energy`
        s = s[..., 0].std() + s[..., 1].std() + s[..., 2].std()
    else:
        s = 0
    return s


def num_processes():
    return os.cpu_count() - 4


def skeleton_parts(num_joints=25, dataset='ntu', cat=True):
    if 'ntu' in dataset:
        sk_adj = torch.tensor(
            # [[0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
            #   9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 16, 16, 17, 17, 18,
            #   18, 19, 20, 20, 20, 20, 21, 22, 22, 23, 24, 24],
            #  [1, 16, 12, 20, 0, 3, 20, 2, 20, 5, 4, 6, 7, 5, 22, 6, 20, 9,
            #   8, 10, 9, 11, 24, 10, 0, 13, 12, 14, 15, 13, 14, 0, 17, 16, 18, 17,
            #   19, 18, 8, 1, 4, 2, 22, 7, 21, 24, 11, 23]]
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24],
             [1, 20, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18, 22, 7, 24, 11]])
    elif 'kinetics' in dataset:
        sk_adj = torch.tensor(
            # [[4, 3, 7, 6, 13, 12, 10, 9, 11, 8, 5, 2, 0, 15, 14, 17,
            #   16, 3, 2, 6, 5, 12, 11, 9, 8, 5, 2, 1, 1, 1, 0, 0, 15, 14],
            #  [3, 2, 6, 5, 12, 11, 9, 8, 5, 2, 1, 1, 1, 0, 0, 15, 14,
            #   4, 3, 7, 6, 13, 12, 10, 9, 11, 8, 5, 2, 0, 15, 14, 17, 16]]
            [[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
             [1, 1, 2, 3, 1, 5, 6, 2, 8, 9, 5, 11, 12, 0, 0, 14, 15]]
        )
    else:
        return None

    if not cat:
        return sk_adj

    sk_adj_undirected = torch.cat((sk_adj, torch.stack([sk_adj[1], sk_adj[0]])), dim=1)

    _, idx = sk_adj_undirected[0].sort()
    sk_adj_undirected = sk_adj_undirected[:, idx]

    cat_adj = torch.cat([sk_adj_undirected,
                         power_adj(sk_adj_undirected, max(num_joints, max(sk_adj_undirected[1]) + 1), 2),
                         power_adj(sk_adj_undirected, max(num_joints, max(sk_adj_undirected[1]) + 1), 3)], dim=1)

    _, idx = cat_adj[0].sort()
    cat_adj = cat_adj[:, idx]

    return cat_adj


def power_adj(adj, dim, p):
    val = torch.ones(adj.shape[1])
    ic, vc = spspmm(adj, val, adj, val, dim, dim, dim)
    if p > 2:
        for i in range(p - 2):
            ic, vc = spspmm(ic, vc, adj, val, dim, dim, dim)
    return ic


def resolve_filename(name):
    action_class = int(name[name.find('A') + 1:name.find('A') + 4])
    subject_id = int(name[name.find('P') + 1:name.find('P') + 4])
    camera_id = int(name[name.find('C') + 1:name.find('C') + 4])
    setup_id = int(name[name.find('S') + 1:name.find('S') + 4])
    return action_class, subject_id, camera_id, setup_id


def data_padding(sparse_tensor, pad_length):
    f, n, c = sparse_tensor.shape
    return torch.cat([torch.zeros(pad_length, n, c)] + [sparse_tensor] + [torch.zeros(pad_length, n, c)], dim=0)


class SkeletonDataset(Dataset, ABC):
    def __init__(self,
                 root,
                 name,
                 use_motion_vector=True,
                 transform=None,
                 pre_transform=None,
                 benchmark='xsub',
                 sample='train'):
        self.name = name  # ntu ntu120 kinetics
        self.benchmark = benchmark
        self.sample = sample

        self.num_joints = 25 if 'ntu' in self.name else 18
        #self.skeleton_ = skeleton_parts(num_joints=self.num_joints,
        #                                dataset=self.name)
        self.training_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
                                  38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
                                  80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103]
        # For Cross-View benchmark "xview"
        self.training_setup = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
        self.paris = {
            'xview': (
                (1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                (22, 23), (21, 21), (23, 8), (24, 25), (25, 12)
            ),  # (21, 21)?
            'xsub': (
                (1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                (22, 23), (21, 21), (23, 8), (24, 25), (25, 12)
            ),

            'kinetics': (
                (0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1),
                (6, 5), (7, 6), (8, 2), (9, 8), (10, 9), (11, 5),
                (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15)
            )
        }
        self.max_body_true = 2

        print('processed the adjacency matrices of skeleton')
        self.use_motion_vector = use_motion_vector
        self.missing_skeleton_path = osp.join(os.getcwd(),
                                              'samples_with_missing_skeletons.txt')
        super(SkeletonDataset, self).__init__(root, transform, pre_transform)
        if 'ntu' in self.name:
            path = osp.join(self.processed_dir, self.processed_file_names)
            self.data = torch.load(path)

    @property
    def processed_file_names(self):
        if 'kinetics' in self.name:
            return [f for f in os.listdir(self.processed_dir)]
        else:
            return '{}_{}_{}.pt'.format(self.benchmark, self.sample, self.name)

    @property
    def raw_file_names(self):
        fp = lambda x: osp.join(self.root, 'raw', x)
        return [fp(f) for f in os.listdir(self.raw_dir)]  # if osp.isfile(fp(f))]

    @property
    def download(self):
        # Download to `self.raw_dir`.
        pass

    def read_xyz(self, file, sample, max_body=4):  # 取了前两个body
        filename = osp.split(file)[-1]
        action_class = int(filename[filename.find('A') + 1: filename.find('A') + 4])
        seq_info = read_skeleton_filter(file)
        # Create data tensor of shape: (# persons (M), # frames (T), # nodes (V), # channels (C))
        data = np.zeros((max_body, seq_info['numFrame'], self.num_joints, 3), dtype=np.float32)
        for n, f in enumerate(seq_info['frameInfo']):
            # print("frame: ", n)
            for m, b in enumerate(f['bodyInfo']):
                # print("person: ", m)
                for j, v in enumerate(b['jointInfo']):
                    if m < max_body and j < self.num_joints:
                        data[m, n, j, :] = [v['x'], v['y'], v['z']]

        # select 2 max energy body
        energy = np.array([get_nonzero_std(x) for x in data])
        index = energy.argsort()[::-1][0:self.max_body_true]
        data = data[index]

        torch_data = torch.from_numpy(data)
        del data
        torch_data = rearrange(torch_data, 'm f n c -> (m f) n c')  # <- always even so you can get person idx

        torch_data = pre_normalization(torch_data)
        #torch_data += torch.normal(mean=0, std=0.01, size=torch_data.size())
        pre_data = gen_bone_data(torch_data, self.paris, self.benchmark)
        sparse_data = Data(x=pre_data, y=action_class - 1)

        return sparse_data

        '''if sample == 'train':
            gaussian_noise = torch.normal(mean=0, std=0.01, size=torch_data.size())
            noisy_data = torch_data + gaussian_noise
            noisy_data = gen_bone_data(noisy_data, self.paris, self.benchmark)
            noisy_sparse_data = Data(x=noisy_data, y=action_class - 1)
            return sparse_data, noisy_sparse_data
        else:
            return (sparse_data,)'''


    def add_noise(self, data, scale):
        t = data.x[:, :, :3]
        y = data.y
        t += torch.normal(mean=0, std=scale, size=t.size())
        t = gen_bone_data(t, self.paris, self.benchmark)
        '''print("After gen bone data")

        if torch.isnan(t).sum().item() != 0:
            print("Nan in tensor")
        if torch.isinf(t).sum().item() != 0:
            print("Inf in tensor")'''

        t = Data(x=t, y=y)
        return t


    def process(self):
        if self.missing_skeleton_path is not None:
            with open(self.missing_skeleton_path, 'r') as f:
                ignored_samples = [line.strip() + '.skeleton' for line in f.readlines()]
        else:
            ignored_samples = []

        sample_name = []
        sample_label = []

        sparse_data_list = []

        for file in self.raw_file_names:
            filename = osp.split(file)[-1]
            if filename in ignored_samples:
                print("Found a missing skeleton!")
                continue

            action_class, subject_id, camera_id, setup_id = resolve_filename(filename)

            if self.benchmark == 'xview':
                is_training = (setup_id in self.training_setup)
            elif self.benchmark == 'xsub':
                is_training = (subject_id in self.training_subjects)
            else:
                raise ValueError('Invalid benchmark provided: {}'.format(self.benchmark))

            if self.sample == 'train':
                is_sample = is_training
            elif self.sample == 'val':
                is_sample = not is_training
            else:
                raise ValueError('Invalid data part provided: {}'.format(self.sample))

            if is_sample:
                sample_name.append(file)
                sample_label.append(action_class - 1)

        pool = Pool(processes=num_processes())
        partial_func = partial(self.read_xyz,
                               sample=self.sample, max_body=4)

        progress_bar = tqdm(pool.imap(func=partial_func, iterable=sample_name),
                            total=len(sample_name))

        for data in progress_bar:
            sparse_data_list.append(data) 

        noisy_sparse_data_list = []
        if self.sample == 'train':
            #pool = Pool(processes=num_processes())
            #partial_func = partial(self.add_noise,
            #                    scale=0.01)

            #progress_bar = tqdm(pool.imap(func=partial_func, iterable=sparse_data_list),
            #                    total=len(sparse_data_list))

            #for data in progress_bar:
            #    noisy_sparse_data_list.append(data) 
            for data in sparse_data_list:
                noisy_sparse_data_list.append(self.add_noise(data, scale=0.01))

        torch.save(sparse_data_list + noisy_sparse_data_list, osp.join(self.processed_dir,
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
                                           self.processed_file_names[idx]))
            return [torch.load(osp.join(self.processed_dir,
                                        self.processed_file_names[i])) for i in idx]
        return self.data[idx]


def test():
    from argparse import ArgumentParser
    from torch_geometric.data import DataLoader
    parser = ArgumentParser()
    parser.add_argument('--root', dest='root',
                        default=osp.join(os.getcwd(), 'dataset', 'ntu_60'),
                        type=str, help='Dataset')
    parser.add_argument('--dataset', dest='dataset', default='ntu_60',
                        type=str, help='Dataset')
    args = parser.parse_args()
    ds = SkeletonDataset(root=os.getcwd(),
                         name='ntu_60_test',
                         benchmark='xsub',
                         sample='val')
    loader = DataLoader(ds[0: 8], batch_size=4)
    for b in loader:
        print(b.x.shape)


if __name__ == "__main__":
    test()
