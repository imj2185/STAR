import matplotlib.pyplot as plt
import torch

from args import make_args
from data.dataset3 import SkeletonDataset

parts = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
         (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
         (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
         (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
         (22, 23), (21, 21), (23, 8), (24, 25), (25, 12)]

kinetics = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,11),
            (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
            (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]

def skeleton_visual(skeletons):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in parts:
        start_joint = i[0] - 1
        end_joint = i[1] - 1
        VecStart_x = skeletons[start_joint][0]
        VecEnd_x = skeletons[end_joint][0]
        VecStart_y = skeletons[start_joint][1]
        VecEnd_y = skeletons[end_joint][1]
        VecStart_z = skeletons[start_joint][2]
        VecEnd_z = skeletons[end_joint][2]
        ax.plot(xs=[VecStart_x, VecEnd_x], ys=[VecStart_y, VecEnd_y], zs=[VecStart_z, VecEnd_z])
    # x,y,z = [1, 0, 0],[2, 0, 0],[3, 0, 0]
    # sc = ax.scatter(x, y, z, s=40)
    # ax.plot(x,y, color='r')

    plt.show()

def skeleton_visual_2d(skeletons, kinetics, idx):
    fig = plt.figure()
    x = skeletons[idx].x[5,:,0].numpy()
    y = skeletons[idx].x[5,:,1].numpy()
    #for j in kinetics:
    #    plt.plot(x[j[0]], y[j[1]], x[j[0]], y[j[1]], marker = 'o')
        
    plt.scatter(x, y)
    for bone in kinetics:
        plt.plot([x[bone[0]], x[bone[1]]], [y[bone[0]], y[bone[1]]], 'r')
    plt.show()


# def skeleton_visual_2d(skeletons):
#     fig = plt.figure()
#     ax = plt.axes(projection='3d')
#     for i in kinetics:
#         start_joint = i[0] - 1
#         end_joint = i[1] - 1
#         VecStart_x = skeletons[start_joint][0]
#         VecEnd_x = skeletons[end_joint][0]
#         VecStart_y = skeletons[start_joint][1]
#         VecEnd_y = skeletons[end_joint][1]
#         ax.plot(xs=[VecStart_x, VecEnd_x], ys=[VecStart_y, VecEnd_y])
#     #plt.savefig('kinetics_sk.png')
#     plt.show()

def test():
    args = make_args()
    # skeletons, labels = torch.load("dataset/ntu_60/processed/xsub_val_ntu_60.pt")
    valid_ds = SkeletonDataset('/home/dusko/Documents/projects/APBGCN/raw_kinetics/kinetics_train', name='kinetics',
                               use_motion_vector=False,
                               benchmark='xsub', sample='train')
    # print(skeletons[0].x[0)
    skeleton_visual_2d(valid_ds, kinetics, 10)
    #skeleton_visual_2d(valid_ds[7].x[5])
    


if __name__ == "__main__":
    test()
