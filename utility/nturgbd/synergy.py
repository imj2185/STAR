from ntu_pyg import NTUDataset
from ntu_pyg import *


def to_synergy_matrix(batch, pairs):
    synergy_person_1 = torch.zeros(batch.size(0) // 25, len(pairs), 300)
    synergy_person_2 = torch.zeros(batch.size(0) // 25, len(pairs), 300)
    index = 0
    for i in range(0, batch.size(0), 25):
        graph = batch[i:i + 25]
        synergy_matrix_1 = torch.zeros(0, 300)
        synergy_matrix_2 = torch.zeros(0, 300)
        for joint_1, joint_2 in pairs:
            # pair : (joint 1, joint 2)
            pair_synergy = torch.mul(graph[joint_1][7], graph[joint_2][7]) + torch.mul(graph[joint_1][8],
                                                                                       graph[joint_2][8]) + torch.mul(
                graph[joint_1][9], graph[joint_2][9])
            pair_synergy_person_1 = torch.unsqueeze(pair_synergy.permute(1, 0)[0], 0)
            pair_synergy_person_2 = torch.unsqueeze(pair_synergy.permute(1, 0)[1], 0)
            synergy_matrix_1 = torch.cat((synergy_matrix_1, pair_synergy_person_1))
            synergy_matrix_2 = torch.cat((synergy_matrix_2, pair_synergy_person_2))
        synergy_person_1[index] = synergy_matrix_1
        synergy_person_2[index] = synergy_matrix_2
        index += 1
    return synergy_person_1, synergy_person_2


if __name__ == '__main__':

    n_batch_size = 3
    ntu_dataset = NTUDataset(
        "/home/cchenli/Documents/Apb-gcn/utils/nturgbd",
        batch_size=n_batch_size,
        benchmark='cv',
        part='val',
        ignored_sample_path=None,
        plan="synergy_matrix")

    ntu_dataloader = DataLoader(ntu_dataset, batch_size=n_batch_size, shuffle=True)

    count = 0
    i = 0
    batch = None
    pairs = [(1, 2), (3, 4)]
    for b in (ntu_dataloader):
        batch = b
        matrix1, matrix2 = to_synergy_matrix(batch, pairs)
    print(count)
