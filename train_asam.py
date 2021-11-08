import os
import time
from random import shuffle

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from args import make_args
from data.dataset3 import SkeletonDataset, skeleton_parts
from models.encoding import SeqPosEncoding, KStepRandomWalkEncoding
from models.net3streams import DualGraphEncoder
from optimizer import LabelSmoothingCrossEntropy, ASAM


def run(rank, num_gpu):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=num_gpu)
    args = make_args()

    train_ds = SkeletonDataset(args.dataset_root, name='ntu_60',
                               use_motion_vector=False, sample='train')

    test_ds = SkeletonDataset(args.dataset_root, name='ntu_60',
                              use_motion_vector=False, sample='val')

    shuffled_list = [i for i in range(len(train_ds))]
    shuffle(shuffled_list)
    train_ds = train_ds[shuffled_list]

    # train_sampler = DistributedSampler(train_ds, num_replicas=num_gpu,
    #                                    rank=rank)
    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size)
    test_loader = None

    print('Calculating temporal/sequential positional encoding ......'.format(n=3))
    temporal_pos_enc = SeqPosEncoding(model_dim=args.hid_channels)
    print('Calculating A^{n} for spatial positional encoding ......'.format(n=3))
    spatial_pos_enc = KStepRandomWalkEncoding().eval(train_ds.sk_adj)[1]
    print('Initializing model ......')
    model = DualGraphEncoder(in_channels=args.in_channels,
                             hidden_channels=args.hid_channels,
                             out_channels=args.out_channels,
                             mlp_head_hidden=args.mlp_head_hidden,
                             num_layers=args.num_enc_layers,
                             num_heads=args.heads,
                             sequential=False,
                             use_cross_view=(args.num_of_streams == 3),
                             temporal_pos_enc=temporal_pos_enc,
                             spatial_pos_enc=spatial_pos_enc,
                             num_conv_layers=args.num_conv_layers,
                             drop_rate=args.drop_rate).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = SgdAgc(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    loss_compute = LabelSmoothingCrossEntropy()
    minimizer = ASAM(optimizer, model)

    if rank == 0:
        test_loader = DataLoader(test_ds,
                                 batch_size=args.batch_size)

    last_epoch = 0
    adj = skeleton_parts()[0].to(rank)

    for epoch in range(last_epoch, args.epoch_num + last_epoch):
        model.train()
        running_loss = 0.
        accuracy = 0.
        cnt = 0.
        start = time.time()
        total_batch = len(train_ds) // args.batch_size + 1
        batch_loss = 0.
        # iterate batches
        for i, batch in tqdm(enumerate(train_loader),
                             total=total_batch,
                             desc="Train Epoch {}".format(epoch + 1)):
            batch = batch.to(rank)
            sample, label, bi = batch.x, batch.y, batch.batch

            # Ascent Step
            predictions = model(sample, adj=adj, bi=bi)
            batch_loss = loss_compute(predictions, label.long())
            batch_loss.mean().backward()
            minimizer.ascent_step()

            # Descent Step
            loss_compute(model(sample, adj=adj, bi=bi), label.long()).mean().backward()
            minimizer.descent_step()

            with torch.no_grad():
                running_loss += batch_loss.sum().item()
                accuracy += (torch.argmax(predictions, 1) == label).sum().item()
            cnt += len(label)
        running_loss /= cnt
        accuracy *= 100. / cnt
        elapsed = time.time() - start
        # accuracy = correct / total_samples * 100.
        print('------ loss: %.3f; accuracy: %.3f%%; average time: %.4f\n' %
              (running_loss, accuracy, elapsed / len(train_ds)))

        dist.barrier()

        if rank == 0:  # We evaluate on a single GPU for now.
            model.eval()
            running_loss = 0.
            accuracy = 0.
            correct = 0
            total_samples = 0
            start = time.time()
            total_batch = len(test_ds) // args.batch_size + 1

            for i, batch in tqdm(enumerate(test_loader),
                                 total=total_batch,
                                 desc="Test: "):
                batch = batch.to(rank)
                sample, label, bi = batch.x, batch.y, batch.batch
                with torch.no_grad():
                    out = model.module(sample, adj=adj, bi=bi)
                running_loss += batch_loss.item()
                pred = torch.max(out, 1)[1]
                total_samples += label.size(0)
                corr = (pred == label)
                correct += corr.double().sum().item()
            elapsed = time.time() - start
            accuracy = correct / total_samples * 100.
            print('------ loss: %.3f; accuracy: %.3f%%; average time: %.4f\n' %
                  (running_loss / total_batch, accuracy, elapsed / len(test_ds)))

        dist.barrier()

    dist.destroy_process_group()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
