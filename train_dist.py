import os
import os.path as osp
import time
from random import shuffle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from tqdm import tqdm

from args import make_args
from data.dataset3 import SkeletonDataset, skeleton_parts
from models.net2s import DualGraphEncoder
from optimizer import LabelSmoothingCrossEntropy, SGD_AGC, CosineAnnealingWarmupRestarts, NoamOpt

matplotlib.use('Agg')


def plot_grad_flow(named_parameters, path, writer, step):
    ave_grads = []
    layers = []
    empty_grads = []
    # total_norm = 0
    for n, p in named_parameters:
        if p.requires_grad and not (("bias" in n) or ("dn" in n) or ("ln" in n) or ("gain" in n)):
            if p.grad is not None:
                # writer.add_scalar('gradients/' + n, p.grad.norm(2).item(), step)
                # writer.add_histogram('weights/' + n, p, step)
                # total_norm += p.grad.data.norm(2).item()
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu().item())
            else:
                empty_grads.append({n: p.mean().cpu().item()})
    # total_norm = total_norm ** (1. / 2)
    # print("Norm : ", total_norm)
    plt.tight_layout()
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1.5, color="k")
    plt.xticks(np.arange(0, len(ave_grads), 1), layers, rotation="vertical", fontsize=4)
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow" + str(step))
    plt.grid(True)
    plt.savefig(path, dpi=300)


def run(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    args = make_args()

    train_ds = SkeletonDataset(args.dataset_root, name='ntu_60',
                               use_motion_vector=False, sample='train')

    test_ds = SkeletonDataset(args.dataset_root, name='ntu_60',
                              use_motion_vector=False, sample='val')

    shuffled_list = [i for i in range(len(train_ds))]
    shuffle(shuffled_list)
    train_ds = train_ds[shuffled_list]

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                       rank=rank)
    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size,
                              shuffle=True)

    model = DualGraphEncoder(in_channels=args.in_channels,
                             hidden_channels=args.hid_channels,
                             out_channels=args.out_channels,
                             mlp_head_hidden=args.mlp_head_hidden,
                             num_layers=args.num_enc_layers,
                             num_heads=args.heads,
                             sequential=False,
                             num_conv_layers=args.num_conv_layers,
                             drop_rate=args.drop_rate).to(rank)

    model = DistributedDataParallel(model, device_ids=[rank])
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = NoamOpt(args.model_dim,  # TODO num_nodes is not fixed
                   args.opt_train_factor,
                   args.warmup_steps,
                   opt.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9,
                   weight_decay=args.weight_decay))
    #optimizer = SGD_AGC(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    #lr_scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=12, cycle_mult=1.0, max_lr=0.1,
    #                                             min_lr=1e-4, warmup_steps=3, gamma=0.4)
    loss_compute = LabelSmoothingCrossEntropy()

    if rank == 0:
        writer = SummaryWriter(args.log_dir)
        test_loader = DataLoader(test_ds,
                                 batch_size=args.batch_size)

    last_epoch = 0
    adj = skeleton_parts()[0].to(rank)

    for epoch in range(last_epoch, args.epoch_num + last_epoch):
        model.train()
        running_loss = 0.
        accuracy = 0.
        correct = 0
        total_samples = 0
        start = time.time()
        total_batch = len(train_ds) // args.batch_size + 1

        for i, batch in tqdm(enumerate(train_loader),
                             total=total_batch,
                             desc="Train Epoch {}".format(epoch + 1)):
            batch = batch.to(rank)
            sample, label, bi = batch.x, batch.y, batch.batch
            optimizer.zero_grad()
            out = model(sample, adj=adj, bi=bi)
            loss = loss_compute(out, label.long())
            loss.backward()
            optimizer.step()

            if rank == 0 and i % 400 == 0:
                step = (i + 1) + total_batch * epoch
                path = osp.join(os.getcwd(), args.gradflow_dir)
                if not osp.exists(path):
                    os.mkdir(path)
                plot_grad_flow(model.named_parameters(), osp.join(path, '%d_%d.png' % (epoch, i)), writer, step)

            running_loss += loss.item()
            pred = torch.max(out, 1)[1]
            total_samples += label.size(0)
            corr = (pred == label)
            correct += corr.double().sum().item()
        elapsed = time.time() - start
        accuracy = correct / total_samples * 100.
        print('\n------ loss: %.3f; accuracy: %.3f; average time: %.4f' %
              (running_loss / total_batch, accuracy, elapsed / len(train_ds)))
        if rank == 0:
            writer.add_scalar('train/train_loss', running_loss / total_batch, epoch + 1)
            writer.add_scalar('train/train_overall_acc', accuracy, epoch + 1)
            
        #lr_scheduler.step()
        dist.barrier()

        if rank == 0:  # We evaluate on a single GPU for now.
            model.eval()
            running_loss = 0.
            accuracy = 0.
            correct = 0
            total_samples = 0
            start = time.time()
            total_batch = len(test_ds) // args.batch_size + 1
            # adj = skeleton_parts()[0].to(rank)

            for i, batch in tqdm(enumerate(test_loader),
                                 total=total_batch,
                                 desc="Test: "):
                batch = batch.to(rank)
                sample, label, bi = batch.x, batch.y, batch.batch
                with torch.no_grad():
                    out = model.module(sample, adj=adj, bi=bi)
                running_loss += loss.item()
                pred = torch.max(out, 1)[1]
                total_samples += label.size(0)
                corr = (pred == label)
                correct += corr.double().sum().item()
            elapsed = time.time() - start
            accuracy = correct / total_samples * 100.
            print('\n------ loss: %.3f; accuracy: %.3f; average time: %.4f' %
                  (running_loss / total_batch, accuracy, elapsed / len(test_ds)))
            writer.add_scalar('test/test_loss', running_loss / total_batch, epoch + 1)
            writer.add_scalar('test/test_overall_acc', accuracy, epoch + 1)

        dist.barrier()

    dist.destroy_process_group()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
