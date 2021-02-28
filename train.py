import os
import os.path as osp
import time
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch_geometric.data import DataLoader
from tqdm import tqdm, trange

from args import make_args
from data.dataset3 import SkeletonDataset
from models.net import DualGraphEncoder
from optimizer import SGD_AGC, CosineAnnealingWarmupRestarts
from utility.helper import make_checkpoint, load_checkpoint


def plot_grad_flow(named_parameters, path, writer, step):
    ave_grads = []
    layers = []
    empty_grads = []
    # total_norm = 0
    for n, p in named_parameters:
        if p.requires_grad and not (("bias" in n) or ("norm" in n) or ("bn" in n) or ("gain" in n)):
            if p.grad is not None:
                # writer.add_scalar('gradients/' + n, p.grad.norm(2).item(), step)
                # writer.add_histogram('gradients/' + n, p.grad, step)
                # total_norm += p.grad.data.norm(2).item()
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu().item())
            else:
                empty_grads.append({n: p.mean().cpu().item()})
    # total_norm = total_norm ** (1. / 2)
    # print("Norm : ", total_norm)
    #plt.tight_layout()
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1.5, color="k")
    plt.xticks(np.arange(0, len(ave_grads), 1), layers, rotation="vertical", fontsize=4)
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig(path, dpi=300)
    # plt.show()


def run_epoch(data_loader,
              model,
              optimizer,
              loss_compute,
              dataset,
              device,
              is_train=True,
              desc=None,
              args=None,
              writer=None,
              epoch_num=0):
    """Standard Training and Logging Function

        :param data_loader:
        :param model:
        :param optimizer:
        :param loss_compute:
        :param dataset:
        :param device:
        :param is_train:
        :param desc:
        :param args:
        :param writer:
        :param epoch_num:

    """
    # torch.autograd.set_detect_anomaly(True)
    running_loss = 0.
    accuracy = 0.
    correct = 0
    total_samples = 0
    start = time.time()
    total_batch = len(dataset) // args.batch_size + 1
    for i, batch in tqdm(enumerate(data_loader),
                         total=total_batch,
                         desc=desc):
        batch = batch.to(device)
        sample, label, bi = batch.x, batch.y, batch.batch.to(device)

        with torch.set_grad_enabled(is_train):
            out = model(sample, adj=dataset.skeleton_.to(device), bi=bi)
            loss = loss_compute(out, label.long())
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 9.0)
                optimizer.step()
                if i % 400 == 0:
                    step = (i + 1) + total_batch * epoch_num
                    path = osp.join(os.getcwd(), 'gradflow')
                    if not osp.exists(path):
                        os.mkdir(path)
                    plot_grad_flow(model.named_parameters(), osp.join(path, 'grad%3d:%d.png' % (epoch_num, i)), writer,
                                   step)

                # plot_grad_flow(model.named_parameters(), writer, (i + 1) + total_batch * epoch_num)
                # for name, param in model.named_parameters():
                # if param.requires_grad and param.grad is not None:
                # writer.add_scalar('gradients/' + name, param.grad.norm(2).item(), (i + 1) + total_batch * epoch_num)

            # statistics
            running_loss += loss.item()
            pred = torch.max(out, 1)[1]
            total_samples += label.size(0)
            correct += (pred == label).double().sum().item()

    elapsed = time.time() - start
    accuracy = correct / total_samples * 100.
    print('\n------ loss: %.3f; accuracy: %.3f; average time: %.4f' %
          (running_loss / total_batch, accuracy, elapsed / len(dataset)))

    return running_loss / total_batch, accuracy


def main():
    # torch.cuda.empty_cache()
    args = make_args()
    writer = SummaryWriter(args.log_dir)
    device = torch.device('cuda:0') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # download and save the dataset
    train_ds = SkeletonDataset(args.dataset_root, name='ntu_60',
                               use_motion_vector=False,
                               benchmark='xsub', sample='train')
    test_ds = SkeletonDataset(args.dataset_root, name='ntu_60',
                              use_motion_vector=False,
                              benchmark='xsub', sample='val')

    last_train = int(len(train_ds) * 0.8)

    # randomly split into around 80% train, 10% val and 10% train
    # train_loader = DataLoader(train_ds.data,
    #                          batch_size=args.batch_size,
    #                          shuffle=True)
    test_loader = DataLoader(test_ds,
                             batch_size=args.batch_size,
                             shuffle=True)

    # criterion = LabelSmoothing(V, padding_idx=dataset.pad_id, smoothing=0.1)
    # make_model black box
    last_epoch = 0
    model = DualGraphEncoder(in_channels=args.in_channels,
                             hidden_channels=args.hid_channels,
                             out_channels=args.out_channels,
                             num_layers=args.num_enc_layers,
                             num_heads=args.heads,
                             sequential=False,
                             num_conv_layers=args.num_conv_layers,
                             drop_rate=args.drop_rate)
    model = model.to(device)
    # noam_opt = get_std_opt(model, args)
    optimizer = SGD_AGC(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    decay_rate = 0.97
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)
    #lr_scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=12, cycle_mult=1.0, max_lr=0.1,
    #                                             min_lr=1e-4, warmup_steps=3, gamma=0.4)
    if args.load_model:
        last_epoch = args.load_epoch
        last_epoch, loss = load_checkpoint(osp.join(args.save_root,
                                                    args.save_name + '_' + str(last_epoch) + '.pickle'),
                                           model, optimizer)
        print("Load Model: ", last_epoch)

    loss_compute = nn.CrossEntropyLoss().to(device)

    for epoch in trange(last_epoch, args.epoch_num + last_epoch):
        shuffled_list = [i for i in range(len(train_ds))]
        shuffle(shuffled_list)
        train_ds = train_ds[shuffled_list]

        train_ds_ = train_ds[:last_train]
        valid_ds_ = train_ds[last_train:]

        train_loader = DataLoader(train_ds_,
                                  batch_size=args.batch_size,
                                  shuffle=True)
        valid_loader = DataLoader(valid_ds_,
                                  batch_size=args.batch_size,
                                  shuffle=True)
        # print('Epoch: {} Training...'.format(epoch))
        model.train(True)
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        writer.add_scalar('params/lr', lr, epoch)

        loss, accuracy = run_epoch(train_loader, model, optimizer,
                                   loss_compute, train_ds_, device, is_train=True,
                                   desc="Train Epoch {}".format(epoch + 1), args=args, writer=writer, epoch_num=epoch)
        print('Epoch: {} Evaluating...'.format(epoch + 1))

        # TODO Save model
        writer.add_scalar('train/train_loss', loss, epoch + 1)
        writer.add_scalar('train/train_overall_acc', accuracy, epoch + 1)

        if epoch % args.epoch_save == 0:
            make_checkpoint(args.save_root, args.save_name, epoch, model, optimizer, loss)

        # Validation
        model.eval()
        loss, accuracy = run_epoch(valid_loader, model, optimizer,
                                   loss_compute, valid_ds_, device, is_train=False,
                                   desc="Valid Epoch {}".format(epoch + 1), args=args, writer=writer, epoch_num=epoch)

        writer.add_scalar('val/val_loss', loss, epoch + 1)
        writer.add_scalar('val/val_overall_acc', accuracy, epoch + 1)
        # if epoch > 15:
        lr_scheduler.step()

        if epoch % 10 == 0:
            model.eval()
            loss, accuracy = run_epoch(test_loader, model, optimizer,
                                    loss_compute, test_ds, device, is_train=False,
                                    desc="Final test: ", args=args, writer=writer, epoch_num=epoch)

            writer.add_scalar('test/test_loss', loss, , epoch + 1)
            writer.add_scalar('test/test_overall_acc', accuracy, epoch + 1)

    writer.export_scalars_to_json(osp.join(args.log_dir, "all_scalars.json"))
    writer.close()


if __name__ == "__main__":
    main()
