import os
import os.path as osp
import time
from random import shuffle

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch_geometric.data import DataLoader
from tqdm import tqdm, trange

from args import make_args
from data.dataset3 import SkeletonDataset, skeleton_parts
from models.net2s import DualGraphEncoder
from optimizer import SGD_AGC, CosineAnnealingWarmupRestarts, ZeroOneClipper, MaxOneClipper, LabelSmoothingCrossEntropy
from utility.helper import make_checkpoint, load_checkpoint
from random import shuffle

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
                writer.add_histogram('weights/' + n, p, step)
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
    # plt.close()
    # plt.show()


def plot_distribution(gt_list, cr_list, wr_list, path):
    labels = [i + 1 for i in range(60)]
    x = np.arange(len(labels))  # the label locations

    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 0.2, gt_list, width, label='gt_list')
    rects2 = ax.bar(x, cr_list, width, label='cr_list')
    rects3 = ax.bar(x + 0.2, wr_list, width, label='wr_list')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('number of samples')
    ax.set_title('Data distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation="vertical", fontsize=5)
    ax.legend()

    plt.savefig(path, dpi=300)
    plt.close()


def run_epoch(data_loader,
              model,
              optimizer,
              loss_compute,
              dataset,
              device,
              gt_list,
              cr_list,
              wr_list,
              is_train=True,
              is_test=False,
              desc=None,
              args=None,
              writer=None,
              epoch_num=0,
              adj=None,
              l1_penalty=False):
    """Standard Training and Logging Function

        :param adj:
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
        sample, label, bi = batch.x, batch.y, batch.batch

        with torch.set_grad_enabled(is_train) and torch.autograd.set_detect_anomaly(True):
            out = model(sample, adj=adj, bi=bi)
            loss = loss_compute(out, label.long())
            loss_ = loss
            if is_train:
                if l1_penalty:
                    l1p = torch.nn.L1Loss(size_average=False)
                    l1_loss = 0
                    for param in model.parameters():
                        l1_loss += l1p(param, target=torch.zeros_like(param))
                    factor = 0.9
                    loss += l1_loss * factor                    #l1_regularization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i % 400 == 0:
                    step = (i + 1) + total_batch * epoch_num
                    path = osp.join(os.getcwd(), args.gradflow_dir)
                    if not osp.exists(path):
                        os.mkdir(path)
                    plot_grad_flow(model.named_parameters(), osp.join(path, '%3d_%d.png' % (epoch_num, i)), writer,
                                   step)

            # statistics
            running_loss += loss_.item()
            pred = torch.max(out, 1)[1]
            total_samples += label.size(0)
            corr = (pred == label)
            correct += corr.double().sum().item()
            if is_test:
                for j in range(len(label)):
                    gt_list[label[j].item()] += 1
                    cr_list[label[j].item()] += corr[j].item()
                    wr_list[label[j].item()] += not (corr[j].item())

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
                               benchmark=args.benchmark, sample='train')
    test_ds = SkeletonDataset(args.dataset_root, name='ntu_60',
                              use_motion_vector=False,
                              benchmark=args.benchmark, sample='val')

    adj = skeleton_parts(dataset=args.dataset_name)[0].to(device)

    train_loader = DataLoader(train_ds.data,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_ds,
                             batch_size=args.batch_size,
                             shuffle=True)

    last_epoch = 0
    model = DualGraphEncoder(in_channels=args.in_channels,
                             hidden_channels=args.hid_channels,
                             out_channels=args.out_channels,
                             mlp_head_hidden=args.mlp_head_hidden,
                             num_layers=args.num_enc_layers,
                             num_heads=args.heads,
                             num_features=args.num_features,
                             sequential=False,
                             num_conv_layers=args.num_conv_layers,
                             drop_rate=args.drop_rate)

    if torch.cuda.device_count() > 1 and args.data_parallel:
        num_gpu = torch.cuda.device_count()
        print("Let's use ", num_gpu, " GPUs!")
        adj = torch.stack([adj] * num_gpu).to(device)
        model = nn.DataParallel(model)

    model = model.to(device)
    print(sum(p.numel() for p in model.parameters()))
    # noam_opt = get_std_opt(model, args)

    optimizer = SGD_AGC(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # decay_rate = 0.96
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)
    lr_scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=8, cycle_mult=1.0, max_lr=0.1,
                                                 min_lr=1e-4, warmup_steps=3, gamma=0.7)

    # weight_clipper = ZeroOneClipper()
    weight_clipper = MaxOneClipper()

    if args.load_model:
        last_epoch = args.load_epoch
        last_epoch, loss = load_checkpoint(osp.join(args.save_root,
                                                    args.save_name + '_' + str(last_epoch) + '.pickle'),
                                           model, optimizer)
        print("Load Model: ", last_epoch)

    loss_compute = LabelSmoothingCrossEntropy().to(device)
    l1_penalty = False

    for epoch in trange(last_epoch, args.epoch_num + last_epoch):
        gt_list = [0 for _ in range(60)]
        cr_list = [0 for _ in range(60)]
        wr_list = [0 for _ in range(60)]

        model.train(True)
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        writer.add_scalar('params/lr', lr, epoch)

        train_loss, train_accuracy = run_epoch(train_loader, model, optimizer,
                                               loss_compute, train_ds, device, gt_list=gt_list, cr_list=cr_list,
                                               wr_list=wr_list, is_train=True, is_test=False,
                                               desc="Train Epoch {}".format(epoch + 1), args=args, writer=writer,
                                               epoch_num=epoch,
                                               adj=adj,
                                               l1_penalty=l1_penalty)
        print('Epoch: {} Evaluating...'.format(epoch + 1))

        # TODO Save model
        writer.add_scalar('train/train_loss', train_loss, epoch + 1)
        writer.add_scalar('train/train_overall_acc', train_accuracy, epoch + 1)

        if epoch % args.epoch_save == 0:
            make_checkpoint(args.save_root, args.save_name, epoch, model, optimizer, train_loss)

        # Validation
        model.eval()
        test_loss, test_accuracy = run_epoch(test_loader, model, optimizer,
                                             loss_compute, test_ds, device, gt_list=gt_list, cr_list=cr_list,
                                             wr_list=wr_list, is_train=False, is_test=True,
                                             desc="Final test: ", args=args, writer=writer, epoch_num=epoch, adj=adj, l1_penalty=l1_penalty)

        writer.add_scalar('test/test_loss', test_loss, epoch + 1)
        writer.add_scalar('test/test_overall_acc', test_accuracy, epoch + 1)
        # plot_distribution(gt_list=gt_list, cr_list=cr_list, wr_list=wr_list,
        #                   path=osp.join(os.getcwd(), 'distribution', str(epoch + 1) + '.png'))

        # if epoch > 15:

        lr_scheduler.step()
        if train_accuracy - 5 > test_accuracy:
            l1_penalty=True
        #     model.apply(weight_clipper)
        #     model.mlp_head[1].apply(weight_clipper)
        #     model.mlp_head[3].apply(weight_clipper)
        #     for name, module in model.named_modules():
        #         if ('ffn' in name or 'mlp_head'in name) and isinstance(module, nn.Linear):
        #             module.apply(weight_clipper)

    writer.export_scalars_to_json(osp.join(args.log_dir, "all_scalars.json"))
    writer.close()


if __name__ == "__main__":
    main()
