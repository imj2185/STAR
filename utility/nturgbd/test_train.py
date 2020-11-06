# from __future__ import print_function
import torchvision.models as models
from pairs import *
import os
import glob
from ntu_pyg import *

import numpy as np
from synergy import to_synergy_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
from tensorboardX import SummaryWriter
from synergy_model import Net_one

mobilenet = Net_one()


class Timer(object):
    def __init__(self,
                 print_log=True,
                 work_dir=''):
        self.curr_time = time.time()
        self.log = print_log
        self.work_dir = work_dir

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.log:
            with open('{}/log.txt'.format(self.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.curr_time = time.time()
        return self.curr_time

    def split_time(self):
        split_time = time.time() - self.curr_time
        self.record_time()
        return split_time


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(epoch):
    if True:
        lr = 0.0001 * (
                0.1 ** np.sum(epoch >= np.array([20, 40, 60])))
        # for param_group in self.optimizer.param_groups:
        #    param_group['lr'] = lr
        return lr
    else:
        raise ValueError()


def top_k(score, label, top_k):
    rank = score.argsort()
    hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(label)]
    return sum(hit_top_k) * 1.0 / len(hit_top_k)


def train(train_loader, epoch, pairs):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    logdir = os.path.join("/home/lawbuntu/Documents/Workspace/Apb-gcn/utils/nturgbd", 'runs')
    mobilenet = Net_one()

    mobilenet.train()
    print('Training epoch: {}'.format(epoch + 1))
    lr = adjust_learning_rate(epoch)
    loss_value = []

    timetracker = Timer(print_log=True, work_dir="/home/lawbuntu/Documents/Workspace/Apb-gcn/utils/nturgbd")
    timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

    optimizer = optim.Adam(
        mobilenet.parameters(),
        lr=10 ** (-4),
        weight_decay=0.001)
    # summary_writer = SummaryWriter(logdir=logdir)
    log_interval = 100

    for batch_idx, data in enumerate(train_loader):
        label = data.y
        # get data
        data = Variable(
            data.x, requires_grad=False)
        label = Variable(
            label, requires_grad=False)
        timer['dataloader'] += timetracker.split_time()

        
        optimizer.zero_grad()
        # forward
        output = mobilenet(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, label.long())

        # backward
        loss.backward()
        optimizer.step()
        loss_value.append(loss.item())
        score_frag = output.data.cpu().numpy()
        label_frag = label.data.cpu().numpy()
        timer['model'] += timetracker.split_time()

        hit1 = top_k(score_frag, label_frag, 1)
        hit5 = top_k(score_frag, label_frag, 5)
        loss_val = loss.item()

        losses.update(loss_val, data[0].size(0))
        top1.update(hit1 * 100., data[0].size(0))
        top5.update(hit5 * 100., data[0].size(0))

        # statistics
        if batch_idx % log_interval == 0:
            print(
                '\tBatch({}/{}) done. Top1: {:.2f} ({:.2f})  Top5: {:.2f} ({:.2f}) '
                ' Loss: {:.4f} ({:.4f})  lr:{:.6f}'.format(
                    batch_idx, len(train_loader), top1.val, top1.avg,
                    top5.val, top5.avg, losses.val, losses.avg, lr))
            step = epoch * len(train_loader) + batch_idx
            # summary_writer.add_scalar('Train/AvgLoss', losses.avg, step)
            # summary_writer.add_scalar('Train/AvgTop1', top1.avg, step)
            # summary_writer.add_scalar('Train/AvgTop5', top5.avg, step)
            # summary_writer.add_scalar('Train/LearningRate', lr, step)

        timer['statistics'] += timetracker.split_time()

    # statistics of time consumption and loss
    proportion = {
        k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
        for k, v in timer.items()
    }
    print(
        '\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
    print(
        '\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(
            **proportion))


if __name__ == '__main__':

    n_batch_size = 2
    ntu_dataset = NTUDataset(
        "/Users/yangzhiping/Documents/deepl/Apb-gcn/utils/nturgbd",
        batch_size=n_batch_size,
        benchmark='cv',
        part='val',
        ignored_sample_path=None,
        plan="synergy_matrix")

    ntu_dataloader = DataLoader(ntu_dataset, batch_size=n_batch_size, shuffle=True)
    get_pair = Pairs()

    count = 0
    i = 0
    batch = None
    pairs = get_pair.total_collection

    '''
    for b in (ntu_dataloader):
        batch = b
        matrix1, matrix2 = to_synergy_matrix(batch, pairs)
    print(count)
    '''

    eval_interval = 5
    num_epoch = 100
    for epoch in range(num_epoch):
        eval_model = ((epoch + 1) % eval_interval == 0) or (
                epoch + 1 == num_epoch)

        train(ntu_dataloader, epoch, pairs)

'''
        if eval_model:
            self.eval(
                epoch,
                save_score=self.args.save_score)
'''
