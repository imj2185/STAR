import os.path as osp
import time

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from tqdm import tqdm, trange

from args import make_args
from data.dataset3 import SkeletonDataset
from models.net import DualGraphEncoder
from optimizer import get_std_opt
from utility.helper import make_checkpoint, load_checkpoint


def run_epoch(data_loader,
              model,
              optimizer,
              loss_compute,
              dataset,
              device,
              is_train=True,
              desc=None):
    """Standard Training and Logging Function

        :param data_loader:
        :param model:
        :param optimizer:
        :param loss_compute:
        :param dataset:
        :param device:
        :param is_train:
        :param desc:
    """
    # torch.autograd.set_detect_anomaly(True)
    running_loss = 0.
    accuracy = 0.
    correct = 0
    total_samples = 0
    start = time.time()
    for i, batch in tqdm(enumerate(data_loader),
                         total=len(dataset),
                         desc=desc):
        batch = batch.to(device)
        sample, label, bi = batch.x, batch.y, batch.batch

        with torch.set_grad_enabled(is_train):
            out = model(sample, adj=dataset.skeleton_, bi=bi)
            loss = loss_compute(out, label.long())
            # if training, backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # statistics
            running_loss += loss.item()
            pred = torch.max(out, 1)[1]
            total_samples += label.size(0)
            correct += (pred == label).double().sum().item()

    elapsed = time.time() - start
    accuracy = correct / total_samples * 100.
    print('------ loss: %.3f; accuracy: %.3f; average time: %.4f' %
          (running_loss / len(dataset),
           accuracy,
           elapsed / len(dataset)))

    return running_loss, accuracy


def main():
    # torch.cuda.empty_cache()
    args = make_args()
    device = torch.device('cuda:0') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # download and save the dataset
    train_ds = SkeletonDataset(args.root, name='ntu_60',
                               use_motion_vector=False,
                               benchmark='xsub', sample='train')
    valid_ds = SkeletonDataset(args.root, name='ntu_60',
                               use_motion_vector=False,
                               benchmark='xsub', sample='val')

    # randomly split into around 80% train, 10% val and 10% train
    train_loader = DataLoader(train_ds.data,
                              batch_size=args.batch_size,
                              shuffle=True)
    valid_loader = DataLoader(valid_ds.data,
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
                             linear_temporal=True,
                             sequential=False)
    model = model.to(device)
    noam_opt = get_std_opt(model, args)
    if args.load_model:
        last_epoch = args.load_epoch
        last_epoch, loss = load_checkpoint(osp.join(args.save_root,
                                                    args.save_name + '_' + str(last_epoch) + '.pickle'),
                                           model, noam_opt.optimizer)
        print("Load Model: ", last_epoch)

    loss_compute = nn.CrossEntropyLoss().to(device)

    for epoch in trange(last_epoch, args.epoch_num + last_epoch):
        # print('Epoch: {} Training...'.format(epoch))
        model.train(True)

        loss, accuracy = run_epoch(train_loader, model, noam_opt.optimizer,
                                   loss_compute, train_ds, device, is_train=True,
                                   desc="Train Epoch {}".format(epoch))
        print('Epoch: {} Evaluating...'.format(epoch))
        # TODO Save model
        if epoch % args.epoch_save == 0:
            make_checkpoint(args.save_root, args.save_name, epoch, model, noam_opt.optimizer, loss)

        # Validation
        model.eval()
        loss, accuracy = run_epoch(valid_loader, model, noam_opt.optimizer,
                                   loss_compute, valid_ds, device, is_train=False,
                                   desc="Valid Epoch {}".format(epoch))


if __name__ == "__main__":
    main()
