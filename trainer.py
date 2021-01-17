import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
from tensorboardX import SummaryWriter
from torch_geometric.data import DataLoader
from tqdm import tqdm
import os

from args import make_args
from data.dataset import SkeletonDataset
from models.net import DualGraphTransformer
from optimizer import get_std_opt


class GCNTrainer(object):
    def __init__(self, model, train_loader, train_labels, val_loader, val_labels, adj, optimizer, loss_fn, log_dir):

        self.model = model
        self.train_loader = train_loader
        self.train_labels = train_labels
        self.val_loader = val_loader
        self.val_labels = val_labels
        self.loss_fn = loss_fn
        self.log_dir = log_dir
        self.adj = adj
        self.optimizer = optimizer
        self.num_classes = 60
        self.device = torch.device('cuda:0')

        self.model = self.model.to(self.device)
        self.adj = self.adj.to(self.device)
        self.train_labels = self.train_labels.to(self.device)
        self.val_labels = self.val_labels.to(self.device)
        if self.log_dir is not None:
            self.writer = SummaryWriter(log_dir)

    def train(self, n_epochs):

        best_acc = 0    
        i_acc = 0
        self.model.train(True)

        for epoch in range(n_epochs):
            # plot learning rate
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.writer.add_scalar('params/lr', lr, epoch)

            for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Train Epoch {}".format(epoch)):
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(batch.x, adj=self.adj, bi=batch.batch)
                target = batch.y  # - 1
                # one_hot = fn.one_hot(target.long(), num_classes = 60)

                # loss = fn.cross_entropy(output, one_hot)
                loss = fn.cross_entropy(output, target.long())
                self.writer.add_scalar('train/train_loss', loss, i_acc + i + 1)

                pred = torch.max(output, 1)[1]
                results = pred == target
                correct_points = torch.sum(results.long())

                acc = correct_points.float() / results.size()[0]
                self.writer.add_scalar('train/train_overall_acc', acc, i_acc + i + 1)

                loss.backward(retain_graph=True)
                self.optimizer.step()

                log_str = 'epoch %d, step %d: train_loss %.3f; train_acc %.3f' % (epoch + 1, i + 1, loss, acc)
                if (i + 1) % 1 == 0:
                    print(log_str)
            i_acc += i

            #evaluation
            with torch.no_grad():
                #loss, val_overall_acc, val_mean_class_acc = self.update_validation_accuracy()
                loss, val_overall_acc = self.update_validation_accuracy()
            #self.writer.add_scalar('val/val_mean_class_acc', val_mean_class_acc, epoch+1)
            self.writer.add_scalar('val/val_overall_acc', val_overall_acc, epoch+1)
            self.writer.add_scalar('val/val_loss', loss, epoch+1)

            # save best model
            if val_overall_acc > best_acc:
                best_acc = val_overall_acc
                #self.model.save(self.log_dir, epoch)
                torch.save(self.model.state_dict(), 
                    os.path.join(self.log_dir, 
                    "best_model.pth"))
 
            # adjust learning rate manually
            if epoch > 0 and (epoch + 1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5

        # export scalar data to JSON for external processing
        self.writer.export_scalars_to_json(self.log_dir + "/all_scalars.json")
        self.writer.close()

    def update_validation_accuracy(self):
        all_correct_points = 0
        all_points = 0

        wrong_class = np.zeros(self.num_classes)
        samples_class = np.zeros(self.num_classes)
        all_loss = 0

        self.model.eval()

        total_time = 0.0
        total_print_time = 0.0
        all_target = []
        all_pred = []

        for i, batch in enumerate(self.val_loader, 0):

            batch = batch.to(self.device)
            output = self.model(batch.x, adj=self.adj, bi=batch.batch)
            target = batch.y - 1

            pred = torch.max(output, 1)[1]
            all_loss += self.loss_fn(output, target).cpu().data.numpy()
            results = pred == target

            for i in range(results.size()[0]):
                if not bool(results[i].cpu().data.numpy()):
                    wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
            correct_points = torch.sum(results.long())

            all_correct_points += correct_points
            all_points += results.size()[0]

        print('Total # of test models: ', all_points)
        # val_mean_class_acc = np.mean((samples_class - wrong_class) / samples_class)
        acc = all_correct_points.float() / all_points
        val_overall_acc = acc.cpu().data.numpy()
        loss = all_loss / len(self.val_loader)

        # print ('val mean class acc. : ', val_mean_class_acc)
        print('val overall acc. : ', val_overall_acc)
        print('val loss : ', loss)

        self.model.train()

        # return loss, val_overall_acc, val_mean_class_acc
        return loss, val_overall_acc


if __name__ == '__main__':
    args = make_args()

    log_dir = '/home/dusko/Documents/projects/APBGCN/log'
    train_dataset = SkeletonDataset(root="/home/dusko/Documents/projects/APBGCN", name='ntu60_cs_train_test', use_motion_vector=False,
                                    benchmark='cs', sample='train')
    valid_dataset = SkeletonDataset(root="/home/dusko/Documents/projects/APBGCN", name='ntu60_cs_val_test', use_motion_vector=False,
                                    benchmark='cs', sample='val')

    train_loader = DataLoader(train_dataset.data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset.data, batch_size=args.batch_size, shuffle=True)

    model = DualGraphTransformer(in_channels=3,
                                 hidden_channels=16,
                                 out_channels=16,
                                 num_layers=4,
                                 num_heads=8,
                                 linear_temporal=True,
                                 sequential=False)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))

    noam_opt = get_std_opt(model, args)

    trainer = GCNTrainer(model, train_loader, train_dataset.labels, valid_loader, valid_dataset.labels,
                         train_dataset.skeleton_, noam_opt.optimizer, nn.CrossEntropyLoss(), log_dir)
    trainer.train(args.epoch_num)
