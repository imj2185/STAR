import torch
import torch.optim as optim
import torch.nn.functional as fn
from data.dataset import SkeletonDataset
from torch_geometric.data import DataLoader
from models.net import DualGraphTransformer
from einops import rearrange
from args import make_args
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

    def train(self, n_epochs):
        self.model.train(True)
        for epoch in range(n_epochs):
            total_loss, _ = run_epoch()
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']

    def run_epoch(self):
        for i, batch in enumerate(train_loader):
            noam_opt.optimizer.zero_grad()
            output = model(batch.x, adj=self.adj)

def main():
    args = make_args()

    log_dir = '/home/project/gcn/APBGCN/log'
    train_dataset = SkeletonDataset(root="/home/project/gcn/Apb-gcn/NTU-RGB+D", name='cv_train', benchmark='cv', sample = 'train')
    valid_dataset = SkeletonDataset(root="/home/project/gcn/Apb-gcn/NTU-RGB+D", name='cv_val', benchmark='cv', sample = 'val')
    
    train_loader = DataLoader(train_dataset.data, batch_size = args.batch_size)
    valid_loader = DataLoader(valid_dataset.data, batch_size = args.batch_size)

    model = DualGraphTransformer(in_channels = 7, hidden_channels = 16, out_channels = 16, num_layers = 4, num_heads = 8)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    trainer = GCNTrainer(model, train_loader, train_dataset.labels, valid_loader, valid_dataset.labels, train_dataset.skeleton_, optimizer, loss_fn, log_dir)
    trainer.train(args.epoch_num)