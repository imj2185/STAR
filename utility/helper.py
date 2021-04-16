import torch
import os.path as osp


def make_checkpoint(root, name, epoch, model, optimizer, loss):
    if not osp.exists(root):
        import os
        os.mkdir(root)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, osp.join(root, name + '_' + str(epoch) + ".pt"))


def load_checkpoint(path, model, optimizer=None, map_location=None, device='cpu'):
    if map_location is None or device in 'cpu':
        map_location = torch.device('cpu')
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

