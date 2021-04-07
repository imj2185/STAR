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
    }, osp.join(root, name + '_' + str(epoch) + ".pickle"))


def load_checkpoint(path, model, optimizer, map_location=None):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

