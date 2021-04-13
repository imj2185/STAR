import torch
from torch.profiler import profiler
from torch_geometric.data import DataLoader
from tqdm import tqdm

from args import make_args
from data.dataset3 import SkeletonDataset, skeleton_parts
from models.net2s import DualGraphEncoder


import os.path as osp
# from utility.helper import load_checkpoint


def time_trace_handler(p):
    print(p.key_averages().table(
        sort_by="self_cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
        row_limit=-1))


def memory_trace_handler(p):
    print(p.key_averages().table(
        sort_by="self_cuda_memory_usage" if torch.cuda.is_available() else "cpu_memory_usage",
        row_limit=10))


def run(model, dataloader, total_batch, adj, prof, device):
    for i, batch in tqdm(enumerate(dataloader),
                         total=total_batch,
                         desc="Profiling"):
        batch = batch.to(device)
        sample, label, bi = batch.x, batch.y, batch.batch
        with torch.no_grad():
            out = model(sample, adj=adj, bi=bi)
        prof.step()
    return out


def profile(device, _args):
    ds = SkeletonDataset(osp.join(_args.dataset_root, 'dataset/ntu60'), name='ntu_60',
                         use_motion_vector=False, sample='val')
    # Load model
    model = DualGraphEncoder(in_channels=_args.in_channels,
                             hidden_channels=_args.hid_channels,
                             out_channels=_args.out_channels,
                             mlp_head_hidden=_args.mlp_head_hidden,
                             num_layers=_args.num_enc_layers,
                             num_heads=_args.heads,
                             use_joint_mean=False,
                             num_conv_layers=_args.num_conv_layers,
                             drop_rate=_args.drop_rate).to(device)
    adj = skeleton_parts()[0].to(device)
    num_batches = 100; ds = ds[:args.batch_size * num_batches]
    dl = DataLoader(ds, batch_size=args.batch_size)
    model.eval()
    total_batch_ = int(len(ds) / args.batch_size) + 1

    print('Profiling of performance ....')
    with profiler.profile(record_shapes=True,
                          activities=[
                              torch.profiler.ProfilerActivity.CPU,
                              torch.profiler.ProfilerActivity.CUDA],
                          schedule=torch.profiler.schedule(
                              wait=1,
                              warmup=5,
                              active=2),
                          on_trace_ready=time_trace_handler
                          ) as prof:
        _ = run(model, dl, total_batch_, adj, prof, device)

    prof.export_chrome_trace(osp.join(args.save_root, "time_trace.json"))

    print('Profiling of memory usage ....')
    with profiler.profile(record_shapes=True,
                          activities=[
                              torch.profiler.ProfilerActivity.CPU,
                              torch.profiler.ProfilerActivity.CUDA],
                          schedule=torch.profiler.schedule(
                              wait=1,
                              warmup=5,
                              active=2),
                          with_stack=True,
                          profile_memory=True,
                          on_trace_ready=memory_trace_handler) as prof:
        _ = run(model, dl, total_batch_, adj, prof, device)

    prof.export_chrome_trace(osp.join(args.save_root, "memory_trace.json"))


if __name__ == '__main__':
    args = make_args()
    dev = torch.device('cuda:0') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
    profile(device=dev, _args=args)
