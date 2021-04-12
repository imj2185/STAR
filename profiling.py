import torch
import os.path as osp

from torch.profiler import profiler
from torch_geometric.data import DataLoader
from tqdm import tqdm

from args import make_args
from data.dataset3 import SkeletonDataset, skeleton_parts
from models.net2s import DualGraphEncoder
from utility.helper import load_checkpoint


def profile(device, _args):
    ds = SkeletonDataset(_args.dataset_root, name='ntu_60',
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
    last_epoch = 100
    last_epoch, loss = load_checkpoint(osp.join(_args.save_root,
                                                _args.save_name + '_' + str(last_epoch) + '.pth'),
                                       model)
    dl = DataLoader(ds, batch_size=args.batch_size)
    # warm-up
    model.eval()
    total_batch_ = len(ds) // args.batch_size + 1
    # for i, batch in tqdm(enumerate(dl),
    #                      total=total_batch_,
    #                      desc="Warm up phase"):
    #     batch = batch.to(device)
    #     sample, label, bi = batch.x, batch.y, batch.batch
    #     with torch.no_grad():
    #         out = model(sample, adj=adj, bi=bi)
    # print(out.shape)

    def trace_handler(p):
        print(p.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1))

    with profiler.profile(record_shapes=True,
                          activities=[
                              torch.profiler.ProfilerActivity.CPU,
                              torch.profiler.ProfilerActivity.CUDA],
                          schedule=torch.profiler.schedule(
                              wait=1,
                              warmup=5,
                              active=2),
                          on_trace_ready=trace_handler
                          ) as prof:
        with profiler.record_function("model_inference"):
            for i, batch in tqdm(enumerate(dl),
                                 total=total_batch_,
                                 desc="Warm up phase"):
                batch = batch.to(device)
                sample, label, bi = batch.x, batch.y, batch.batch
                with torch.no_grad():
                    out = model(sample, adj=adj, bi=bi)
                prof.step()

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    with profiler.profile(with_stack=True, profile_memory=True) as prof:
        out, idx = model(input)


if __name__ == '__main__':
    args = make_args()
    dev = torch.device('cuda:0') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
    profile(device=dev, _args=args)
