# APBGCN
<ins>A</ins>ttention-<ins>B</ins>ased <ins>P</ins>arts-oriented <ins>G</ins>raph <ins>C</ins>onvolution <ins>N</ins>etworks 

# Dataset
download ntu rgb+d 60 action recognition from skeleton from http://rose1.ntu.edu.sg/datasets/actionRecognition.asp

uzip data as the following file structure: APBGCN/nturgb+d_skeletons/raw/.\*skeleton

change default dataset_root in args.py : dataset_root=osp.join(os.getcwd(),"nturgb+d_skeletons"),

# Training
git fetch and checkout to "best_performing" branch
```python
python train_dist.py -#distributed training
```

# Configuration
```python
parser.set_defaults(gpu=True,
                        batch_size=128,
                        dataset_name='NTU',
                        dataset_root=osp.join(os.getcwd()),
                        load_model=False,
                        in_channels=9,
                        num_enc_layers=5,
                        num_conv_layers=2,
                        weight_decay=4e-5,
                        drop_rate=[0.4, 0.4, 0.4, 0.4],  # linear_attention, sparse_attention, add_norm, ffn
                        hid_channels=64,
                        out_channels=64,
                        heads=8,
                        data_parallel=False,
                        cross_k=5,
                        mlp_head_hidden=128)

parser.set_defaults(gpu=True,
                        batch_size=128,
                        dataset_name='NTU',
                        dataset_root=osp.join(os.getcwd()),
                        load_model=False,
                        in_channels=9,
                        num_enc_layers=5,
                        num_conv_layers=2,
                        weight_decay=4e-5,
                        drop_rate=[0.4, 0.4, 0.4, 0.4],  # linear_attention, sparse_attention, add_norm, ffn
                        hid_channels=128,
                        out_channels=128,
                        heads=8,
                        data_parallel=False,
                        cross_k=5,
                        mlp_head_hidden=128)
```

