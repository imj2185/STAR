# STAR
<ins>A</ins>ttention-<ins>B</ins>ased <ins>P</ins>arts-oriented <ins>G</ins>raph <ins>C</ins>onvolution <ins>N</ins>etworks 
<ins>S</ins>parse <ins>T</ins>ransformer-based <ins>A</ins>ction <ins>R</ins>ecognition
# Dataset
download ntu rgb+d 60 action recognition from skeleton from http://rose1.ntu.edu.sg/datasets/actionRecognition.asp

or use google drive 

[NTU60](https://drive.google.com/open?id=1CUZnBtYwifVXS21yVg62T-vrPVayso5H)
[NTU120](https://drive.google.com/open?id=1tEbuaEqMxAV7dNc4fqu1O4M7mC6CJ50w)

uzip data as the following file structure: APBGCN/raw/.\*skeleton (create "raw" directory under APBGCN and put skeleton files)

run the code below to generate dataset:
```python
python datagen.py
```

# Training
git fetch and checkout to "distributed" branch
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

