# APBGCN
<ins>A</ins>ttention-<ins>B</ins>ased <ins>P</ins>arts-oriented <ins>G</ins>raph <ins>C</ins>onvolution <ins>N</ins>etworks 

# Dataset
download ntu rgb+d 60 action recognition from skeleton from http://rose1.ntu.edu.sg/datasets/actionRecognition.asp

uzip data as the following file structure: APBGCN/nturgb+d_skeletons/raw/.\*skeleton

change default dataset_root in args.py : dataset_root=osp.join(os.getcwd(),"nturgb+d_skeletons"),
