from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()
    # general
    parser.add_argument('--dataset_name', dest='dataset_name', default='NTU',
                        type=str, help='RND3SAT DIMACS')
    parser.add_argument('--dataset_root', dest='dataset_root', default='dataset',
                        type=str, help='RND3SAT DIMACS')
    parser.add_argument('--loss', dest='loss', default='l2', type=str,
                        help='l2; cross_entropy')
    parser.add_argument('--gpu', dest='use_gpu', default=True, type=bool,
                        help='whether use gpu')
    parser.add_argument('--cuda', dest='cuda', default='0', type=str)

    # dataset
    parser.add_argument('--graph_valid_ratio', dest='graph_valid_ratio', default=0.1, type=float)
    parser.add_argument('--graph_test_ratio', dest='graph_test_ratio', default=0.1, type=float)
    parser.add_argument('--feature_transform', dest='feature_transform', type=bool, default=False,
                        help='whether pre-transform feature')

    # model
    parser.add_argument('--linear_temporal', dest='linear_temporal', type=bool, default=True,
                        help='set to linear temporal Transformer model')
    parser.add_argument('--drop_rate', dest='drop_rate', type=float, default=0.0,
                        help='whether dropout rate, default 0.5')
    parser.add_argument('--load_model', dest='load_model', default=False, type=bool,
                        help='whether load_model')
    parser.add_argument('--load_epoch', dest='load_epoch', default=0, type=int,
                        help='whether load_model')
    parser.add_argument('--batch_size', dest='batch_size', default=16,
                        type=int)  # implemented via accumulating gradient
    parser.add_argument('--num_enc_layers', dest='num_enc_layers', default=6, type=int)
    parser.add_argument('--num_conv_layers', dest='num_conv_layers', default=6, type=int)
    parser.add_argument('--activation', dest='activation', default='relu', type=str)
    parser.add_argument('--in_channels', dest='in_channels', default=6, type=int)
    parser.add_argument('--hid_channels', dest='hid_channels', default=32, type=int)
    parser.add_argument('--out_channels', dest='out_channels', default=32, type=int)
    parser.add_argument('--heads', dest='heads', default=32, type=int)

    # Training Setting up
    parser.add_argument('--lr', dest='lr', default=0.1, type=float)
    parser.add_argument('--weight_decay', dest='weight_decay', default=0.01, type=float)
    parser.add_argument('--warmup_steps', dest='warmup_steps', default=6e5, type=float)
    parser.add_argument('--opt_train_factor', dest='opt_train_factor', default=4, type=float)
    parser.add_argument('--epoch_num', dest='epoch_num', default=200, type=int)  # paper used: 2001
    parser.add_argument('--epoch_log', dest='epoch_log', default=50, type=int)  # test every
    parser.add_argument('--epoch_save', dest='epoch_save', default=500, type=int)  # save every
    parser.add_argument('--save_root', dest='save_root', default='saved_model', type=str)
    parser.add_argument('--save_name', dest='save_name', default='check_point', type=str)
    parser.add_argument('--model_dim', dest='model_dim', default=150, type=int)

    parser.set_defaults(gpu=True,
                        batch_size=32,
                        dataset_name='NTU',
                        dataset_root='dataset',
                        load_model=False,
                        in_channels=32,
                        hid_channels=32,
                        out_channels=32,
                        heads=8)
    args = parser.parse_args()
    return args
