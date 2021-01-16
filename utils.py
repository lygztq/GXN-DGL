import argparse
import logging
import os

import torch
import torch.cuda
import random
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser("Graph Cross Network")
    parser.add_argument("--pool_ratios", nargs="+", type=float,
                        help="The pooling ratios used in graph cross layers")
    parser.add_argument("--hidden_dim", type=int, default=96,
                        help="The number of hidden channels")
    parser.add_argument("--cross_weight", type=float, default=1.,
                        help="Weight parameter used in graph cross layer")
    parser.add_argument("--fuse_weight", type=float, default=1., help="Weight parameter for feature fusion")
    parser.add_argument("--num_cross_layers", type=int, default=2, help="The number of graph corss layers")
    parser.add_argument("--readout_nodes", type=int, default=30, help="Number of nodes for each graph after final graph pooling")
    parser.add_argument("--conv1d_dims", nargs="+", type=int, help="Number of channels in conv operations in the end of graph cross net")
    parser.add_argument("--conv1d_kws", nargs="+", type=int, help="Kernel sizes of conv1d operations")
    parser.add_argument("--dropout", type=float, default=0., help="Dropout rate")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")

    parser.add_argument("--device", type=int, default=0, help="Computation device id, -1 for cpu")
    parser.add_argument("--dataset", type=str, default="DD", help="Dataset used for training")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed, -1 for unset")
    parser.add_argument("--data_path", type=str, default="./datasets", help="Path holding your dataset")
    parser.add_argument("--output_path", type=str, default="./output", help="Path holding your result files")

    args = parser.parse_args()

    # default value for list hyper-parameters
    if not args.pool_ratios or len(args.pool_ratios) < 2:
        args.pool_ratios = [0.9, 0.7]
        logging.warning("No valid pool_ratios is given, "
                        "using default value '{}'".format(args.pool_ratios))
    if not args.conv1d_dims or len(args.conv1d_dims) < 2:
        args.conv1d_dims = [16, 32]
        logging.warning("No valid conv1d_dims is give, "
                        "using default value {}".format(args.conv1d_dims))
    if not args.conv1d_kws or len(args.conv1d_kws) < 1:
        args.conv1d_kws = [5]
        logging.warning("No valid conv1d_kws is given, "
                        "using default value '{}'".format(args.conv1d_kws))
    
    # device
    args.device = "cpu" if args.device < 0 else "cuda:{}".format(args.device)
    if not torch.cuda.is_available():
        logging.warning("GPU is not available, using CPU for training")
        args.device = "cpu"

    # random seed
    if args.seed >= 0:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        if args.device != "cpu":
            torch.cuda.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    # path
    paths = [args.output_path, args.data_path]
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)

    return args

if __name__ == "__main__":
    args = parse_args()
    print(vars(args))