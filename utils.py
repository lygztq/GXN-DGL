import argparse
import logging
import math
import os
import random

import numpy as np
import torch
import torch.cuda
from scipy.stats import t


def get_stats(array, conf_interval=False, name=None, stdout=False, logout=False):
    """Compute mean and standard deviation from an numerical array
    
    Args:
        array (array like obj): The numerical array, this array can be 
            convert to :obj:`torch.Tensor`.
        conf_interval (bool, optional): If True, compute the confidence interval bound (95%)
            instead of the std value. (default: :obj:`False`)
        name (str, optional): The name of this numerical array, for log usage.
            (default: :obj:`None`)
        stdout (bool, optional): Whether to output result to the terminal. 
            (default: :obj:`False`)
        logout (bool, optional): Whether to output result via logging module.
            (default: :obj:`False`)
    """
    eps = 1e-9
    array = torch.Tensor(array)
    std, mean = torch.std_mean(array)
    std = std.item()
    mean = mean.item()
    center = mean

    if conf_interval:
        n = array.size(0)
        se = std / (math.sqrt(n) + eps)
        t_value = t.ppf(0.975, df=n-1)
        err_bound = t_value * se
    else:
        err_bound = std

    # log and print
    if name is None:
        name = "array {}".format(id(array))
    log = "{}: {:.4f}(+-{:.4f})".format(name, center, err_bound)
    if stdout:
        print(log)
    if logout:
        logging.info(log)

    return center, err_bound


def get_batch_id(num_nodes:torch.Tensor):
    """Convert the num_nodes array obtained from batch graph to batch_id array
    for each node.
    Args:
        num_nodes (torch.Tensor): The tensor whose element is the number of nodes
            in each graph in the batch graph.
    """
    batch_size = num_nodes.size(0)
    batch_ids = []
    for i in range(batch_size):
        item = torch.full((num_nodes[i],), i, dtype=torch.long, device=num_nodes.device)
        batch_ids.append(item)
    return torch.cat(batch_ids)


def topk(x:torch.Tensor, ratio:float, batch_id:torch.Tensor, num_nodes:torch.Tensor):
    """The top-k pooling method. Given a graph batch, this method will pool out some
    nodes from input node feature tensor for each graph according to the given ratio.
    Args:
        x (torch.Tensor): The input node feature batch-tensor to be pooled.
        ratio (float): the pool ratio. For example if :obj:`ratio=0.5` then half of the input
            tensor will be pooled out.
        batch_id (torch.Tensor): The batch_id of each element in the input tensor.
        num_nodes (torch.Tensor): The number of nodes of each graph in batch.
    
    Returns:
        perm (torch.Tensor): The index in batch to be kept.
        k (torch.Tensor): The remaining number of nodes for each graph.
    """
    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()
    
    cum_num_nodes = torch.cat(
        [num_nodes.new_zeros(1),
         num_nodes.cumsum(dim=0)[:-1]], dim=0)
    
    index = torch.arange(batch_id.size(0), dtype=torch.long, device=x.device)
    index = (index - cum_num_nodes[batch_id]) + (batch_id * max_num_nodes)

    dense_x = x.new_full((batch_size * max_num_nodes, ), torch.finfo(x.dtype).min)
    dense_x[index] = x
    dense_x = dense_x.view(batch_size, max_num_nodes)

    _, perm = dense_x.sort(dim=-1, descending=True)
    perm = perm + cum_num_nodes.view(-1, 1)
    perm = perm.view(-1)

    k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
    mask = [
        torch.arange(k[i], dtype=torch.long, device=x.device) + 
        i * max_num_nodes for i in range(batch_size)]

    mask = torch.cat(mask, dim=0)
    perm = perm[mask]

    return perm, k


def parse_args():
    parser = argparse.ArgumentParser("Graph Cross Network")
    parser.add_argument("--pool_ratios", nargs="+", type=float,
                        help="The pooling ratios used in graph cross layers")
    parser.add_argument("--hidden_dim", type=int, default=96,
                        help="The number of hidden channels in GXN")
    parser.add_argument("--cross_weight", type=float, default=1.,
                        help="Weight parameter used in graph cross layer")
    parser.add_argument("--fuse_weight", type=float, default=1., 
                        help="Weight parameter for feature fusion")
    parser.add_argument("--num_cross_layers", type=int, default=2, 
                        help="The number of graph corss layers")
    parser.add_argument("--readout_nodes", type=int, default=30, 
                        help="Number of nodes for each graph after final graph pooling")
    parser.add_argument("--conv1d_dims", nargs="+", type=int, 
                        help="Number of channels in conv operations in the end of graph cross net")
    parser.add_argument("--conv1d_kws", nargs="+", type=int, 
                        help="Kernel sizes of conv1d operations")
    parser.add_argument("--dropout", type=float, default=0., 
                        help="Dropout rate")
    parser.add_argument("--embed_dim", type=int, default=1024, 
                        help="Number of channels of graph embedding")
    parser.add_argument("--final_dense_hidden_dim", type=int, default=128,
                        help="The number of hidden channels in final dense layers")

    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.,
                        help="Weight decay rate")
    parser.add_argument("--epochs", type=int, default=1000, 
                        help="Number of training epochs")
    parser.add_argument("--num_trials", type=int, default=1,
                        help="Number of trials")

    parser.add_argument("--device", type=int, default=0, 
                        help="Computation device id, -1 for cpu")
    parser.add_argument("--dataset", type=str, default="DD", 
                        help="Dataset used for training")
    parser.add_argument("--seed", type=int, default=-1, 
                        help="Random seed, -1 for unset")
    parser.add_argument("--print_every", type=int, default=10,
                        help="Print train log every ? epochs, -1 for silence training")
    parser.add_argument("--dataset_path", type=str, default="./datasets", 
                        help="Path holding your dataset")
    parser.add_argument("--output_path", type=str, default="./output", 
                        help="Path holding your result files")

    args = parser.parse_args()

    # default value for list hyper-parameters
    if not args.pool_ratios or len(args.pool_ratios) < 2:
        args.pool_ratios = [0.8, 0.7]
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
    else:
        logging.warning("Device: {}".format(args.device))

    # random seed
    if args.seed >= 0:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        if args.device != "cpu":
            torch.cuda.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    # print every
    if args.print_every < 0:
        args.print_every = args.epochs + 1
    
    # path
    paths = [args.output_path, args.dataset_path]
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)

    # datasets ad-hoc
    if args.dataset in ['COLLAB', 'IMDBBINARY', 'IMDBMULTI', 'ENZYMES']:
        args.degree_as_feature = True
    else:
        args.degree_as_feature = False

    return args

# if __name__ == "__main__":
#     args = parse_args()
#     print(args.degree_as_feature)
