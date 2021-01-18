import os
import sys
import logging
import torch
import numpy as np
from dgl.data import LegacyTUDataset


def node_label_as_feature(dataset:LegacyTUDataset, mode="concat"):
    """
    Description
    -----------
    Add node labels to graph node features dict

    Parameters
    ----------
    dataset : LegacyTUDataset
        The dataset object
    concat : str, optional
        How to add node label to the graph. Valid options are "add",
        "replace" and "concat".
        - "add": Directly add node_label to graph node feature dict.
        - "concat": Concatenate "feat" and "node_label"
        - "replace": Use "node_label" as "feat"
        Default: :obj:`"concat"`
    """
    # check if node label is not available
    if not os.path.exists(dataset._file_path("node_labels")) or len(dataset) == 0:
        logging.warning("No Node Label Data")
        return dataset
    
    # check if graph has "feat"
    if "feat" not in dataset[0][0].ndata:
        logging.warning("Dataset has no node feature 'feat'")
        if mode.lower() == "concat":
            mode = "replace"
    
    # first read node labels
    DS_node_labels = dataset._idx_from_zero(
        np.loadtxt(dataset._file_path("node_labels"), dtype=int))
    one_hot_node_labels = dataset._to_onehot(DS_node_labels)
    
    # read graph idx
    DS_indicator = dataset._idx_from_zero(
        np.genfromtxt(dataset._file_path("graph_indicator"), dtype=int))
    node_idx_list = []
    for idx in range(np.max(DS_indicator) + 1):
        node_idx = np.where(DS_indicator == idx)
        node_idx_list.append(node_idx[0])
    
    # add to node feature dict
    for idx, g in zip(node_idx_list, dataset.graph_lists):
        node_labels_tensor = torch.tensor(one_hot_node_labels[idx, :])
        if mode.lower() == "concat":
            g.ndata["feat"] = torch.cat(
                (g.ndata["feat"], node_labels_tensor), dim=1)
        elif mode.lower() == "add":
            g.ndata["node_label"] = node_labels_tensor
        else: # replace
            g.ndata["feat"] = node_labels_tensor
    
    dataset.save()
    return dataset


def degree_as_feature(dataset):
    """
    Use node degree (in one-hot format) as node feature
    """
    # first check if already have such feature
    feat_name = "degree"
    g = dataset.graph_lists[0]
    if feat_name in g.ndata:
        return dataset

    logging.warning("Adding node degree into node features...")
    min_degree = sys.maxsize
    max_degree = 0
    for i in range(len(dataset)):
        degrees = dataset.graph_lists[i].in_degrees()
        min_degree = min(min_degree, degrees.min().item())
        max_degree = max(max_degree, degrees.max().item())
    
    vec_len = max_degree - min_degree + 1
    for i in range(len(dataset)):
        num_nodes = dataset.graph_lists[i].num_nodes()
        node_feat = torch.zeros((num_nodes, vec_len))
        degrees = dataset.graph_lists[i].in_degrees()
        node_feat[torch.arange(num_nodes), degrees - min_degree] = 1.
        dataset.graph_lists[i].ndata[feat_name] = node_feat
    dataset.save()
    return dataset

