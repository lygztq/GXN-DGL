import json
import os
from time import time
from datetime import datetime

import dgl
from dgl.data import LegacyTUDataset
from torch.utils.data import random_split
import torch

from dataloader import GraphDataLoader
from networks import GraphClassifier
from utils import get_stats, parse_args, degree_as_feature


def compute_loss(*args, **kwargs):
    pass


def train(*args, **kwargs):
    pass


@torch.no_grad()
def test(*args, **kwargs):
    pass


def main(args):
    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    dataset = LegacyTUDataset(args.dataset, raw_dir=args.dataset_path)

    # add self loop. We add self loop for each graph here since the function "add_self_loop" does not
    # support batch graph.
    for i in range(len(dataset)):
        dataset.graph_lists[i] = dgl.add_self_loop(dataset.graph_lists[i])
    
    # use degree as node feature
    if args.degree_as_feature:
        dataset = degree_as_feature(dataset)

    num_training = int(len(dataset) * 0.9)
    num_test = len(dataset) - num_training
    train_set, test_set = random_split(dataset, [num_training, num_test])

    train_loader = GraphDataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=6)
    test_loader = GraphDataLoader(test_set, batch_size=args.batch_size, num_workers=2)

    device = torch.device(args.device)
    
    # Step 2: Create model =================================================================== #
    num_feature, num_classes, _ = dataset.statistics()
    if args.degree_as_feature:
        num_feature = dataset.graph_lists[0].ndata["degree"].size(1)
    args.in_dim = num_feature
    args.out_dim = num_classes
    args.edge_feat_dim = 0 # No edge feature in datasets that we use.
    
    model = GraphClassifier(args).to(device)

    # Step 3: Create training components ===================================================== #
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Step 4: training epoches =============================================================== #
    best_test_acc = 0.0
    best_epoch = -1
    train_times = []
    for e in range(args.epochs):
        s_time = time()
        train_loss = train(model, optimizer, train_loader, device)
        train_times.append(time() - s_time)
        test_acc, _ = test(model, test_loader, device)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = e + 1

        if (e + 1) % args.print_every == 0:
            log_format = "Epoch {}: loss={:.4f}, test_acc={:.4f}, best_test_acc={:.4f}"
            print(log_format.format(e + 1, train_loss, test_acc, best_test_acc))
    print("Best Epoch {}, final test acc {:.4f}".format(best_epoch, best_test_acc))
    return best_test_acc, sum(train_times) / len(train_times)


if __name__ == "__main__":
    args = parse_args()
    res = []
    train_times = []
    for i in range(args.num_trials):
        print("Trial {}/{}".format(i + 1, args.num_trials))
        # acc, train_time = main(args)
        acc, train_time = 0, 0
        res.append(acc)
        train_times.append(train_time)

    mean, err_bd = get_stats(res, conf_interval=False)
    print("mean acc: {:.4f}, error bound: {:.4f}".format(mean, err_bd))

    out_dict = {"hyper-parameters": vars(args),
                "result_date": datetime.now(),
                "result": "{:.4f}(+-{:.4f})".format(mean, err_bd),
                "train_time": "{:.4f}".format(sum(train_times) / len(train_times))}

    with open(os.path.join(args.output_path, "train.log"), "w") as f:
        json.dump(out_dict, f, sort_keys=True, indent=4)
