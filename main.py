import json
import os

import dgl
import torch

from dataloader import GraphDataLoader
from networks import GraphClassifier
from utils import get_stats, parse_args

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
                "result": "{:.4f}(+-{:.4f})".format(mean, err_bd),
                "train_time": "{:.4f}".format(sum(train_times) / len(train_times))}

    with open(os.path.join(args.output_path, "train.log"), "w") as f:
        json.dump(out_dict, f, sort_keys=True, indent=4)
