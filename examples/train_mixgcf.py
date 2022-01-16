"""isort:skip_file."""
import argparse
import os
import sys
import time

sys.path.append("../")

import numpy as np
import torch
from ray import tune

from beta_rec.core.train_engine import TrainEngine
from beta_rec.models.mixgcf import LightGCNEngine
from beta_rec.utils.common_util import DictToObject
from beta_rec.utils.monitor import Monitor


def parse_args():
    """Parse args from command line.

    Returns:
        args object.
    """
    parser = argparse.ArgumentParser(description="Run MixGCF-LightGCN..")
    parser.add_argument(
        "--config_file",
        nargs="?",
        type=str,
        default="../configs/mixgcf_default.json",
        help="Specify the config file name. Only accept a file from ../configs/",
    )
    # If the following settings are specified with command line,
    # These settings will used to update the parameters received from the config file.
    parser.add_argument(
        "--emb_dim", nargs="?", type=int, help="Dimension of the embedding."
    )
    parser.add_argument(
        "--tune", nargs="?", type=str, default=False, help="Tun parameter",
    )
    parser.add_argument(
        "--n_negs",
        nargs="?",
        type=float,
        help="Number of candidate negatives",
        default=64,
    )
    parser.add_argument("--lr", nargs="?", type=float, help="Initialize learning rate.")
    parser.add_argument(
        "--K",
        nargs="?",
        type=float,
        default=1,
        help="number of negative in K-pair loss",
    )
    parser.add_argument("--max_epoch", nargs="?", type=int, help="Number of max epoch.")
    parser.add_argument(
        "--pool", type=str, default="concat", help="[concat, mean, sum]"
    )
    parser.add_argument("--context_hops", type=int, default=3, help="hop")
    parser.add_argument(
        "--batch_size", nargs="?", type=int, help="Batch size for training."
    )
    return parser.parse_args()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class LightGCN_train(TrainEngine):
    """An instance class from the TrainEngine base class."""

    def __init__(self, config):
        """Initialize LightGCN_train Class.

        Args:
            config (dict): All the parameters for the model.
        """
        self.config = config
        super(LightGCN_train, self).__init__(config)
        self.load_dataset()
        self.build_data_loader()
        self.engine = LightGCNEngine(self.config)

    def build_data_loader(self):
        """Build dataloader for the MixGCF model including the adj matrix and sparse matrix to tensor transferring"""
        adj_mat, norm_adj_mat, mean_adj_mat = self.data.get_adj_mat(self.config)
        norm_adj = sparse_mx_to_torch_sparse_tensor(norm_adj_mat)
        self.config["model"]["norm_adj"] = norm_adj
        self.config["model"]["n_users"] = self.data.n_users
        self.config["model"]["n_items"] = self.data.n_items

    def train(self):
        """Train the model."""
        self.monitor = Monitor(
            log_dir=self.config["system"]["run_dir"], delay=1, gpu_id=self.gpu_id
        )
        self.model_save_dir = os.path.join(
            self.config["system"]["model_save_dir"], self.config["model"]["save_name"]
        )
        self.max_n_update = self.config["model"]["max_n_update"]
        for epoch in range(self.config["model"]["max_epoch"]):
            print(f"Epoch {epoch} starts !")
            print("-" * 80)
            if epoch > 0 and self.eval_engine.n_no_update == 0:
                # previous epoch have already obtained better result
                self.engine.save_checkpoint(model_dir=self.model_save_dir)

            if self.eval_engine.n_no_update >= self.max_n_update:
                print(
                    "Early stop criterion triggered, no performance update for {:} times".format(
                        self.max_n_update
                    )
                )
                break

            train_loader = self.data.instance_mul_neg_loader(
                self.config["model"]["batch_size"],
                self.config["model"]["device_str"],
                self.config["model"]["n_negs"],
            )
            self.engine.train_an_epoch(epoch_id=epoch, train_loader=train_loader)
            self.eval_engine.train_eval(
                self.data.valid[0], self.data.test[0], self.engine.model, epoch
            )
        self.config["run_time"] = self.monitor.stop()

    def test(self):
        """Test the model."""
        self.engine.resume_checkpoint(model_dir=self.model_save_dir)
        super(LightGCN_train, self).test()


def tune_train(config):
    """Train the model with a hypyer-parameter tuner (ray).

    Args:
        config (dict): All the parameters for the model.
    """
    train_engine = LightGCN_train(DictToObject(config))
    best_performance = train_engine.train()
    tune.track.log(valid_metric=best_performance)
    train_engine.test()
    while train_engine.eval_engine.n_worker > 0:
        time.sleep(20)


if __name__ == "__main__":
    args = parse_args()
    train_engine = LightGCN_train(args)
    if args.tune:
        train_engine.tune(tune_train)
    else:
        train_engine.train()
        train_engine.test()
