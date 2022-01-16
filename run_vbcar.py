import argparse
import sys

sys.path.append("../")

import os
import random

import numpy as np

from beta_rec.data.grocery_data import GroceryData
from beta_rec.datasets.instacart import Instacart_25

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def parse_args():
    """Parse args from command line.

    Returns:
        args object.
    """
    parser = argparse.ArgumentParser(description="Run VBCAR..")
    parser.add_argument("--item_fea_type", nargs="?", type=str, help="--item_fea_type")
    parser.add_argument("--device", nargs="?", type=str, help="--devide")

    return parser.parse_args()


def train(args):
    seed = 2021
    random.seed(seed)  # Fix random seeds for reproducibility
    np.random.seed(seed)

    # make sure that you have already download the Instacart data from this link: https://www.kaggle.com/c/instacart-market-basket-analysis#
    # uncompressed them and put them in this folder: ../datasets/instacart_25/raw/*.csv

    dataset = Instacart_25(
        min_u_c=20, min_i_c=30, min_o_c=10
    )  # Specifying the filtering conditions.

    # Split the data
    split_dataset = dataset.load_temporal_basket_split(test_rate=0.2, n_test=10)
    data = GroceryData(split_dataset)
    config = {"config_file": "../configs/vbcar_default.json"}
    config["n_sample"] = 1000000  # To reduce the test running time
    config["max_epoch"] = 80
    config["emb_dim"] = 64
    config["late_dim"] = 512
    config["root_dir"] = "/home/zm324/workspace/beta-recsys/"
    config["dataset"] = "instacart_25"
    config["batch_size"] = 512
    config["lr"] = 0.001

    # config["item_fea_type"] = "random_word2vec"
    # config["tunable"] = [
    #     {"name": "lr", "type": "choice", "values": [0.5, 0.05, 0.025, 0.001, 0.005]},
    # ]
    # config["tune"] = True
    # the 'config_file' key is required, that is used load a default config.
    # Other keys can be specified to replace the default settings.
    from beta_rec.recommenders import VBCAR

    config["item_fea_type"] = args.item_fea_type
    config["device"] = args.device
    model = VBCAR(config)
    model.train(data)
    model.test(data.test)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    train(args)
