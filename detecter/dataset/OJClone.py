import itertools
import json
from typing import *

import torch
from torch.utils import data

from .. import parser, tree_tools
from ..word2vec import create_word_dict


class DataSet(data.Dataset):
    def __init__(self, data_path, max_node_count=None) -> None:
        super().__init__()
        self.raw_data_list = list()
        self.tree_VE_list = list()
        self.word_dict = dict()  # str -> vector

        with open(data_path, "r") as f:
            lines = f.readlines()
        self.raw_data_list = [json.loads(line) for line in lines]

        save_path = data_path + ".pt"

        try:
            save = torch.load(save_path)
            self.tree_VE_list = save["tree_VE_list"]
            self.word_dict = save["word_dict"]

        except IOError:
            self.tree_VE_list = list(map(parser.parse, (raw["code"] for raw in self.raw_data_list)))
            nodes_list = [tree_V for tree_V, tree_E in self.tree_VE_list]
            self.word_dict = create_word_dict(list(itertools.chain(*nodes_list)) + ["<CODE_COMPARE>"])

            torch.save(
                {
                    "tree_VE_list": self.tree_VE_list,
                    "word_dict": self.word_dict,
                },
                save_path,
            )

        if max_node_count:
            self.tree_VE_list = [tree_tools.tree_VE_prune(tree_VE, max_node_count) for tree_VE in self.tree_VE_list]

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, torch.Tensor]:
        label = self.raw_data_list[index]["label"]
        tree_VE = self.tree_VE_list[index]

        nodes, mask = tree_tools.tree_VE_to_tensor(tree_VE, word2vec_cache=self.word_dict)
        return int(label) - 1, nodes, mask

    def __len__(self) -> int:
        return len(self.raw_data_list)
