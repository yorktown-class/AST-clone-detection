from typing import *

import torch
from torch.utils import data
import json
import multiprocessing
from tqdm import tqdm
import math
import itertools
import numpy

from .. import config
from .. import parser
from ..word2vec import word2vec, create_word_dict
from .. import tree_tools


def convert(code: Dict) -> Tuple[List[str], Tuple[List[int], List[int]]]:
    code_string = code["code"]
    index = code["index"]
    try:
        V, E = parser.parse(code_string, "c")
        return (V, E)
    except parser.ParseError as err:
        logger = multiprocessing.get_logger()
        logger.warn('parse error ("index": "{}")'.format(index))
        return None


def convert_code(code_list: List[Dict]):
    result = []
    with multiprocessing.Pool(processes=config.NUM_CORE) as p:
        iresult = p.imap_unordered(convert, code_list, 20)
        with tqdm(total=len(code_list)) as pbar:
            for item in iresult:
                if item:
                    result.append(item)
                pbar.update()
    return result


class DataSet(data.Dataset):
    def __init__(self, data_path, item_count=None, max_node_count=None) -> None:
        super().__init__()
        self.raw_data_list = list()
        self.tree_VE_list = list()
        self.word_dict = dict() # str -> vector
        self.length = 0

        with open(data_path, "r") as f:
            lines = f.readlines()
        self.raw_data_list = [json.loads(line) for line in lines]
        self.length = len(self.raw_data_list)

        save_path = data_path + ".pt"

        try:
            save = torch.load(save_path)
            self.tree_VE_list = save["tree_VE_list"]
            self.word_dict = save["word_dict"]

        except IOError:
            self.tree_VE_list = list(map(parser.parse, (raw["code"] for raw in self.raw_data_list)))
            nodes_list = [tree_V for tree_V, tree_E in self.tree_VE_list]
            self.word_dict = create_word_dict(list(itertools.chain(*nodes_list)) + ["<CODE_COMPARE>"])

            torch.save({
                "tree_VE_list": self.tree_VE_list, 
                "word_dict": self.word_dict,
            }, save_path)
        
        if max_node_count:
            self.tree_VE_list = [tree_tools.tree_VE_prune(tree_VE, max_node_count) for tree_VE in self.tree_VE_list]
            # index = [idx for idx, (tree_V, tree_E) in enumerate(self.tree_VE_list) if len(tree_V) <= max_node_count]
            # self.tree_VE_list = [self.tree_VE_list[i] for i in index]
            # self.raw_data_list = [self.raw_data_list[i] for i in index]
            # self.length = len(self.raw_data_list)
        if item_count:
            self.length = min(self.length, item_count)

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, torch.Tensor]:
        label = self.raw_data_list[index]["label"]
        tree_VE = self.tree_VE_list[index]

        nodes, mask = tree_tools.tree_VE_to_tensor(tree_VE, word2vec_cache=self.word_dict)
        return int(label) - 1, nodes, mask

    def __len__(self) -> int:
        return self.length

class BiDataSet(DataSet):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.separator = self.length
        self.randlist = torch.randperm(self.length).tolist()
    
    def __getitem__(self, index):
        if index < self.separator:
            i, j = index, index
        else:
            index -= self.separator
            i, j = index, self.randlist[index]
        tree_VE = tree_tools.merge_tree_VE(self.tree_VE_list[i], self.tree_VE_list[j], "<CODE_COMPATE>")
        nodes, mask = tree_tools.tree_VE_to_tensor(tree_VE, self.word_dict)
        return self.raw_data_list[i]["label"] == self.raw_data_list[j]["label"], nodes, mask

    def __len__(self) -> int:
        return self.separator * 2


def collate_fn(batch: List[Union[int, torch.Tensor, torch.Tensor]]):
    label_list = [label for label, nodes, mask in batch]
    nodes_list = [nodes for label, nodes, mask in batch]
    mask_list = [mask for label, nodes, mask in batch]

    label_batch = torch.tensor(label_list, dtype=torch.long)

    node_batch = torch.nn.utils.rnn.pad_sequence(nodes_list)

    n = node_batch.shape[0]

    mask_base = ~torch.eye(n, dtype=torch.bool)
    mask_batch = mask_base.repeat(len(batch), 1, 1)

    for idx, mask in enumerate(mask_list):
        n, m = mask.shape
        mask_batch[idx, :n, :m] = mask
    
    return label_batch, node_batch, mask_batch

