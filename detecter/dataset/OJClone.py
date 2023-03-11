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
        self.data_path = data_path
        self.data_list = list() # tuple(label, V, E)
        self.word_dict = dict() # str -> vector

        save_path = data_path + "({})".format(item_count) + ".pt"

        try:
            with open(save_path, "rb") as f:
                save = torch.load(f)
            self.data_list = save["data_list"]
            self.word_dict = save["word_dict"]
        
        except IOError:
            with open(data_path, "r") as f:
                lines = f.readlines()
            raw_data_list = [json.loads(line) for line in lines]
            if item_count:
                raw_data_list = raw_data_list[:min(item_count, len(raw_data_list))]
            
            label_list = [raw["label"] for raw in raw_data_list]
            code_tree_list = convert_code(raw_data_list)
            self.data_list = [(label, V, E) for label, (V, E) in zip(label_list, code_tree_list)]
            
            nodes_list = [v for l, v, e in self.data_list]
            self.word_dict = create_word_dict(list(itertools.chain(*nodes_list)))

            with open(save_path, "wb") as f:
                torch.save({
                    "data_list": self.data_list, 
                    "word_dict": self.word_dict,
                }, f)
        
        if max_node_count:
            self.data_list = [(l, v, e) for l, v, e in self.data_list if len(v) <= max_node_count]

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, torch.Tensor]:
        label, V, E = self.data_list[index]

        wordvec_list = [self.word_dict[v] for v in V]
        wordvec_list = numpy.array(wordvec_list)
        tensorV = torch.tensor(wordvec_list, dtype=torch.float)

        tensorE = torch.tensor(E, dtype=torch.long)
        n, _ = tensorV.shape
        _, m = tensorE.shape
        mask = ~torch.eye(n, dtype=torch.bool)
        for _ in range(int(math.log(m, 2))):
            mask[tensorE[1, :]] &= mask[tensorE[0, :]]
        
        return int(label) - 1, tensorV, mask

    def __len__(self) -> int:
        return len(self.data_list)

class BiDataSet(data.Dataset):
    def __init__(self, dataset: DataSet) -> None:
        super().__init__()
        self.dataset = dataset
        n = len(dataset)
        self.index_map = list(range(n))
        self.separator = n

        self.randlist = torch.randperm(n).tolist()
    
    def __getitem__(self, index):
        def merge(batch1, batch2):
            labeli, nodesi, maski = batch1
            labelj, nodesj, maskj = batch2
            leni = nodesi.shape[0]
            lenj = nodesj.shape[0]
            nodes = torch.cat([word2vec("<CODE_COMPARE>").reshape(1, -1), nodesi, nodesj], dim=0)
            mask = torch.ones((1 + leni + lenj, 1 + leni + lenj), dtype=torch.bool)
            mask[0, :] = False
            mask[1: leni+1, 1: leni+1] = maski
            mask[leni+1: leni+lenj+1, leni+1: leni+lenj+1] = maskj
            return labeli == labelj, nodes, mask

        if index < self.separator:
            i, j = index, index
        else:
            index -= self.separator
            i, j = index, self.randlist[index]

        return merge(self.dataset[i], self.dataset[j])

    def __len__(self) -> int:
        return self.separator * 2


def collate_fn(batch: List[Union[str, torch.Tensor, torch.Tensor]]):
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

