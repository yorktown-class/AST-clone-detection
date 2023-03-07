from typing import *

import torch
from torch.utils import data
import json
import logging
from torch_geometric.loader import DataLoader as gDataLoader
import random
import itertools
import multiprocessing
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import functools

# from .collate import collate_fn
from ..parser import code2tree
from ..parser.code2tree import Tree


def shuffle_slice(array, start, stop):
    clone = array[start:stop]
    random.shuffle(clone)
    array[start:stop] = clone


def batched(array, n):
    for i in range(0, len(array), n):
        yield(array[i: min(len(array), i + n)])


def convert_impl(data: Dict) -> Tuple[str, Tree]:
    label = data.get("label")
    code = data.get("code")
    code_index = data.get("index")

    try:
        V, E = code2tree.parse(code, "c")
        return (label, V, E)
    except code2tree.ParseError as err:
        logger = multiprocessing.get_logger()
        # logger = logging.getLogger("loader")
        logger.warn('parse error ("index": "{}")'.format(code_index))
        return None


def convert(data_list: List) -> List[Tuple[str, Tree]]:
    label_list = []
    V_list = []
    E_list = []

    with multiprocessing.Pool(processes=12) as p:
        iresult = p.imap_unordered(convert_impl, data_list, 40)
        with tqdm(total=len(data_list)) as pbar:
            for item in iresult:
                if item:
                    label, V, E = item
                    label_list.append(label)
                    V_list.append(V)
                    E_list.append(E)
                pbar.update()

    tpos_list = list(itertools.accumulate([len(v) for v in V_list]))
    node_list = list(itertools.chain(*V_list))

    sentence2emb = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_list = sentence2emb.encode(
        node_list,
        batch_size=1024,
        show_progress_bar=True, 
        device="cuda")
    
    s = 0
    for idx, t in enumerate(tpos_list):
        V_list[idx] = embedding_list[s: t]
        s = t
    
    return [
        (label, Tree(x=torch.tensor(V), edge_index=torch.tensor(E, dtype=torch.long), root=0))
        for label, V, E in zip(label_list, V_list, E_list)
    ]


class DataSet(data.Dataset):
    CHUNK_SIZE = 1000
    def __init__(self, data_path, item_count=None) -> None:
        super().__init__()
        self.data_path = data_path
        self.chunk_path = dict() # int -> str

        with open(data_path, "r") as f:
            lines = f.readlines()
        raw_data_list = [json.loads(line) for line in lines]
        if item_count:
            raw_data_list = raw_data_list[:min(item_count, len(raw_data_list))]

        for spos in range(0, len(raw_data_list), self.CHUNK_SIZE):
            tpos = min(spos + self.CHUNK_SIZE, len(raw_data_list))

            raw_data_slice = raw_data_list[spos: tpos]
            save_path = data_path + "@" + str(tpos) + ".pt"
            try:
                with open(save_path, "rb") as f:
                    pass
            except IOError:
                data_slice = convert(raw_data_slice)
                with open(save_path, "wb") as f:
                    torch.save(data_slice, f)
            self.chunk_path[spos // self.CHUNK_SIZE] = save_path
        
        self.idx_map = list(range(len(raw_data_list)))
        for i in range(0, len(self.idx_map), 1000):
            shuffle_slice(self.idx_map, i, min(len(self.idx_map), i + 1000))

    @functools.lru_cache(maxsize=3)
    def get_chunk(self, idx):
        path = self.chunk_path[idx]
        with open(path, "rb") as f:
            return torch.load(f)

    def __getitem__(self, index) -> Tuple[str, Tree]:
        index = self.idx_map[index]
        
        idx = index // self.CHUNK_SIZE
        chunk = self.get_chunk(idx)

        offset = min(index - idx * self.CHUNK_SIZE, len(chunk) - 1)

        return chunk[offset]

    def __len__(self) -> int:
        return len(self.idx_map)



def DataLoader(data_path, batch_size, item_count=None):
    ds = DataSet(data_path, item_count)
    print(len(ds))
    return gDataLoader(ds, batch_size, shuffle=False, follow_batch=["x"], drop_last=True, num_workers=4)


def verification(labels: List[str], hidden: torch.Tensor, similarity: torch.nn.Module):
    logger = logging.getLogger("verification")

    n, h = hidden.shape

    l_index = [i for i in range(n) for j in range(i + 1, n)]
    r_index = [j for i in range(n) for j in range(i + 1, n)]

    logger.debug(hidden)
    outputs: torch.Tensor = similarity(hidden[l_index], hidden[r_index])
    logger.debug(outputs)
    t_outputs = outputs > 0

    results = [labels[i] == labels[j] for i, j in zip(l_index, r_index)]
    results = torch.tensor(results, dtype=torch.bool, device=hidden.device)

    tp = torch.count_nonzero(torch.logical_and(t_outputs, results)).item()
    tn = torch.count_nonzero(torch.logical_and(~t_outputs, ~results)).item()
    fp = torch.count_nonzero(torch.logical_and(t_outputs, ~results)).item()
    fn = torch.count_nonzero(torch.logical_and(~t_outputs, results)).item()

    loss_func = torch.nn.BCEWithLogitsLoss()
    loss: torch.Tensor = loss_func(outputs, results.float())
    loss.backward()
    
    try:
        logger = logging.getLogger("verification")
        logger.debug("tp {} tn {} fp {} fn {}".format(tp, tn, fp, fn))
        logger.debug("ratio {}".format((tp + tn) / (tp + fp + fn + tn)))
        
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        logger.debug("f1    {}".format(2 / (1 / precision + 1 / recall)))
    except:
        pass

    return tp, tn, fp, fn
