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
import math

# from .collate import collate_fn
from ..parser import code2tree
from ..parser.code2tree import Tree
from detecter import config


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

    with multiprocessing.Pool(processes=config.NUM_CORE) as p:
        iresult = p.imap_unordered(convert_impl, data_list, 20)
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
        batch_size=config.WORD2VEC_BATCH_SIZE,
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
    CHUNK_SIZE = config.CHUNK_SIZE
    def __init__(self, data_path, item_count=None) -> None:
        super().__init__()
        self.data_path = data_path
        self.chunk_path = dict() # int -> str
        self.length = 0

        with open(data_path, "r") as f:
            lines = f.readlines()
        raw_data_list = [json.loads(line) for line in lines]
        if item_count:
            raw_data_list = raw_data_list[:min(item_count, len(raw_data_list))]
        
        self.length = len(raw_data_list)

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
        
        # self.idx_map = list(range(len(raw_data_list)))
        
        # n = len(self.idx_map)
        # for i in range(0, n, self.CHUNK_SIZE * 2):
        #     shuffle_slice(self.idx_map, i, min(n, i + self.CHUNK_SIZE * 2))

    @functools.lru_cache(maxsize=config.LRU_CACHE_SIZE)
    def get_chunk(self, idx):
        path = self.chunk_path[idx]
        logger = logging.getLogger("dataset")
        logger.debug("debug chunk {} -> {}".format(idx, path))
        with open(path, "rb") as f:
            return torch.load(f)

    def __getitem__(self, index) -> Tuple[str, Tree]:
        # index = self.idx_map[index]
        
        idx = index // self.CHUNK_SIZE
        chunk = self.get_chunk(idx)

        offset = min(index - idx * self.CHUNK_SIZE, len(chunk) - 1)

        return chunk[offset]

    def __len__(self) -> int:
        return self.length



class SubRandomSampler(data.Sampler):
    def __init__(self, data_source: DataSet) -> None:
        self.data_source = data_source

    def __iter__(self):
        logger = logging.getLogger("iter")

        n = len(self.data_source)
        m = self.data_source.CHUNK_SIZE
        idx = list(range(n // m))
        random.shuffle(idx)
        index = [i * m + j for i in idx for j in range(m)]
        for i in range(0, len(index) - 2 * m + 1, m):
            shuffle_slice(index, i, i + 2 * m)
        
        logger.debug("index {}".format(index))
        return iter(index)
    
    def __len__(self):
        n = len(self.data_source)
        m = self.data_source.CHUNK_SIZE
        return n // m * m



def collate_fn(batch: List[Union[str, Tree]]):
    max_n = 0
    label_list = []
    node_list = []
    mask_list = []
    for label, tree in batch:
        label_list.append(label)
        node: torch.Tensor = tree.x
        edge: torch.Tensor = tree.edge_index
        
        n, _ = node.shape
        _, m = edge.shape

        mask = ~torch.eye(n, dtype=torch.bool)
        for _ in range(int(math.log(m, 2))):
            mask[edge[1, :]] &= mask[edge[0, :]]
        
        max_n = max(n, max_n)
        node_list.append(node)
        mask_list.append(mask)
    
    node_batch = torch.nn.utils.rnn.pad_sequence(node_list)
    
    mask_base = ~torch.eye(max_n, dtype=torch.bool)
    mask_batch = torch.broadcast_to(mask_base, (len(batch), max_n, max_n))

    for idx, mask in enumerate(mask_list):
        n, m = mask.shape
        mask_batch[idx, :n, :m] = mask
    return label_list, node_batch, mask_batch


def DataLoader(data_path, batch_size, item_count=None):
    ds = DataSet(data_path, item_count)
    print(len(ds))
    print(config.NUM_CORE)
    # return gDataLoader(ds, batch_size, shuffle=False, follow_batch=["x"], drop_last=True, num_workers=config.NUM_CORE, sampler=SubRandomSampler(ds), collate_fn=collate_fn)

    return data.DataLoader(ds, 
                           batch_size=batch_size,
                           sampler=SubRandomSampler(ds),
                           collate_fn=collate_fn)


# loss_func = torch.nn.MSELoss()
loss_func = torch.nn.CrossEntropyLoss()

def verification(labels: List[str], hidden: torch.Tensor, similarity: torch.nn.Module):
    logger = logging.getLogger("verification")

    n, h = hidden.shape

    l_index = [i for i in range(n) for j in range(i + 1, n)]
    r_index = [j for i in range(n) for j in range(i + 1, n)]

    # logger.debug(hidden)
    outputs: torch.Tensor = similarity(hidden[l_index], hidden[r_index])
    # logger.debug(outputs)
    t_outputs = outputs[:, 1] > outputs[:, 0]

    results = [labels[i] == labels[j] for i, j in zip(l_index, r_index)]
    results = torch.tensor(results, dtype=torch.bool, device=hidden.device)

    tp = torch.count_nonzero(torch.logical_and(t_outputs, results)).item()
    tn = torch.count_nonzero(torch.logical_and(~t_outputs, ~results)).item()
    fp = torch.count_nonzero(torch.logical_and(t_outputs, ~results)).item()
    fn = torch.count_nonzero(torch.logical_and(~t_outputs, results)).item()

    # loss_func = torch.nn.BCEWithLogitsLoss()
    # loss: torch.Tensor = loss_func(outputs, results.float())
    # loss.backward()

    # loss: torch.Tensor = loss_func(torch.relu(outputs), torch.where(results, 1, 0).float())
    # loss.backward()
    loss = loss_func(outputs, results.long())
    loss.backward()

    try:
        logger = logging.getLogger("verification")
        logger.debug("loss  {}".format(loss.item()))
        logger.debug("tp {} tn {} fp {} fn {}".format(tp, tn, fp, fn))
        logger.debug("ratio {}".format((tp + tn) / (tp + fp + fn + tn)))
        
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        logger.debug("f1    {}".format(2 / (1 / precision + 1 / recall)))
    except:
        pass

    return tp, tn, fp, fn, loss