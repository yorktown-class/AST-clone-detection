from typing import *

import torch
from torch.utils import data
import json
import functools
import multiprocessing
from tqdm import tqdm
import math
import itertools

from .. import config
from ..parser import code2tree


# @functools.lru_cache(maxsize=None)
# def word2vec(word: str):
#     from sentence_transformers import SentenceTransformer

#     sentence2emb = SentenceTransformer('all-MiniLM-L6-v2')
#     return sentence2emb.encode(word, device="cuda", show_progress_bar=False)


def create_word_dict(word_list: List[str]) -> Dict[str, torch.Tensor]:
    word_list = list(set(word_list))
    
    from sentence_transformers import SentenceTransformer

    sentence2emb = SentenceTransformer('all-MiniLM-L6-v2')
    result = sentence2emb.encode(word_list, 
                                 batch_size=config.WORD2VEC_BATCH_SIZE, 
                                 device="cuda")
    return dict(zip(word_list, result))



def convert(code: Dict) -> Tuple[List[str], Tuple[List[int], List[int]]]:
    code_string = code["code"]
    index = code["index"]
    try:
        V, E = code2tree.parse(code_string, "c")
        return (V, E)
    except code2tree.ParseError as err:
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
    def __init__(self, data_path, item_count=None) -> None:
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
            data_list = [(label, V, E) for label, (V, E) in zip(label_list, code_tree_list)]
            self.data_list = [(l, v, e) for l, v, e in data_list if len(v) < config.MAX_NODE_COUNT]
            
            nodes_list = [v for l, v, e in self.data_list]
            self.word_dict = create_word_dict(list(itertools.chain(*nodes_list)))

            with open(save_path, "wb") as f:
                torch.save({
                    "data_list": self.data_list, 
                    "word_dict": self.word_dict,
                }, f)

    def __getitem__(self, index) -> Tuple[str, torch.Tensor, torch.Tensor]:
        import numpy
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
        
        return label, tensorV, mask

    def __len__(self) -> int:
        return len(self.data_list)


def collate_fn(batch: List[Union[str, torch.Tensor, torch.Tensor]]):
    label_list = [label for label, nodes, mask in batch]
    nodes_list = [nodes for label, nodes, mask in batch]
    mask_list = [mask for label, nodes, mask in batch]

    node_batch = torch.nn.utils.rnn.pad_sequence(nodes_list)

    n = node_batch.shape[0]

    mask_base = ~torch.eye(n, dtype=torch.bool)
    mask_batch = mask_base.repeat(len(batch), 1, 1)
    # print(mask_batch.shape)
    # mask_batch = torch.broadcast_to(mask_base, (len(batch), n, n))

    for idx, mask in enumerate(mask_list):
        n, m = mask.shape
        mask_batch[idx, :n, :m] = mask
    
    return label_list, node_batch, mask_batch


def getDataLoader(dataset: DataSet, **kwargs):
    return data.DataLoader(dataset, num_workers=2, collate_fn=collate_fn, **kwargs)
