from typing import *

import torch
import functools
from sentence_transformers import SentenceTransformer

from . import config

@functools.lru_cache()
def word2vec(word: str):
    sentence2emb = SentenceTransformer('all-MiniLM-L6-v2')
    return sentence2emb.encode(word, device="cuda", show_progress_bar=False, convert_to_tensor=True).cpu()


def create_word_dict(word_list: List[str]) -> Dict[str, torch.Tensor]:
    word_list = list(set(word_list))
    
    sentence2emb = SentenceTransformer('all-MiniLM-L6-v2')
    result = sentence2emb.encode(word_list, 
                                 batch_size=config.WORD2VEC_BATCH_SIZE, 
                                 device="cuda")
    return dict(zip(word_list, result))
