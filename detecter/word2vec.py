import functools
from typing import *

import torch
from sentence_transformers import SentenceTransformer

from . import config


@functools.lru_cache()
def word2vec(word: str) -> torch.Tensor:
    sentence2emb = SentenceTransformer("all-MiniLM-L6-v2")
    return sentence2emb.encode(word, device="cuda", show_progress_bar=False, convert_to_tensor=True).cpu()


def create_word_dict(word_list: List[str]) -> Dict[str, torch.Tensor]:
    word_list = list(set(word_list))

    sentence2emb = SentenceTransformer("all-MiniLM-L6-v2")
    results: List[torch.Tensor] = sentence2emb.encode(
        word_list,
        show_progress_bar=True,
        batch_size=config.WORD2VEC_BATCH_SIZE,
        convert_to_numpy=False,
        device="cuda",
    )
    results = [result.cpu() for result in results]
    return dict(zip(word_list, results))
