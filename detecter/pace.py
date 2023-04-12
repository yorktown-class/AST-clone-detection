import torch
import functools


@functools.lru_cache(maxsize=None)
def position_aware_char_embedding(word: str, device = "cpu") -> torch.Tensor:
    char_ord = torch.tensor([ord(c) for c in list(word)], dtype=torch.long, device=device)
    n = char_ord.shape[0]
    one_hot_mat = torch.zeros((n, 128), device=device)
    k = torch.arange(n, 0, -1, device=device) / n

    one_hot_mat[:, char_ord] = k
    embedding = torch.sum(one_hot_mat, dim=0, keepdim=False)
    return embedding
