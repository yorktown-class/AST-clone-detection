import functools

import torch

emb = torch.eye(128, dtype=torch.float)


@functools.lru_cache(maxsize=None)
def position_aware_char_embedding(word: str, device="cpu") -> torch.Tensor:
    char_ord = torch.tensor([ord(c) for c in list(word)], dtype=torch.long, device=device)
    n = char_ord.shape[0]
    k = torch.arange(n, 0, -1, device=device, dtype=torch.float) / n
    embedding = torch.sum(k.unsqueeze(1) * emb[char_ord, :], dim=0, keepdim=False)
    return embedding
