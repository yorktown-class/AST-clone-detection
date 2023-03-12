import torch


"""
PE(pos, 2i) = sin(pos / 10000 ^ {2i / d})
PE(pos, 2i+1) = cos(pos / 10000 ^ {2i / d})

10000 ^ {- 2i / d} = e^log_e{10000 ^ {-2i/d}} = e^{ -2i/d * log_e{10000} }
"""

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, channels, max_length, dropout=0.5):
        super().__init__()
        self.channels = channels
        self.max_length = max_length

        self.drop = torch.nn.Dropout(p=dropout)

        i_pos = torch.arange(0, channels, step=2)
        loge_10000 = torch.log(torch.tensor(10000, dtype=torch.float))
        div = torch.exp( - i_pos / channels * loge_10000 )

        self.register_buffer("div", div, persistent=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        F = x.shape[-1]
        assert(N < self.max_length)
        assert(F == self.channels)

        with torch.no_grad():
            pe = torch.zeros((N, self.channels), dtype=torch.float, device=x.device)
            position = torch.arange(0, N).reshape(-1, 1).to(x.device)
            pe[:, 0::2] = torch.sin(position * self.div.reshape(1, -1))
            pe[:, 1::2] = torch.cos(position * self.div.reshape(1, -1))

        if x.dim() == 3:
            pe = pe.unsqueeze(1)
        return x + self.drop(pe)
