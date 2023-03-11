import torch
import logging

from .. import logger

class AttentionLayer(torch.nn.Module):

    def __init__(self, hidden_size: int, num_heads: int = 1, dropout: float = 0.5) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.attn = torch.nn.MultiheadAttention(hidden_size, num_heads=num_heads)
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.drop = torch.nn.Dropout(p=dropout)
    
    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L, _ = mask.shape
        assert(mask.shape == (B, L, L))
        assert(input.shape == (L, B, self.hidden_size))

        output, _ = self.attn(
            input, input, input, attn_mask=mask.repeat_interleave(self.num_heads, dim=0))
        output = self.norm(output)
        return input + self.drop(output)


class FCLayer(torch.nn.Module):

    def __init__(self, hidden_size: int, dropout: float=0.5) -> None:
        super().__init__()
        self.w1 = torch.nn.Linear(hidden_size, hidden_size * 2)
        self.w2 = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.drop = torch.nn.Dropout(p=dropout)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.w1(input))
        output = self.w2(hidden)
        output = self.norm(output)
        return input + self.drop(output)


class EncodeLayer(torch.nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, dropout: float=0.5) -> None:
        super().__init__()
        self.attn = AttentionLayer(hidden_size, num_heads, dropout)
        self.fc = FCLayer(hidden_size, dropout)
    
    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.fc(self.attn(input, mask))


class AstAttention(torch.nn.Module):
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_heads: int = 1, dropout: float = 0.5) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dense = torch.nn.Linear(input_size, hidden_size)
        self.layers = torch.nn.ModuleList([
            EncodeLayer(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = torch.nn.LayerNorm(hidden_size)

    def forward(self, input: torch.Tensor, mask: torch.Tensor):
        reshape = False
        if input.dim() == 2:
            reshape = True
            N, F = input.shape
            input = input.reshape(N, 1, F)
            mask = mask.reshape(1, *mask.shape)

        N, B, F = input.shape
        assert(F == self.input_size)
        assert(mask.shape == (B, N, N))

        hidden = self.dense(input)
        for layer in self.layers:
            hidden = layer(hidden, mask)

        output = self.norm(hidden)
        
        if reshape:
            output = output.reshape(N, -1)
        return output