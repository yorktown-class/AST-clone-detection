import torch
import math
import logging

logger = logging.getLogger("ast_attention")

class AttentionLayer(torch.nn.Module):

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(hidden_size, num_heads=2)
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.drop = torch.nn.Dropout(p=0.3)
    
    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        output, _ = self.attn.forward(input, input, input, attn_mask=mask)
        output = self.norm(output)
        return input + self.drop(output)


class FCLayer(torch.nn.Module):

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.w1 = torch.nn.Linear(hidden_size, hidden_size * 2)
        self.w2 = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.drop = torch.nn.Dropout(p=0.3)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.w1(input))
        output = self.w2(hidden)
        return input + self.drop(output)


class EncodeLayer(torch.nn.Module):

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attn = AttentionLayer(hidden_size)
        self.fc = FCLayer(hidden_size)
    
    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.fc(self.attn(input, mask))


class AstAttention(torch.nn.Module):
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dense = torch.nn.Linear(input_size, hidden_size)
        self.layers = torch.nn.ModuleList([
            EncodeLayer(hidden_size)
            for _ in range(num_layers)
        ])
        self.norm = torch.nn.LayerNorm(hidden_size)

    def forward(self, input: torch.Tensor, edges: torch.Tensor):
        N, _ = input.shape
        _, E = edges.shape

        assert(input.shape == (N, self.input_size))
        assert(edges.shape == (2, E))

        with torch.no_grad():
            mask = ~torch.eye(N, dtype=torch.bool, device=input.device)
            for _ in range(int(math.log(E, 2)) + 1):
                mask[edges[1, :]] &= mask[edges[0, :]]
        
        input = self.dense(input)
        for layer in self.layers:
            input = layer(input, mask)

        return self.norm(input)
