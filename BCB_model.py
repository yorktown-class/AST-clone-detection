from typing import *
import torch
from torcheval import metrics


class Detecter(torch.nn.Module):
    def __init__(self, encoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.linear = torch.nn.Linear(128, 1)

    def forward(self, nodes: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(nodes, dist)
        score = self.linear(hidden[0::2] * hidden[1::2]).reshape(-1)
        return score


class Trainer(torch.nn.Module):
    def __init__(self, model: Detecter, device = "cuda") -> None:
        super().__init__()
        self.device = device

        self.model = model

        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        self.loss_list = []
        self.evaluators: Dict[str, metrics.Metric] = {
            "f1": metrics.BinaryF1Score(device=device),
            "precision": metrics.BinaryPrecision(device=device),
            "recall": metrics.BinaryRecall(device=device),
            "accuracy": metrics.BinaryAccuracy(device=device),
        }

        self.to(device=device)

    def evaluate(self) -> Dict:
        result = dict()
        result["loss"] = sum(self.loss_list) / len(self.loss_list)
        for key, evaluator in self.evaluators.items():
            result[key] = evaluator.compute()
        return result
    
    def reset(self) -> None:
        self.loss_list = []
        for evaluator in self.evaluators.values():
            evaluator.reset()
    
    def forward(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        label, nodes, dist = [item.to(self.device) for item in batch]
        result: torch.Tensor = self.model(nodes, dist)

        loss = self.loss_fn(result, label.float())

        self.loss_list.append(loss.item())
        for evaluator in self.evaluators.values():
            evaluator.update(result, label.long())
        
        return loss
