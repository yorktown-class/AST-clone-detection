from typing import *

import torch
from torch.cuda.amp import autocast

from .. import logger, tree_tools
from ..dataset.OJClone import DataSet
from ..model import AstAttention, Similarity


class Evaluator:
    pass


class Trainer(torch.nn.Module):
    def __init__(self, model: AstAttention, similarity: Similarity):
        super().__init__()
        self.model: AstAttention = model
        self.similarity: Similarity = similarity
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        self.acc_list = []
        self.loss_list = []

    def device(self) -> bool:
        return next(self.parameters()).device

    @autocast(enabled=True)
    def forward(self, batch: Union[torch.Tensor, torch.Tensor, torch.Tensor]):
        label, nodes, mask = batch

        device = self.device()
        label = label.to(device)
        nodes = nodes.to(device)
        mask = mask.to(device)

        hidden = self.model(nodes, mask)[0]
        lhs = hidden[0::2]
        rhs = hidden[1::2]

        score = self.similarity(lhs, rhs)
        loss = self.loss_fn(score, label.float())

        acc = torch.count_nonzero((score > 0) == label).item()
        acc_ratio = acc / label.shape[0]

        logger.debug("acc  {}".format(acc_ratio))
        logger.debug("loss {}".format(loss.item()))
        self.acc_list.append(acc_ratio)
        self.loss_list.append(loss.item())

        return loss

    def evaluate(self) -> float:
        acc = sum(self.acc_list) / len(self.acc_list)
        logger.info("aggr evaluate {}".format(acc))
        loss = sum(self.loss_list) / len(self.loss_list)
        logger.info("aggr loss     {}".format(loss))
        self.acc_list = []
        self.loss_list = []
        return loss


def check_point(trainer: Trainer, optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler, epoch) -> Dict:
    return {
        "trainer_state_dict": trainer.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
    }


def model_pt(model: torch.nn.Module, similarity: Similarity, loss) -> Dict:
    return {
        "model_state_dict": model.state_dict(),
        "similarity_state_dict": similarity.state_dict(),
        "loss": float(loss),
    }
