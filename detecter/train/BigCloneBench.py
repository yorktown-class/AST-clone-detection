from typing import *

import torch
from torch.cuda.amp import autocast
from torcheval.metrics import MulticlassF1Score

from .. import logger
from ..model import AstAttention, Classifier


class Trainer(torch.nn.Module):
    def __init__(self, model: AstAttention, classifier: Classifier):
        super().__init__()
        self.model: AstAttention = model
        self.classifier: Classifier = classifier
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.evaluator = MulticlassF1Score(num_classes=2)
        self.evaluator_all = MulticlassF1Score(num_classes=2)
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
        self.evaluator.to(device)
        self.evaluator_all.to(device)

        hidden = self.model(nodes, mask)[0]
        score = self.classifier(hidden)
        loss = self.loss_fn(score, label.long())
        self.evaluator.update(score, label.long())
        self.evaluator_all.update(score, label.long())

        logger.debug("f1   {}".format(self.evaluator.compute().item()))
        self.evaluator.reset()
        logger.debug("loss {}".format(loss.item()))
        self.loss_list.append(loss.item())

        return loss

    def evaluate(self) -> float:
        f1 = self.evaluator_all.compute()
        self.evaluator_all.reset()
        logger.info("aggr evaluate {}".format(f1))
        loss = sum(self.loss_list) / len(self.loss_list)
        logger.info("aggr loss     {}".format(loss))
        self.loss_list = []
        return loss


def check_point(trainer: Trainer, optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler, epoch) -> Dict:
    return {
        "trainer_state_dict": trainer.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
    }


def model_pt(model: torch.nn.Module, classifier: Classifier, loss) -> Dict:
    return {
        "model_state_dict": model.state_dict(),
        "classifier_state_dict": classifier.state_dict(),
        "loss": float(loss),
    }
