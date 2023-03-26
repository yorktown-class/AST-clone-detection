from typing import *

import torch
from torch.cuda.amp import autocast
from torcheval.metrics import MulticlassF1Score

from .. import logger
from ..model import AstAttention, Classifier


class Trainer(torch.nn.Module):
    def __init__(self, model: AstAttention, classifier: Classifier, evaluate_step_gap: int):
        super().__init__()
        self.model: AstAttention = model
        self.classifier: Classifier = classifier
        self.evaluate_step_gap = evaluate_step_gap

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.half_evaluator = MulticlassF1Score(num_classes=2)
        self.final_evaluator = MulticlassF1Score(num_classes=2)
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
        self.half_evaluator.to(device)
        self.final_evaluator.to(device)

        hidden = self.model(nodes, mask)[0]
        score = self.classifier(hidden)
        loss = self.loss_fn(score, label.long())
        self.half_evaluator.update(score, label.long())
        self.final_evaluator.update(score, label.long())

        self.loss_list.append(loss.item())

        if len(self.loss_list) % self.evaluate_step_gap == 0:
            logger.debug("f1   {}".format(self.half_evaluator.compute().item()))
            self.half_evaluator.reset()
            logger.debug("loss {}".format(sum(self.loss_list[-self.evaluate_step_gap :]) / self.evaluate_step_gap))

        return loss

    def evaluate(self) -> float:
        logger.info("aggr evaluate {}".format(self.final_evaluator.compute().item()))
        loss = sum(self.loss_list) / len(self.loss_list)
        logger.info("aggr loss     {}".format(loss))

        self.loss_list = []
        self.final_evaluator.reset()
        self.half_evaluator.reset()
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
