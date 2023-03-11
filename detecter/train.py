from typing import *

import torch
import logging

from .model import AstAttention, Classifier
from .evaluator import Evaluator
from . import logger

class Trainer(torch.nn.Module):
    def __init__(self, model: AstAttention, classifier: Classifier):
        super().__init__()
        self.model: AstAttention = model
        self.classifier: Classifier = classifier
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.evaluator = Evaluator(num_classes=2)
        self.loss_list = []

    def device(self) -> bool:
        return next(self.parameters()).device

    def forward(self, batch: Union[torch.Tensor, torch.Tensor, torch.Tensor]):
        labels, input, mask = batch
        
        device = self.device()
        labels = labels.to(device)
        input = input.to(device)
        mask = mask.to(device)

        hidden = self.model(input, mask)
        output = self.classifier(hidden)

        loss: torch.Tensor = self.loss_fn(output, labels)
        self.evaluator.update(output, labels)

        logger.debug("loss {}".format(loss))
        self.loss_list.append(loss.item())

        return loss

    def evaluate(self) -> float:
        logger.info("evaluate {}".format(self.evaluator.compute().item()))
        self.evaluator.reset()
        loss = sum(self.loss_list) / len(self.loss_list)
        logger.info("loss     {}".format(loss))
        self.loss_list = []
        return loss


def check_point(trainer: Trainer, optimizer: torch.optim.Optimizer, epoch) -> Dict:
    return {
        "trainer_state_dict": trainer.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }


def model_pt(model: torch.nn.Module, classifier: torch.nn.Module, loss) -> Dict:
    return {
        "model_state_dict": model.state_dict(),
        "classifier_state_dict": classifier.state_dict(),
        "loss": float(loss),
    }
