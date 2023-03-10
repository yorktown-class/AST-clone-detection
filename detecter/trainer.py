from typing import *
import dataclasses
import torch
from torch.nn import Module
from torch.optim import Optimizer
import logging
from tqdm import tqdm
import pprint
from torcheval.metrics import MulticlassF1Score

from .model import AstAttention, Classifier

logger = logging.getLogger("train")

class Trainer(Module):
    def __init__(self, model: AstAttention, classifier: Classifier, loss_fn, evaluator):
        super().__init__()
        self.model = model
        self.classifier = classifier
        self.loss_fn = loss_fn
        self.evaluator = evaluator
        self.info_list = []

    def log(self, info):
        logger = logging.getLogger("train" if self.training else "eval")
        logger.debug(pprint.pformat(info))

    def device(self) -> bool:
        return next(self.parameters()).device

    def print_aggr_log(self):
        if not self.info_list:
            return

        loss = sum(item["loss"] for item in self.info_list) / len(self.info_list)
        evaluate = self.evaluator.compute()

        pprint({
            "loss": loss, 
            "evaluate": evaluate,
        })
        self.evaluator.reset()
        self.info_list = []

    def forward(self, batch):
        labels, input, mask = batch
        
        device = self.device()
        labels = labels.to(device)
        input = input.to(device)
        mask = mask.to(device)

        hidden = self.model(input, mask)
        output = self.classifier(hidden)

        loss: torch.Tensor = self.loss_fn(output, labels)
        self.evaluator.update(output, labels)

        self.info_list.append({
            "loss": loss.item(),
            "eval": self.evaluator.compute().item(),
        })
        self.log(self.info_list[-1])

        return loss


def check_point(model, optimizer, epoch, loss) -> Dict:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }

