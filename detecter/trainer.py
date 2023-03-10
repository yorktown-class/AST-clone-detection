from typing import *
import dataclasses
import torch
from torch.nn import Module
from torch.optim import Optimizer
import logging
from tqdm import tqdm
import pprint
# from torcheval.metrics import MulticlassF1Score

from .model import AstAttention, Classifier

logger = logging.getLogger("train")

class Evaluator:
    def __init__(self, num_classes) -> None:
        self.num_classes = num_classes
        self.reset()

    def update(self, output: torch.Tensor, target: torch.Tensor):
        # logger.debug("output {}".format(output))
        # logger.debug("target {}".format(target))
        with torch.no_grad():
            t_output = output.argmax(dim=1)
            true_count = torch.count_nonzero(t_output == target)
            false_count = torch.count_nonzero(t_output != target)
            self.true_count += true_count.item()
            self.false_count += false_count.item()

            if self.num_classes == 2:
                t_output = t_output.bool()
                target = target.bool()
                tp = torch.count_nonzero(torch.logical_and(t_output, target))
                tn = torch.count_nonzero(torch.logical_and(~t_output, ~target))
                fp = torch.count_nonzero(torch.logical_and(t_output, ~target))
                fn = torch.count_nonzero(torch.logical_and(~t_output, target))
                self.tp += tp.item()
                self.tn += tn.item()
                self.fp += fp.item()
                self.fn += fn.item()
    
    def compute(self):
        if self.num_classes == 2:
            logger.debug("acc {}".format((self.true_count / (self.true_count + self.false_count))))
            logger.debug("{} {} {} {}".format(self.tp, self.tn, self.fp, self.fn))
            try:
                precision = self.tp / (self.tp + self.fp)
                recall = self.tp / (self.tp + self.fn)
                f1 = 2 / (1/precision + 1/recall)
                return f1
            except ZeroDivisionError:
                return torch.tensor(0)
        else:
            return self.true_count / (self.true_count + self.false_count)

    def reset(self):
        self.true_count, self.false_count = 0, 0
        self.tp, self.tn, self.fp, self.fn = 0, 0, 0, 0



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

        pprint.pprint({
            "loss": loss, 
            "evaluate": float(self.evaluator.compute()) if self.evaluator else None,
        })

        if self.evaluator:
            self.evaluator.reset()

        self.info_list = []

    def forward2(self, batch):
        labels, input, mask = batch
        device = self.device()
        labels = labels.to(device)
        input = input.to(device)
        mask = mask.to(device)
        n = labels.shape[0]
        hidden = self.model(input, mask)
        l_index = [i for i in range(n) for j in range(i + 1, n)]
        r_index = [j for i in range(n) for j in range(i + 1, n)]

        output = self.classifier(hidden[:, l_index, :], hidden[:, r_index, :])

        result = torch.eq(labels[l_index], labels[r_index]).long()

        loss = self.loss_fn(output, result)
        self.evaluator.update(output, result)

        self.info_list.append({
            "loss": loss.item(),
            "eval": float(self.evaluator.compute())
        })
        self.log(self.info_list[-1])
        return loss


    def forward(self, batch):
        labels, input, mask = batch
        
        device = self.device()
        labels = labels.to(device)
        input = input.to(device)
        mask = mask.to(device)

        hidden = self.model(input, mask)
        output = self.classifier(hidden)

        loss: torch.Tensor = self.loss_fn(output, labels)
        if self.evaluator:
            self.evaluator.update(output, labels)

        self.info_list.append({
            "loss": loss.item(),
            "eval": float(self.evaluator.compute()) if self.evaluator else None,
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

