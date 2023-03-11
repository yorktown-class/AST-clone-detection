import torch
import logging

from . import logger

def compute_f1(output: torch.Tensor, target: torch.Tensor):
    t_output = torch.argmax(output, dim=-1).bool()
    target = target.bool()
    tp = torch.count_nonzero(torch.logical_and(t_output, target))
    tn = torch.count_nonzero(torch.logical_and(~t_output, ~target))
    fp = torch.count_nonzero(torch.logical_and(t_output, ~target))
    fn = torch.count_nonzero(torch.logical_and(~t_output, target))

    logger.debug("accuracy {}".format(((tp + tn) / (tp + tn + fp + fn)).item()))
    logger.debug("{} {} {} {}".format(tp.item(), tn.item(), fp.item(), fn.item()))

    try:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 / (1 / precision + 1 / recall)
        return f1
    except ZeroDivisionError:
        return torch.zeros(1).float().to(output.device)
        

def compute_acc(output: torch.Tensor, target: torch.Tensor):
    t_output = output.argmax(dim=-1)
    true_count = torch.count_nonzero(t_output == target)
    false_count = torch.count_nonzero(t_output != target)
    logger.debug("true count  {}".format(true_count))
    logger.debug("false count {}".format(false_count))
    
    return true_count / (true_count + false_count)


class Evaluator:
    def __init__(self, num_classes) -> None:
        self.num_classes = num_classes
        self.output_cat = None
        self.target_cat = None
        self.compute_func = compute_f1 if self.num_classes == 2 else compute_acc
        self.reset()

    def update(self, output: torch.Tensor, target: torch.Tensor):
        with torch.no_grad():
            self.output_cat = output if self.output_cat is None else torch.cat([self.output_cat, output], dim=0)
            self.target_cat = target if self.target_cat is None else torch.cat([self.target_cat, target], dim=0)
            evaluate = self.compute_func(output, target).item()
            logger.debug("curr eval {}".format(evaluate))
            logger.debug("aggr eval {}".format(self.compute().item()))

    def compute(self):
        assert(self.output_cat is not None)
        assert(self.target_cat is not None)
        with torch.no_grad():
            return self.compute_func(self.output_cat, self.target_cat)

    def reset(self):
        self.output_cat = None
        self.target_cat = None


