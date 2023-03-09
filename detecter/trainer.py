from typing import *
import dataclasses
import torch
from torch.nn import Module
from torch.optim import Optimizer
import logging
from tqdm import tqdm
import pprint

from .model import AstAttention, Classifier, Similarity

logger = logging.getLogger("train")

class TrainerSimilarity(Module):
    def __init__(self, model: Similarity, classifier: Classifier):
        super().__init__()
        self.model = model
        self.classifier = classifier
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.info_list = []

    def log(self, info):
        logger = logging.getLogger("train" if self.training else "eval")
        logger.debug(pprint.pformat(info))

    def device(self) -> bool:
        return next(self.parameters()).device

    def print_aggr_log(self):
        if not self.info_list:
            return

        aggr = dict()
        keys = self.info_list[0].keys()
        for key in keys:
            aggr[key] = sum([item[key] for item in self.info_list])
        aggr["loss"] /= len(self.info_list)
        pprint(aggr)
        self.info_list = []

    def forward(self, batch1, batch2):
        labels1, input1, mask1 = batch1
        labels2, input2, mask2 = batch2
        
        input1 = input1.to(self.device())
        input2 = input2.to(self.device())
        mask1 = mask1.to(self.device())
        mask2 = mask2.to(self.device())


        hidden = self.model(input1, mask1, input2, mask2)
        output = self.classifier(hidden)

        result = [l1 == l2 for l1, l2 in zip(labels1, labels2)]
        result = torch.tensor(result, dtype=torch.long, device=output.device)
        
        loss: torch.Tensor = self.loss_fn(output, result)
        t_output = output[:, 1] > output[:, 0]

        tp = torch.count_nonzero(torch.logical_and(t_output, result)).item()
        tn = torch.count_nonzero(torch.logical_and(~t_output, ~result)).item()
        fp = torch.count_nonzero(torch.logical_and(t_output, ~result)).item()
        fn = torch.count_nonzero(torch.logical_and(~t_output, result)).item()
        try:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)

        self.info_list.append({
            "tp": tp, 
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall, 
            "f1": 2 / (1/(precision + 1e-8) + 1/(recall+1e-8)),
            "loss": loss.item()
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



# @dataclasses.dataclass
# class Trainer:
    model: AstAttention = AstAttention()
    classifier: Classifier = Classifier()
    similarity: Similarity = Similarity()
    optimizer: torch.optim.Optimizer
    iteration: int = 0

    def __post_init__(self):
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW([
                {"params": self.model.parameters(), "lr": 3e-5, "weight_decay": 0.1}, 
                {"params": self.classifier.parameters(), "lr": 3e-4}, 
                {"params": self.similarity.parameters(), "lr": 3e-4}, 
            ], amsgrad=True)

    def load_state_dict(self, dict: Dict):
        for field in dataclasses.fields(Trainer):
            key = field.name
            tp = field.type
            try:
                if tp != int:
                    getattr(self, key).load_state_dict(dict[key])
                else:
                    setattr(self, key, dict[key])
            except KeyError:
                print("can't load {}".format(key))

    def state_dict(self) -> Dict:
        return {
            "model": self.model.state_dict(),
            "downstream": self.downstream.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iteration": self.iteration,
        }

    def through_classifier(self, loader, train):
        loss_func = torch.nn.CrossEntropyLoss()
        totol_loss = 0
        totol_tc, totol_fc = 0, 0
        self.model.train(mode=train)
        self.classifier.train(mode=train)
        iteration = self.iteration if train else self.iteration - 1

        for labels, input, mask in tqdm(loader):
            if train:
                self.optimizer.zero_grad()
            
            output = self.model(input, mask)
            output = self.classifier(output)
            result = torch.tensor([int(l) - 1 for l in labels], dtype=torch.long, device=output.device)
            
            loss: torch.Tensor = loss_func(output, result)
            
            if train:
                loss.backward()
                self.optimizer.step()
            
            t_output = torch.argmax(output, dim=1)
            tc = torch.count_nonzero(t_output == result).item()
            fc = torch.count_nonzero(t_output != result).item()

            logger.debug("{} iteration {}".format("train" if train else "eval", iteration))
            logger.debug("true {} false {}".format(tc, fc))
            logger.debug("ratio       {}".format(tc / (tc + fc)))
            logger.debug("loss        {}".format(loss.item()))

            totol_loss += loss.item()
            totol_tc += tc 
            totol_fc += fc
        
        print("=====================:{}".format("train" if train else "eval"))
        print("iteration   {}".format(iteration))
        print("true {} false {}".format(totol_tc, totol_fc))
        print("ratio       {}".format(totol_tc / (totol_tc + totol_fc)))
        print("loss        {}".format(totol_loss / len(loader)))
        print("=====================")
        if train:
            self.iteration += 1


    def through_similarity(self, loader, train):
        loss_func = torch.nn.CrossEntropyLoss()
        totol_loss = 0
        counts = torch.zeros(4, dtype=torch.long)
        iteration = self.iteration if train else self.iteration - 1

        self.model.train(mode=train)
        self.classifier.train(mode=train)
        for labels, input, mask in tqdm(loader):
            if train:
                self.optimizer.zero_grad()
            n = len(labels)
            hidden = self.model(input, mask)

            l_index = [i for i in range(n) for j in range(i + 1, n)]
            r_index = [j for i in range(n) for j in range(i + 1, n)]
            
            output = self.similarity(hidden[:, l_index, :], hidden[:, r_index, :])

            result = [labels[i] == labels[j] for i, j in range(zip(l_index, r_index))]
            result = torch.tensor(result, dtype=torch.bool, device=output.device)
            
            loss: torch.Tensor = loss_func(output, result)
            
            if train:
                loss.backward()
                self.optimizer.step()
            
            t_output = output[:, 1] > output[:, 0]

            tp = torch.count_nonzero(torch.logical_and(t_output, result)).item()
            tn = torch.count_nonzero(torch.logical_and(~t_output, ~result)).item()
            fp = torch.count_nonzero(torch.logical_and(t_output, ~result)).item()
            fn = torch.count_nonzero(torch.logical_and(~t_output, result)).item()
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            
            logger.debug("{} iteration {}".format("train" if train else "eval", iteration))
            logger.debug("tp {} tn {} fp {} fn {}".format(tp, tn, fp, fn))
            logger.debug("ratio       {}".format((tp + tn) / (tp + tn + fp + fn)))
            try:
                logger.debug("precision   {}".format(precision))
                logger.debug("recall      {}".format(recall))
                logger.debug("f1          {}".format(2 / (1 / precision + 1/ recall)))
            except ZeroDivisionError:
                pass
            logger.debug("loss        {}".format(loss.item()))

            totol_loss += loss.item()
            counts += torch.tensor([tp, tn, fp, fn], dtype=torch.long)
        
        tp, tn, fp, fn = counts
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        print("=====================")
        print("{} iteration {}".format("train" if train else "eval", iteration))
        print("tp {} tn {} fp {} fn {}".format(tp, tn, fp, fn))
        print("ratio       {}".format((tp + tn) / (tp + tn + fp + fn)))
        try:
            print("precision   {}".format(precision))
            print("recall      {}".format(recall))
            print("f1          {}".format(2 / (1 / precision + 1/ recall)))
        except ZeroDivisionError:
            pass
        print("loss        {}".format(totol_loss / len(loader)))
        print("=====================")
        self.iteration += 1

    def validate_similarity(self, loader):
        loss_func = torch.nn.CrossEntropyLoss()
        totol_loss = 0
        counts = torch.zeros(4, dtype=torch.long)
        
        self.model.eval()
        self.classifier.eval()
        with torch.no_grad():
            for labels, input, mask in tqdm(loader):
                n = len(labels)
                hidden = self.model(input, mask)

                l_index = [i for i in range(n) for j in range(i + 1, n)]
                r_index = [j for i in range(n) for j in range(i + 1, n)]
                
                output = self.similarity(hidden[:, l_index, :], hidden[:, r_index, :])

                result = [labels[i] == labels[j] for i, j in range(zip(l_index, r_index))]
                result = torch.tensor(result, dtype=torch.bool, device=output.device)
                
                loss: torch.Tensor = loss_func(output, result)
                
                t_output = output[:, 1] > output[:, 0]

                tp = torch.count_nonzero(torch.logical_and(t_output, result)).item()
                tn = torch.count_nonzero(torch.logical_and(~t_output, ~result)).item()
                fp = torch.count_nonzero(torch.logical_and(t_output, ~result)).item()
                fn = torch.count_nonzero(torch.logical_and(~t_output, result)).item()
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                
                logger.debug("train iteration {}".format(self.iteration))
                logger.debug("tp {} tn {} fp {} fn {}".format(tp, tn, fp, fn))
                logger.debug("ratio       {}".format((tp + tn) / (tp + tn + fp + fn)))
                try:
                    logger.debug("precision   {}".format(precision))
                    logger.debug("recall      {}".format(recall))
                    logger.debug("f1          {}".format(2 / (1 / precision + 1/ recall)))
                except ZeroDivisionError:
                    pass
                logger.debug("loss        {}".format(loss.item()))

                totol_loss += loss.item()
                counts += torch.tensor([tp, tn, fp, fn], dtype=torch.long)
        
        tp, tn, fp, fn = counts
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        print("=====================")
        logger.debug("train iteration {}".format(self.iteration))
        logger.debug("tp {} tn {} fp {} fn {}".format(tp, tn, fp, fn))
        logger.debug("ratio       {}".format((tp + tn) / (tp + tn + fp + fn)))
        try:
            logger.debug("precision   {}".format(precision))
            logger.debug("recall      {}".format(recall))
            logger.debug("f1          {}".format(2 / (1 / precision + 1/ recall)))
        except ZeroDivisionError:
            pass
        logger.debug("loss        {}".format(totol_loss / len(loader)))
        print("=====================")
        self.iteration += 1