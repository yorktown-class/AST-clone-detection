import logging
import random
from typing import *

import torch
from torch.cuda.amp import GradScaler, autocast

from .. import logger, tree_tools
from ..dataset.OJClone import DataSet
from ..model import AstAttention, Classifier


class BatchSampler:
    def __init__(self, data_source: DataSet, batch_size: int, shuffle: bool) -> None:
        self.data_source = data_source
        assert batch_size % 2 == 0
        assert batch_size >= 4
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        label_cluster = dict()
        for i in range(len(self.data_source)):
            label = self.data_source.raw_data_list[i]["label"]
            label_cluster[label] = label_cluster.get(label, []) + [i]

        cluster_list = [cluster for cluster in label_cluster.values() if len(cluster) >= 2]

        n_cluster = self.batch_size // 2

        if n_cluster > len(cluster_list):
            raise ValueError("batch size too big")

        if self.shuffle:
            for cluster in cluster_list:
                random.shuffle(cluster)
            random.shuffle(cluster_list)

        while len(cluster_list) >= n_cluster:
            batch_list = []
            for cluster in cluster_list[:n_cluster]:
                batch_list.append(list.pop(cluster))
                batch_list.append(list.pop(cluster))
            if self.shuffle:
                random.shuffle(cluster_list)
            else:
                cluster_list = cluster_list[1:] + cluster_list[:1]
            cluster_list = list(filter(lambda cluster: len(cluster) >= 2, cluster_list))
            yield batch_list

    def __len__(self) -> int:
        return len(self.data_source) // self.batch_size


def collate_fn(batch: List[Tuple[int, torch.Tensor, torch.Tensor]]):
    label_list = [label for label, nodes, mask in batch]
    tree_tensor_list = [(nodes, mask) for label, nodes, mask in batch]
    label_batch = torch.tensor(label_list, dtype=torch.long)
    return label_batch, *tree_tools.collate_tree_tensor(tree_tensor_list)


class Evaluator:
    def __init__(self) -> None:
        self.reset()

    @torch.no_grad()
    def update(self, score: torch.Tensor):
        MAP = (score.argmax(dim=-1) == 0).float()
        logger.debug("eval map {}".format(MAP.mean().item()))
        self.map_sum += MAP.sum().item()
        self.map_count += MAP.shape[0]

    def reset(self):
        self.map_sum = 0
        self.map_count = 0

    def compute(self):
        try:
            return self.map_sum / self.map_count
        except ZeroDivisionError:
            return 0


class Trainer(torch.nn.Module):
    def __init__(self, model: AstAttention):
        super().__init__()
        self.model: AstAttention = model
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.evaluator = Evaluator()
        self.loss_list = []

    def device(self) -> bool:
        return next(self.parameters()).device

    @autocast(enabled=True)
    def forward(self, batch: Union[torch.Tensor, torch.Tensor, torch.Tensor]):
        label, input, mask = batch

        device = self.device()
        label = label.to(device)
        input = input.to(device)
        mask = mask.to(device)

        hidden = self.model(input, mask)[0]
        positive_feature = torch.zeros_like(hidden)
        positive_feature[0::2] = hidden[1::2]
        positive_feature[1::2] = hidden[0::2]

        # positive_score = torch.cosine_similarity(hidden, positive_feature, dim=-1)
        # negtive_score = torch.cosine_similarity(hidden.unsqueeze(1), hidden.unsqueeze(0), dim=-1)
        positive_score = torch.sum(hidden * positive_feature, dim=-1)  # [N]
        negtive_score = torch.matmul(hidden, hidden.T)  # [N, 2N]

        is_positive = label[:, None] == label[None, :]
        negtive_score = negtive_score[~is_positive].reshape(hidden.shape[0], -1)

        score = torch.cat([positive_score[:, None], negtive_score], dim=-1)
        score = torch.softmax(score, dim=-1)
        self.evaluator.update(score)
        loss = -torch.log(score[:, 0] + 1e-9)
        loss = loss.mean()

        logger.debug("loss {}".format(loss))
        self.loss_list.append(loss.item())

        return loss

    def evaluate(self) -> float:
        logger.info("aggr evaluate {}".format(self.evaluator.compute()))
        self.evaluator.reset()
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


def model_pt(model: torch.nn.Module, loss) -> Dict:
    return {
        "model_state_dict": model.state_dict(),
        "loss": float(loss),
    }