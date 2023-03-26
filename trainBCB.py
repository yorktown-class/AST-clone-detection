import logging
from typing import *

import torch
import yaml
from torch.cuda.amp import GradScaler
from torch.utils import data
from tqdm import tqdm

from detecter.dataset import BigCloneBench
from detecter.model import AstAttention, Classifier
from detecter.train.BigCloneBench import Trainer, check_point, model_pt

TRAINER_CKPT_PATH = "log/trainerBCB.ckpt"
BEST_MODEL_PATH = "log/modelBCB.pt"
BEST_PMODEL_PATH = "log/model.pt"


def array_split(arr, n):
    k, m = divmod(len(arr), n)
    return (arr[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


if __name__ == "__main__":
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)

    stderr = logging.StreamHandler()
    stderr.setLevel(logging.INFO)
    logger.addHandler(stderr)

    with open("config-default.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    try:
        with open("config.yaml", "r") as f:
            new_cfg = yaml.safe_load(f)
        if new_cfg:
            cfg = {**cfg, **new_cfg}
    except IOError:
        pass

    ds = BigCloneBench.DataSet(
        "dataset/BigCloneBench/data.jsonl.txt",
        "dataset/BigCloneBench/test.txt",
        max_node_count=cfg["max_node_count"],
        fixed_prune=False,
    )

    indices_list = array_split(list(range(len(ds))), cfg["train_num_chunk"])
    subdataset_list = [data.Subset(ds, indices) for indices in indices_list]
    loaders = [
        data.DataLoader(
            subdataset,
            batch_size=cfg["train_batch_size"],
            collate_fn=BigCloneBench.collate_fn,
            num_workers=cfg["num_workers"],
            pin_memory=True,
        )
        for subdataset in subdataset_list
    ]

    ds = BigCloneBench.DataSet(
        "dataset/BigCloneBench/data.jsonl.txt",
        "dataset/BigCloneBench/valid.txt",
        max_node_count=cfg["max_node_count"],
        fixed_prune=True,
    )
    ds = data.Subset(ds, list(range(cfg["limit_valid_set_size"])))
    v_loader = data.DataLoader(
        ds,
        batch_size=cfg["valid_batch_size"],
        collate_fn=BigCloneBench.collate_fn,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    model = AstAttention(cfg["word_embedding_size"], cfg["hidden_size"], cfg["num_layers"], cfg["num_heads"]).cuda()
    classifier = Classifier(cfg["hidden_size"], 2).cuda()
    trainer = Trainer(model=model, classifier=classifier, evaluate_step_gap=cfg["evaluate_step_gap"]).cuda()

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.parameters(),
                "lr": cfg["model_lr"],
                "weight_decay": 0.1,
            },
            {
                "params": classifier.parameters(),
                "lr": cfg["classifier_lr"],
            },
        ]
    )

    scaler = GradScaler()

    try:
        save = torch.load(BEST_PMODEL_PATH)
        model.load_state_dict(save["model_state_dict"], strict=False)
        logger.info("load OJClone model")
    except IOError:
        pass
    try:
        save = torch.load(BEST_MODEL_PATH)
        model.load_state_dict(save["model_state_dict"], strict=False)
        classifier.load_state_dict(save["classifier_state_dict"], strict=False)
        min_loss = save["loss"]
        logger.info("load model")
    except IOError:
        logger.info("no model")
        min_loss = 1e8
    logger.info("min_loss {}".format(min_loss))
    try:
        save = torch.load(TRAINER_CKPT_PATH)
        trainer.load_state_dict(save["trainer_state_dict"], strict=True)
        optimizer.load_state_dict(save["optimizer_state_dict"])
        scaler.load_state_dict(save["scaler_state_dict"])
        epoch, trained_chunks = save["epoch"]
    except IOError:
        epoch, trained_chunks = 0, len(loaders)
        logger.info("no ckpt")
    logger.info("epoch {}".format(epoch))

    while True:
        if trained_chunks >= len(loaders):
            epoch += 1
            trained_chunks = 0
            logger.info("===========================")
            logger.info("epoch {}".format(epoch))

        trainer.train()
        loader = loaders[trained_chunks]
        for batch in tqdm(loader, desc="TRAIN"):
            optimizer.zero_grad()
            loss = trainer(batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        trainer.evaluate()
        trained_chunks += 1

        torch.save(check_point(trainer, optimizer, scaler, (epoch, trained_chunks)), TRAINER_CKPT_PATH)

        if (trained_chunks + len(loaders) * epoch) % cfg["valid_chunk_gap"] == 0:
            with torch.inference_mode():
                trainer.eval()
                for batch in tqdm(v_loader, desc="VALID"):
                    trainer(batch)
                loss = trainer.evaluate()

            if loss < min_loss:
                min_loss = loss
                torch.save(model_pt(model, classifier, loss), BEST_MODEL_PATH)
