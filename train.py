import logging
import multiprocessing
import os
from typing import *

import torch
from torch.utils import data
from tqdm import tqdm

from detecter.dataset import OJClone
from detecter.model import AstAttention, Classifier
from detecter.train import Trainer, check_point, model_pt
from detecter.word2vec import word2vec

TRAINER_CKPT_PATH = "log/trainer.ckpt"
BEST_MODEL_PATH = "log/model.pt"


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)

    stderr = logging.StreamHandler()
    stderr.setLevel(logging.INFO)
    logger.addHandler(stderr)

    multiprocessing.set_start_method("spawn")

    word2vec("1")

    batch_size = 1
    ds = OJClone.BiDataSet("dataset/OJClone/train.jsonl", max_node_count=512)
    loader = data.DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=OJClone.collate_fn,
        shuffle=True,
        num_workers=2,
    )
    ds = OJClone.BiDataSet("dataset/OJClone/valid.jsonl", max_node_count=512)
    v_loader = data.DataLoader(ds, batch_size=batch_size, collate_fn=OJClone.collate_fn, num_workers=2)

    model = AstAttention(384, 768, num_layers=6, num_heads=8).cuda()
    classifier = Classifier(768, 2).cuda()
    trainer = Trainer(model=model, classifier=classifier).cuda()

    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": 3e-5, "weight_decay": 0.1},
            {"params": classifier.parameters(), "lr": 3e-4},
        ]
    )

    try:
        with open(BEST_MODEL_PATH, "rb") as f:
            save = torch.load(f)
        model.load_state_dict(save["model_state_dict"], strict=True)
        classifier.load_state_dict(save["classifier_state_dict"], strict=True)
        min_loss = save["loss"]
    except IOError:
        logger.info("no model")
        min_loss = 1e8
    logger.info("min_loss {}".format(min_loss))
    try:
        save = torch.load(TRAINER_CKPT_PATH)
        trainer.load_state_dict(save["trainer_state_dict"], strict=True)
        optimizer.load_state_dict(save["optimizer_state_dict"])
        epoch = save["epoch"]
    except IOError:
        epoch = 1
        logger.info("no ckpt")
    logger.info("epoch {}".format(epoch))

    while True:
        logger.info("epoch {}".format(epoch))
        trainer.train()
        for batch in tqdm(loader):
            optimizer.zero_grad()
            loss = trainer(batch)
            loss.backward()
            optimizer.step()
        trainer.evaluate()

        trainer.eval()
        for batch in tqdm(v_loader):
            with torch.no_grad():
                trainer(batch)
        loss = trainer.evaluate()

        torch.save(check_point(trainer, optimizer, epoch), TRAINER_CKPT_PATH)
        if loss < min_loss:
            min_loss = loss
            torch.save(model_pt(model, classifier, loss), BEST_MODEL_PATH)

        epoch += 1
