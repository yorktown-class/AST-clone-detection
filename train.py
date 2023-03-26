import logging
from typing import *

import torch
from torch.cuda.amp import GradScaler
from torch.utils import data
from tqdm import tqdm

from detecter.dataset import OJClone
from detecter.model import AstAttention, Classifier
from detecter.train.OJClone import (
    BatchSampler,
    Trainer,
    check_point,
    collate_fn,
    model_pt,
)
from detecter.word2vec import word2vec

TRAINER_CKPT_PATH = "log/trainer.ckpt"
BEST_MODEL_PATH = "log/model.pt"


if __name__ == "__main__":
    # os.environ["TOKENIZERS_PARALLELISM"] = "true"
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)

    stderr = logging.StreamHandler()
    stderr.setLevel(logging.INFO)
    logger.addHandler(stderr)

    # multiprocessing.set_start_method("spawn")

    batch_size = 30
    ds = OJClone.DataSet("dataset/OJClone/train.jsonl", max_node_count=512)
    loader = data.DataLoader(
        ds,
        batch_sampler=BatchSampler(ds, batch_size=batch_size, shuffle=True),
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    ds = OJClone.DataSet("dataset/OJClone/valid.jsonl", max_node_count=512)
    v_loader = data.DataLoader(
        ds,
        batch_sampler=BatchSampler(ds, batch_size=min(32, batch_size), shuffle=False),
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    model = AstAttention(384, 768, num_layers=6, num_heads=8).cuda()
    trainer = Trainer(model=model).cuda()

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=3e-5, weight_decay=0.1)

    scaler = GradScaler()

    try:
        with open(BEST_MODEL_PATH, "rb") as f:
            save = torch.load(f)
        model.load_state_dict(save["model_state_dict"], strict=False)
        min_loss = save["loss"]
    except IOError:
        logger.info("no model")
        min_loss = 1e8
    logger.info("min_loss {}".format(min_loss))
    try:
        save = torch.load(TRAINER_CKPT_PATH)
        trainer.load_state_dict(save["trainer_state_dict"], strict=True)
        optimizer.load_state_dict(save["optimizer_state_dict"])
        scaler.load_state_dict(save["scaler_state_dict"])
        epoch = save["epoch"]
    except IOError:
        epoch = 0
        logger.info("no ckpt")
    logger.info("epoch {}".format(epoch))

    while True:
        epoch += 1

        logger.info("===========================")
        logger.info("epoch {}".format(epoch))
        trainer.train()
        for batch in tqdm(loader):
            optimizer.zero_grad()
            loss = trainer(batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        trainer.evaluate()

        torch.save(check_point(trainer, optimizer, scaler, epoch), TRAINER_CKPT_PATH)

        if epoch % 1 == 0:
            with torch.no_grad():
                trainer.eval()
                for batch in tqdm(v_loader):
                    trainer(batch)
                loss = trainer.evaluate()

            if loss < min_loss:
                min_loss = loss
                torch.save(model_pt(model, loss), BEST_MODEL_PATH)
