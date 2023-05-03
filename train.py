import logging

import pandas

import torch
from torch.utils import data

from detecter.dataset import PairCodeset, collate_fn
from BCB_model import Trainer
from torch.utils.tensorboard import SummaryWriter
from torcheval import metrics

from tqdm import tqdm
from detecter import module_tools
import cProfile
import register
from typing import *


def log_metrics(log_func: Callable, title: str, detail: Dict):
    log_func(title + " " + ", ".join(["{} {:.4f}".format(key, value) for key, value in detail.items()]))

def add_scalar(dir: str, tag: str, detail: Dict, step):
    with SummaryWriter(dir) as sw:
        for key, value in detail.items():
            sw.add_scalar("{}/{}".format(tag, key), value, global_step=step)

def log_grads(log_func: Callable, model: torch.nn.Module):
    for name, paramter in model.named_parameters():
        if paramter.requires_grad and paramter.grad is not None:
            log_func("  {}: {}, {}".format(name, paramter.data.norm(), paramter.grad.norm()))


def train(model_name: str, num_epoch: int = None, use_tpe: bool = False, max_node_count: int = None, prune_node_count: int = None, shuffle: bool = False):
    print("train {}".format(model_name))
    logger = logging.getLogger("{}_train".format(model_name))
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("log/{}.train.log".format(model_name), mode="a+")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("[%(asctime)s:%(levelname)s] - %(message)s"))
    logger.addHandler(fh)

    data_source = pandas.read_pickle("dataset/BigCloneBench/data.jsonl.txt.bin")
    train_ds = PairCodeset(data_source, pandas.read_pickle("dataset/BigCloneBench/train.txt.bin"))
    train_ds.drop(max_node_count).prune(prune_node_count).use_tpe(use_tpe)
    print(len(train_ds))
    train_loader = data.DataLoader(
        train_ds, 
        batch_size=16, 
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    valid_ds = PairCodeset(data_source, pandas.read_pickle("dataset/BigCloneBench/valid.txt.bin"))
    valid_ds.drop(max_node_count).prune(prune_node_count).use_tpe(use_tpe).sample(10000)
    valid_loader = data.DataLoader(
        valid_ds, 
        batch_size=16, 
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    model = module_tools.get_module(model_name).cuda()
    trainer = Trainer(model, device="cuda")
    optimizer = torch.optim.AdamW(params=trainer.parameters(), lr=1e-4, weight_decay=0.1)
    trained_chunks = 0
    min_loss = 1e8
    tbdir = "log/tensor_board/{}".format(model_name)

    try:
        save = torch.load("log/train.{}.ckpt".format(model_name))
        trainer.load_state_dict(save["trainer_state_dict"])
        optimizer.load_state_dict(save["optimizer_state_dict"])
        trained_chunks = save["trained_chunks"]
        min_loss = save["min_loss"]
        print(min_loss)
    except IOError:
        print("no ckpt")
    
    while num_epoch is None or trained_chunks < num_epoch:
        epoch = trained_chunks + 1
        print("============== EPOCH: {} ============== ".format(epoch))

        trainer.train()
        for idx, batch in enumerate(tqdm(train_loader, desc="TRAIN")):
            optimizer.zero_grad()
            loss = trainer(batch)
            loss.backward()

            if (idx + 1) % 1000 == 0:
                detail = trainer.evaluate()
                trainer.reset()
                log_metrics(logger.debug, "train {}".format(epoch), detail)
                add_scalar(tbdir, "train", detail, trained_chunks * len(train_loader) + idx + 1)
            if idx % 1000 == 0:
                log_grads(logger.debug, trainer)

            optimizer.step()

        trainer.reset()
        log_metrics(print, "", detail)
        trained_chunks += 1

        save = {
            "trainer_state_dict": trainer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "trained_chunks": trained_chunks,
            "min_loss": min_loss,
        }
        torch.save(save, "log/train.{}.ckpt".format(model_name))

        trainer.cuda().eval()
        with torch.inference_mode(True):
            for idx, batch in enumerate(tqdm(valid_loader, desc="VALID")):
                trainer(batch)

                if idx % 1000 == 0:
                    log_metrics(logger.debug, "valid {}".format(epoch), trainer.evaluate())

        detail = trainer.evaluate()
        trainer.reset()
        log_metrics(print, "", detail)
        add_scalar(tbdir, "valid", detail, epoch)
        
        if detail["loss"] < min_loss:
            min_loss = detail["loss"]
            module_tools.save_module(model_name)
            save["min_loss"] = min_loss
            torch.save(save, "log/train.{}.ckpt".format(model_name))


def find_threshold(model_name: str, use_tpe: bool = False, max_node_count: int = None, prune_node_count: int = None):
    # logger = logging.getLogger("{}_train".format(model_name))
    # logger.setLevel(logging.DEBUG)
    # fh = logging.FileHandler("log/{}.train.log".format(model_name), mode="a+")
    # fh.setLevel(logging.DEBUG)
    # fh.setFormatter(logging.Formatter("[%(asctime)s:%(levelname)s] - %(message)s"))
    # logger.addHandler(fh)

    
    data_source = pandas.read_pickle("dataset/BigCloneBench/data.jsonl.txt.bin")
    train_ds = PairCodeset(data_source, pandas.read_pickle("dataset/BigCloneBench/valid.txt.bin"))
    train_ds.drop(max_node_count).prune(prune_node_count).use_tpe(use_tpe).sample(5000)
    print(len(train_ds))
    train_loader = data.DataLoader(
        train_ds, 
        batch_size=16, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    model = module_tools.get_module(model_name).cuda().eval()
    pr_curve = metrics.BinaryPrecisionRecallCurve(device="cuda")

    with torch.inference_mode():
        for idx, batch in enumerate(tqdm(train_loader)):
            label, node, dist = [item.cuda() for item in batch]
            score = torch.sigmoid(model(node, dist))
            pr_curve.update(score, label.long())

    pcurve, rcurve, threshold = pr_curve.compute()
    f1curve = 2 / (1 / pcurve + 1 / rcurve)
    max_id = f1curve.argmax()
    print("f1 {:.4f}, precision {:.4f}, recall {:.4f}, threshold {:.4f}".format(
        f1curve[max_id], pcurve[max_id], rcurve[max_id], threshold[max_id]
    ))


def fine_tune(model_name: str, use_tpe: bool = False, max_node_count: int = None, prune_node_count: int = None):
    data_source = pandas.read_pickle("dataset/BigCloneBench/data.jsonl.txt.bin")
    train_ds = PairCodeset(data_source, pandas.read_pickle("dataset/BigCloneBench/valid.txt.bin"))
    train_ds.drop(max_node_count).prune(prune_node_count).use_tpe(use_tpe).sample(20000)

    train_loader = data.DataLoader(
        train_ds, 
        batch_size=16, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    model = module_tools.get_module(model_name).cuda().eval()
    trainer = Trainer(model, "cuda").cuda().eval()
    prefix = "linear"
    for name, param in model.named_parameters():
        if name[: len(prefix)] == prefix:
            print(name)
            continue
        param.requires_grad = False
    optimizer = torch.optim.AdamW(model.linear.parameters(), lr=1e-4)

    for idx, batch in enumerate(tqdm(train_loader)):
        loss = trainer(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    log_metrics(print, "", trainer.evaluate())
    import copy
    mn = model_name + "_finetune"
    module_tools.save_module(mn)


if __name__ == "__main__":
    # train("BCBdetecter_complete", num_epoch=1, max_node_count=512, prune_node_count=256, use_tpe=True, shuffle=False)
    # train("BCBdetecter_basic", num_epoch=1, max_node_count=512, prune_node_count=256, use_tpe=False, shuffle=False)
    # train("BCBdetecter_mask", num_epoch=1, max_node_count=512, prune_node_count=256, use_tpe=False, shuffle=False)
    # train("BCBdetecter_tpe", num_epoch=1, max_node_count=512, prune_node_count=256, use_tpe=True, shuffle=False)
    
    train("BCBdetecter", num_epoch=3, prune_node_count=1000, use_tpe=True, shuffle=True)
    find_threshold("BCBdetecter", use_tpe=True, prune_node_count=1000)
    # fine_tune("BCBdetecter", use_tpe=True, prune_node_count=1000)
