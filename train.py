import logging

import pandas

import torch
from torch.utils import data

from detecter.dataset import PairCodeset, collate_fn
from BCB_model import Trainer
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from detecter import module_tools
import cProfile
import register


def train(model_name: str, num_epoch: int = None, use_tpe: bool = False, max_node_count: int = None, prune_node_count: int = None):
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
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    valid_ds = PairCodeset(data_source, pandas.read_pickle("dataset/BigCloneBench/valid.txt.bin"))
    valid_ds.drop(max_node_count).prune(prune_node_count).sample(30000).use_tpe(use_tpe)
    valid_loader = data.DataLoader(
        valid_ds, 
        batch_size=16, 
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    model = module_tools.PretrainModule(model_name)
    trainer = Trainer(model, device="cuda")
    optimizer = torch.optim.AdamW(params=trainer.parameters(), lr=1e-4, weight_decay=0.1)
    trained_chunks = 0
    min_loss = 1e8
    writer = SummaryWriter("log/tensor_board")

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

        trainer.cuda()
        trainer.train()
        for idx, batch in enumerate(tqdm(train_loader, desc="TRAIN")):
            optimizer.zero_grad()
            loss = trainer(batch)
            loss.backward()

            if idx % 100 == 0:
                detail = trainer.evaluate()
                logger.debug("train {}: ".format(trained_chunks) + ", ".join(["{} {:.4f}".format(key, value) for key, value in detail.items()]))

            if idx % 1000 == 0:
                for name, paramter in trainer.named_parameters():
                    if paramter.requires_grad and paramter.grad is not None:
                        logger.debug("  {}: {}, {}".format(name, paramter.data.norm(), paramter.grad.norm()))
            optimizer.step()

        detail = trainer.evaluate()
        print(", ".join(["{} {:.4f}".format(key, value) for key, value in detail.items()]))
        for key, value in detail.items():
            writer.add_scalar("{}/train/{}".format(model_name, key), value, global_step=trained_chunks)
        trainer.reset()
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

                if idx % 100 == 0:
                    detail = trainer.evaluate()
                    logger.debug("valid: " + ", ".join(["{} {:.4f}".format(key, value) for key, value in detail.items()]))
        detail = trainer.evaluate()
        print(", ".join(["{} {:.4f}".format(key, value) for key, value in detail.items()]))
        for key, value in detail.items():
            writer.add_scalar("{}/valid/{}".format(model_name, key), value, global_step=trained_chunks)
        trainer.reset()
        
        if detail["loss"] < min_loss:
            min_loss = detail["loss"]
            module_tools.save_module(model_name)
            save["min_loss"] = min_loss
            torch.save(save, "log/train.{}.ckpt".format(model_name))


def finetune(model_name: str, use_tpe: bool = False, max_node_count: int = None, prune_node_count: int = None):
    logger = logging.getLogger("{}_train".format(model_name))
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("log/{}.train.log".format(model_name), mode="a+")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("[%(asctime)s:%(levelname)s] - %(message)s"))
    logger.addHandler(fh)

    
    data_source = pandas.read_pickle("dataset/BigCloneBench/data.jsonl.txt.bin")
    train_ds = PairCodeset(data_source, pandas.read_pickle("dataset/BigCloneBench/valid.txt.bin"))
    train_ds.drop(max_node_count).prune(prune_node_count).use_tpe(use_tpe)
    print(len(train_ds))
    train_loader = data.DataLoader(
        train_ds, 
        batch_size=16, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    model = module_tools.PretrainModule(model_name)
    trainer = Trainer(model, device="cuda")

    prefix = "model.linear"
    for name, param in model.named_parameters():
        if name[:len(prefix)] == prefix:
            print("skip {}".format(name))
            continue
        param.requires_grad = False

    optimizer = torch.optim.AdamW(params=trainer.parameters(), lr=1e-4)

    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        loss = trainer(batch)
        loss.backward()
        optimizer.step()
    detail = trainer.evaluate()
    print(", ".join(["{} {:.4f}".format(key, value) for key, value in detail.items()]))
    module_tools.save_module(model_name)


if __name__ == "__main__":
    # cProfile.run('train()', 'profile.txt')
    # train("BCBdetecter_tpe", use_tpe=True, num_epoch=3, max_node_count=256)
    # train("BCBdetecter", num_epoch=3, max_node_count=256)
    # train("BCBdetecter_no_mask", num_epoch=3, max_node_count=256)
    # train("BCBdetecter_final", num_epoch=3, use_tpe=True, max_node_count=1000)
    finetune("BCBdetecter_final", use_tpe=True, max_node_count=1000)
