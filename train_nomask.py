
import BCB_model
from detecter import module_tools
import torch
import logging
from torch.utils import data
import pandas
from detecter.dataset import PairCodeset, collate_fn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from detecter import tree_transformer


class TreeTransformerNoMask(tree_transformer.TreeTransformer):
    def forward(self, nodes: torch.Tensor, dist: torch.Tensor):
        embedding = self.dense(self.position_embedding(nodes))
        hidden = self.encoder(embedding)[0]
        return self.bn(hidden)

module_tools.register_module("ast_transformer_no_mask", TreeTransformerNoMask(128, 128, 2, 2, 4, 2, 0.1))

module_tools.register_module("BCBdetecter_no_mask", BCB_model.Detecter("ast_transformer_no_mask"))

class Trainer(BCB_model.Trainer):
    def __init__(self, device="cuda") -> None:
        super().__init__(device)
        self.model = module_tools.PretrainModule("BCBdetecter_no_mask")


if __name__ == "__main__":
    logger = logging.getLogger("BCBtrainNoMask")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("log/BCBtrainNoMask.log", mode="a+")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("[%(asctime)s:%(levelname)s] - %(message)s"))
    logger.addHandler(fh)

    data_source = pandas.read_pickle("dataset/BigCloneBench/data.jsonl.txt.bin")
    train_ds = PairCodeset(data_source, pandas.read_pickle("dataset/BigCloneBench/train.txt.bin"))
    train_ds.drop(max_node_count=1024)
    print(len(train_ds))
    train_loader = data.DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    valid_ds = PairCodeset(data_source, pandas.read_pickle("dataset/BigCloneBench/test.txt.bin"))
    valid_ds.drop(max_node_count=1024).sample(10000)
    valid_loader = data.DataLoader(
        valid_ds, 
        batch_size=16, 
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    trainer = Trainer("cuda")
    optimizer = torch.optim.AdamW(params=trainer.parameters(), lr=1e-4, weight_decay=0.1)
    trained_chunks = 0
    min_loss = 1e8
    writer = SummaryWriter("log/tensor_board")

    try:
        save = torch.load("log/train_no_mask.ckpt")
        trainer.load_state_dict(save["trainer_state_dict"])
        optimizer.load_state_dict(save["optimizer_state_dict"])
        trained_chunks = save["trained_chunks"]
        min_loss = save["min_loss"]
        print(min_loss)
    except IOError:
        print("no ckpt")
    
    while True:
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
            writer.add_scalar("trainBCBNoMask/{}".format(key), value, global_step=trained_chunks)
        trainer.reset()
        trained_chunks += 1

        save = {
            "trainer_state_dict": trainer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "trained_chunks": trained_chunks,
            "min_loss": min_loss,
        }
        torch.save(save, "log/train_no_mask.ckpt")

        trainer.eval()
        with torch.inference_mode():
            for idx, batch in enumerate(tqdm(valid_loader, desc="VALID")):
                trainer(batch)

                if idx % 100 == 0:
                    detail = trainer.evaluate()
                    logger.debug("valid: " + ", ".join(["{} {:.4f}".format(key, value) for key, value in detail.items()]))
        detail = trainer.evaluate()
        print(", ".join(["{} {:.4f}".format(key, value) for key, value in detail.items()]))
        for key, value in detail.items():
            writer.add_scalar("validBCBNoMask/{}".format(key), value, global_step=trained_chunks)
        trainer.reset()
        
        if detail["loss"] < min_loss:
            min_loss = detail["loss"]
            module_tools.save_module("BCBdetecter_no_mask")
            module_tools.save_module("ast_transformer_no_mask")
            save["min_loss"] = min_loss
            torch.save(save, "log/train_no_mask.ckpt")
