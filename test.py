import logging
from typing import *
import pandas

import torch
from torch.utils import data

from detecter.dataset import PairCodeset, collate_fn
from BCB_model import Trainer
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from detecter import module_tools
from torcheval import metrics
import register


def test(model_name: str, use_tpe: bool = False, max_node_count: int = None, prune_node_count: int = None, batch_size: int = 32):
    logger = logging.getLogger("BCBtest")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("log/{}.test.log".format(model_name), mode="a+")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("[%(asctime)s:%(levelname)s] - %(message)s"))
    logger.addHandler(fh)

    test_ds = PairCodeset(pandas.read_pickle("dataset/BigCloneBench/data.jsonl.txt.bin"), pandas.read_pickle("dataset/BigCloneBench/test.txt.bin"))
    test_ds.drop(max_node_count).prune(prune_node_count).sample(30000).use_tpe(use_tpe)
    test_loader = data.DataLoader(
        test_ds, 
        batch_size=batch_size, 
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    model = module_tools.PretrainModule(model_name).cuda()
    evaluators: Dict[str, metrics.Metric] = {
        "f1": metrics.BinaryF1Score(device="cuda"),
        "precision": metrics.BinaryPrecision(device="cuda"),
        "recall": metrics.BinaryRecall(device="cuda"),
        "accuracy": metrics.BinaryAccuracy(device="cuda"),
    }
    pr_curve = metrics.BinaryPrecisionRecallCurve(device="cuda")

    model.eval().cuda()
    with torch.inference_mode():
        for idx, batch in enumerate(tqdm(test_loader, desc="TEST")):
            label, nodes, dist = [item.cuda() for item in batch]
            score = torch.sigmoid(model(nodes, dist)) 

            for evaluator in evaluators.values():
                evaluator.update(score, label.long())
            pr_curve.update(score, label.long())

            if idx % 100 == 0:
                logger.debug("test: " + ", ".join(["{} {:.4f}".format(key, evaluator.compute()) for key, evaluator in evaluators.items()]))
    print("test: " + ", ".join(["{} {:.4f}".format(key, evaluator.compute()) for key, evaluator in evaluators.items()]))

    # pcurve, rcurve, threshold = pr_curve.compute()
    # f1curve = 2 / (1 / pcurve + 1 / rcurve)
    # max_id = f1curve.argmax()
    # print("f1 {:.4f}, precision {:.4f}, recall {:.4f}, threshold {:.4f}".format(
    #     f1curve[max_id], pcurve[max_id], rcurve[max_id], threshold[max_id]
    # ))


if __name__ == "__main__":
    # test("BCBdetecter_no_mask")
    # test("BCBdetecter")  
    test("BCBdetecter_final", use_tpe=True, prune_node_count=1000, batch_size=32)