import json
from typing import *

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from detecter import parser, tree_tools, word2vec
from detecter.dataset import OJClone
from detecter.model import AstAttention, Classifier


def parse_code_to_VE(code):
    return tree_tools.tree_VE_prune(parser.parse(code, "c"), 512)


class BiDataset(OJClone.DataSet):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.indexs = [(i, j) for i in range(1) for j in range(self.length)]

    def __len__(self):
        return len(self.indexs)

    def __getitem__(self, idx) -> Tuple[str, str, torch.Tensor, torch.Tensor]:
        i, j = self.indexs[idx]
        tree_VE_i = self.tree_VE_list[i]
        tree_VE_j = self.tree_VE_list[j]
        tree_VE = tree_tools.merge_tree_VE(tree_VE_i, tree_VE_j, "<CODE_COMPARE>")
        return (
            self.raw_data_list[i]["index"],
            self.raw_data_list[j]["index"],
            *tree_tools.tree_VE_to_tensor(tree_VE, word2vec_cache=self.word_dict),
        )


def collate_fn(batch: List[Tuple[str, str, torch.Tensor, torch.Tensor]]):
    # from torch.utils.data import default_collate

    # collate_index = default_collate([(idx1, idx2) for idx1, idx2, _, _ in batch])
    idx1_list = [idx1 for idx1, idx2, _, _ in batch]
    idx2_list = [idx2 for idx1, idx2, _, _ in batch]
    collate_tree = tree_tools.collate_tree_tensor([(nodes, mask) for _, _, nodes, mask in batch])

    return idx1_list, idx2_list, *collate_tree


class ResultDict:
    def __init__(self) -> None:
        self.result_dict = dict()

    def insert(self, lhs, rhs, result):
        N = 499
        topn = self.result_dict.get(lhs, [])
        k = len(topn) - 1
        item = {
            "index": rhs,
            "result": result,
        }
        topn.append(item)

        while k >= 0 and item["result"] > topn[k]["result"]:
            topn[k + 1] = topn[k]
            k -= 1
        topn[k + 1] = item

        if len(topn) > N:
            topn = topn[:N]

        self.result_dict[lhs] = topn

    def jsonl(self) -> List[str]:
        output = []
        for key, topn in self.result_dict.items():
            result = {
                "index": key,
                "answers": [item["index"] for item in topn],
                "values": [item["result"] for item in topn],
            }
            output.append(json.dumps(result))
        return output

    def from_jsonl(self, jsonl: List[str]):
        for line in jsonl:
            data = json.loads(line)
            self.result_dict[data["index"]] = [
                {
                    "index": idx,
                    "result": value,
                }
                for idx, value in zip(data["answers"], data["values"])
            ]


if __name__ == "__main__":
    model = AstAttention(384, 768, 6, 8).eval().cuda()
    classifier = Classifier(768, 2).eval().cuda()

    save = torch.load("log/model.pt")
    model.load_state_dict(save["model_state_dict"])
    classifier.load_state_dict(save["classifier_state_dict"])

    dataloader = DataLoader(
        BiDataset("dataset/OJClone/test.jsonl", max_node_count=512),
        batch_size=32,
        collate_fn=collate_fn,
        num_workers=2,
    )

    result_dict = ResultDict()

    for p in model.parameters():
        p.requires_grad = False
    for p in classifier.parameters():
        p.requires_grad = False

    try:
        with open("OJCloneTest.pt", "r") as f:
            prev_case_idx = int(f.readline())
            jsonl = f.readlines()
        result_dict.from_jsonl(jsonl)
    except IOError:
        prev_case_idx = -1

    with torch.no_grad():
        for case_idx, (idx_list, jdx_list, nodes, mask) in enumerate(tqdm(dataloader)):
            if case_idx <= prev_case_idx:
                continue

            hidden = model(nodes.cuda(), mask.cuda())
            # print(hidden[0])
            # print(torch.mean(hidden, dim=0))
            # hidden = torch.mean(hidden, dim=0)
            output = classifier(hidden[0])
            output = torch.softmax(output, dim=-1)
            result = output[:, 1]
            result_list = [result[i].item() for i in range(result.shape[0])]

            for idx, jdx, result in zip(idx_list, jdx_list, result_list):
                result_dict.insert(idx, jdx, result)
                # result_dict.insert(jdx, idx, result)

            if case_idx % 100 == 0:
                lines = [str(case_idx)] + result_dict.jsonl()
                lines = [line + "\n" for line in lines]
                with open("OJCloneTest.pt", "w") as f:
                    f.writelines(lines)

    with open("OJCloneResult.jsonl", "w") as f:
        lines = result_dict.jsonl()
        lines = [line + "\n" for line in lines]
        f.writelines(lines)
