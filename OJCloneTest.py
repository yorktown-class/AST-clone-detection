import json
from typing import *

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from detecter import parser, tree_tools, word2vec
from detecter.Xdataset import OJClone
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


# def collate_fn(batch: List[Tuple[str, str, torch.Tensor, torch.Tensor]]):
# from torch.utils.data import default_collate

# collate_index = default_collate([(idx1, idx2) for idx1, idx2, _, _ in batch])
# idx1_list = [idx1 for idx1, idx2, _, _ in batch]
# idx2_list = [idx2 for idx1, idx2, _, _ in batch]
# collate_tree = tree_tools.collate_tree_tensor([(nodes, mask) for _, _, nodes, mask in batch])

# return idx1_list, idx2_list, *collate_tree


def collate_fn(batch: List[Tuple[str, tree_tools.TreeV, tree_tools.TreeE]]):
    index = [idx for idx, _, _ in batch]
    tree_VE = [(tree_V, tree_E) for _, tree_V, tree_E in batch]
    return torch.tensor(index, dtype=torch.long), *tree_tools.collate_tree_tensor(tree_VE)


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
    # classifier = Classifier(768, 2).eval().cuda()

    save = torch.load("log/model.pt")
    model.load_state_dict(save["model_state_dict"])
    # classifier.load_state_dict(save["classifier_state_dict"])

    dataset = OJClone.DataSet("dataset/OJClone/test.jsonl", max_node_count=1024)
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn, pin_memory=True, num_workers=4)

    result_dict = ResultDict()

    try:
        with open("log/OJCloneTest.pt", "r") as f:
            prev_case_idx = int(f.readline())
            jsonl = f.readlines()
        result_dict.from_jsonl(jsonl)
    except IOError:
        prev_case_idx = -1

    ids = 0
    with torch.no_grad():
        tlabel, tree_V, tree_E = dataset[0]
        tfeature = model(tree_V.unsqueeze(1).cuda(), tree_E.unsqueeze(0).cuda())[0, 0, :]

        for label, tree_V, tree_E in tqdm(dataloader):
            hidden = model(tree_V.cuda(), tree_E.cuda())[0]

            score = torch.sum(tfeature[None, :] * hidden, dim=1)

            for s in score:
                result_dict.insert(0, ids, s.item())
                ids += 1

    with open("log/OJCloneResult.jsonl", "w") as f:
        lines = result_dict.jsonl()
        lines = [line + "\n" for line in lines]
        f.writelines(lines)

    arr = result_dict.result_dict[0]
    cnt = filter(lambda x: x["index"] < 500, arr)
    print(len(list(cnt)))
