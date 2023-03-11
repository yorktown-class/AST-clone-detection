from typing import *

import torch
import json
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import itertools

from detecter.model import AstAttention, Classifier
from detecter import tree_tools
from detecter import parser
from detecter import word2vec


def parse_code_to_VE(code):
	return parser.parse(code, "c")

class BiDataset(Dataset):
	def __init__(self, path: str) -> None:
		super().__init__()
		with open(path, "r") as f:
			lines = f.readlines()
		self.raw_data_list = [json.loads(line) for line in lines]

		code_list = [item["code"] for item in self.raw_data_list]
		self.tree_VE_list = list(map(parse_code_to_VE, code_list))

		nodes_list = [V for (V, E) in self.tree_VE_list]
		all_nodes = list(itertools.chain(*nodes_list))
		self.word2vec_cache = word2vec.create_word_dict(all_nodes)
		print(len(self.word2vec_cache))

		n = len(self.raw_data_list)
		self.indexs = [(i, j) for i in range(n) for j in range(i + 1, n)]

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
			*tree_tools.tree_VE_to_tensor(tree_VE, word2vec_cache=self.word2vec_cache)
		)

def collate_fn(batch: List[Tuple[str, str, torch.Tensor, torch.Tensor]]):
	from torch.utils.data import default_collate

	collate_index = default_collate([(idx1, idx2) for idx1, idx2, _, _ in batch])
	collate_tree = tree_tools.collate_tree_tensor([(nodes, mask) for _, _, nodes, mask in batch])

	return *collate_index, *collate_tree


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
				"values": [item["result"] for item in topn]
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
				} for idx, value in zip(data["answers"], data["values"])
			]


if __name__ == "__main__":

	model = AstAttention(384, 768, 6, 8).cuda().eval()
	classifier = Classifier(768, 2).cuda().eval()

	save = torch.load("log/model.pt")
	model.load_state_dict(save["model_state_dict"])
	classifier.load_state_dict(save["classifier_state_dict"])

	dataloader = DataLoader(BiDataset("dataset/OJClone/test.jsonl"),
			 batch_size=4,
			 collate_fn=collate_fn,
			 num_workers=0)

	result_dict = ResultDict()

	try:
		with open("OJCloneTest.pt", "r") as f:
			prev_case_idx = int(f.readline())
			jsonl = f.readlines()
		result_dict.from_jsonl(jsonl)
	except IOError:
		prev_case_idx = -1

	for case_idx, (idx_list, jdx_list, nodes, mask) in enumerate(tqdm(dataloader)):
		if case_idx <= prev_case_idx:
			continue

		with torch.no_grad():
			hidden = model(nodes.cuda(), mask.cuda())
			output = classifier(hidden)
			result = output[:, 1] - output[:, 0]
			result_list = [result[i].item() for i in range(result.shape[0])]
		for idx, jdx, result in zip(idx_list, jdx_list, result_list):
			result_dict.insert(idx, jdx, result)
			result_dict.insert(jdx, idx, result)

		if case_idx % 10000 == 0:
			lines = [str(case_idx)] + result_dict.jsonl()
			lines = [line + "\n" for line in lines]
			with open("OJCloneTest.pt", "w") as f:
				f.writelines(lines)
	
	with open("OJCloneResult.jsonl", "w") as f:
		lines = result_dict.jsonl()
		lines = [line + "\n" for line in lines]
		f.writelines(lines)