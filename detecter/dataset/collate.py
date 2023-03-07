# from typing import *

# import torch
# import itertools
# from torch_geometric.data import Data



# def collate_fn(batch: List[str, Data]) -> Tuple[List[str], Data]: # [label, v, e] -> labels, V, E, root_ids
# 	labels: List[str] = [label for (label, data) in batch]
# 	data_list: List[Data] = [data for (label, data) in batch]

# 	# for i in range(len(v_list)):
# 	# 	V = v_list[i]
# 	# 	E = e_list[i]
# 	# 	assert(max(max(E[0]), max(E[1])) < len(V))
	
# 	vlen = torch.tensor([0] + [len(v) for v in v_list], dtype=torch.int64)
# 	root_ids = torch.cumsum(vlen[:-1], dim=0)
# 	V = list(itertools.chain(*v_list))
# 	# V = torch.cat([torch.tensor(v) for v in v_list])
# 	E = torch.cat([torch.tensor(e) + root_ids[idx] for idx, e in enumerate(e_list)], dim=1)
# 	print(len(V))
# 	return labels, V, E, root_ids
	