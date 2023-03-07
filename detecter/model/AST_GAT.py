from typing import *

import torch
from torch.nn import Embedding, Linear
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear, PositionalEncoding
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer


sentence2emb = SentenceTransformer('all-MiniLM-L6-v2')


class AST_GAT(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        
        # self.var_embedding = Embedding(1**64, 128, sparse=True)
        self.var_embedding = PositionalEncoding(128)
        self.word_bag = dict() # hash -> int
        self.code_embedding = Embedding(200, 384)
        self.linear = Linear(384, 384)

        self.conv = HeteroConv({
            ("variable", "compose", "subcode"): SAGEConv(384, hidden_size, aggr="sum"),
            ("subcode", "combine", "subcode"): SAGEConv(384, hidden_size, aggr="sum"),
        }, aggr='sum')

        self.conv_loop = SAGEConv(hidden_size, hidden_size, aggr="mean")


    def densification_keywords(self, t: torch.Tensor):
        def try_map(key: int):
            if key not in self.word_bag:
                value = len(self.word_bag)
                assert(value < 200)
                self.word_bag[key] = value
            return self.word_bag[key]
             
        t.apply_(try_map)

    def densification_keyword(self, key: str):
        key = hash(key)        
        if key not in self.word_bag:
            value = len(self.word_bag)
            assert(value < 200)
            self.word_bag[key] = value
        return self.word_bag[key]
            

    def forward(self, V: List[str], E: torch.Tensor, root_ids: torch.Tensor) -> torch.Tensor:
        """
        V (N): 顶点
        E (2, M): 有向边
        root_ids (B): 树根下标
        N 定点数, M 边数, B 树的数量(也是Batch大小)
        """

        E = E.cuda()
        root_ids = root_ids.cuda()

        N = len(V)
        _, M = E.shape
        B = root_ids.shape[0]

        non_leaf = torch.unique(E[1, :]).to(dtype=torch.int64)

        leaf_mask = torch.ones(N, dtype=torch.bool).cuda()
        leaf_mask[non_leaf] = False

        data = HeteroData().cuda()

        varible_list = []
        subcode_list = []

        for idx, is_leaf in enumerate(leaf_mask):
            if is_leaf:
                varible_list.append(V[idx])
            else:
                subcode_list.append(self.densification_keyword(V[idx]))
        varible_list = torch.tensor(sentence2emb.encode(varible_list)).cuda()
        # print(varible_list.shape)
        subcode_list = self.code_embedding(torch.tensor(subcode_list).cuda())

        data["variable"].v = self.linear(varible_list)
        data["subcode"].v = subcode_list
        # data["variable"].v = V.masked_select(leaf_mask)
        # data["subcode"].v = V.masked_select(~leaf_mask)
        # data["variable"].v = self.var_embedding(data["variable"].v)
        # self.densification_keywords(data["subcode"].v)
        # data["subcode"].v = self.code_embedding(data["subcode"].v)

        leaf_index = torch.cumsum(leaf_mask, dim=0, dtype=torch.int) - 1
        nleaf_index = torch.cumsum(~leaf_mask, dim=0, dtype=torch.int) - 1
        index_map = torch.where(leaf_mask, leaf_index, nleaf_index)

        compose_edge_index = torch.where(leaf_mask[E[0, :].to(dtype=torch.int64)].to(dtype=torch.int64))[0]
        compose_edge = index_map[E[:, compose_edge_index].to(dtype=torch.int64)]
        combine_edge_index = torch.where(~leaf_mask[E[0, :].to(dtype=torch.int64)].to(dtype=torch.int64))[0]
        combine_edge = index_map[E[:, combine_edge_index].to(dtype=torch.int64)]

        data["variable", "compose", "subcode"].e = compose_edge.to(dtype=torch.int64)
        data["subcode", "combine", "subcode"].e = combine_edge.to(dtype=torch.int64)

        # print(data.v_dict)
        # print(data.e_dict)

        output = self.conv.forward(data.v_dict, data.e_dict)["subcode"]

        for i in range(20):
            output = self.conv_loop.forward(output, combine_edge.to(dtype=torch.int64))

            non_leaf = torch.unique(combine_edge[1, :]).to(dtype=torch.int64)
            leaf_mask = torch.ones(N, dtype=torch.bool).cuda()
            # leaf_mask.index_fill(0, non_leaf, False)
            leaf_mask[non_leaf] = False

            index = torch.where(~leaf_mask[combine_edge[0, :].to(dtype=torch.int64)])[0]
            combine_edge = combine_edge[:, index]
            if len(index) == 0:
                break



        # print(index_map[root_ids])
        root_hidden = output.index_select(dim=0, index=index_map[root_ids])
        return root_hidden

