from typing import *

import torch
# from torch_geometric.data import Data
# from torch_geometric.typing import OptTensor
# from sentence_transformers import SentenceTransformer

from . import parse_c, parse_java


# class Tree(Data):
#     def __init__(self, x: OptTensor = None, edge_index: OptTensor = None, root: int = None, edge_attr: OptTensor = None, y: OptTensor = None, pos: OptTensor = None, **kwargs):
#         super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)
#         self.root = root
    
#     # def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
#     #     if key == 'root':
#     #         return None
#     #     return super().__cat_dim__(key, value, *args, **kwargs)

#     def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
#         if key == 'root':
#             return self.x.shape[0]
#         return super().__inc__(key, value, *args, **kwargs)


class ParseError(Exception):
    pass


code_to_ast = {
    "c": parse_c.code_to_ast,
    "java": parse_java.code_to_ast,
}

ast_to_dict = {
    "c": parse_c.ast_to_dict,
    "java": parse_java.ast_to_dict,
}


# sentence2emb = SentenceTransformer('all-MiniLM-L6-v2').cuda()


def parse(code: str, lang: str) -> Tuple[List[str], Tuple[List[int], List[int]]]:
    assert(lang in ("c", "java"))
    
    try:
        node = code_to_ast[lang](code)
    except:
        raise ParseError
    
    tree: Dict = ast_to_dict[lang](node)

    V: List[int] = list()
    E: Tuple[List, List] = (list(), list())

    V.append("<CODE_ROOT>")

    def get_VE_from_dict(tree: Dict, parent_id: int):
        assert(isinstance(tree, (str, dict)))

        if isinstance(tree, dict) and len(tree):
            for key, subtree in tree.items():
                V.append(key)
                vid = len(V) - 1
                E[0].append(vid)
                E[1].append(parent_id)
                get_VE_from_dict(subtree, vid)
        else:
            value = tree if isinstance(tree, str)  else ""
            V.append(value)
            E[0].append(len(V) - 1)
            E[1].append(parent_id)

    get_VE_from_dict(tree, 0)
    return (V, E)
