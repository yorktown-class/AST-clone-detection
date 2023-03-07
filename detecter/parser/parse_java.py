from typing import *

import javalang
from javalang import ast as j_ast


def ast_to_dict(node: Union[j_ast.Node, List, Tuple, str, int, float]):
    if isinstance(node, j_ast.Node):
        res = {}
        for name in node.attrs:
            child = getattr(node, name)
            res[name] = ast_to_dict(child)
        return {node.__class__.__name__: res}

    if isinstance(node, (list, tuple)):
        return {"list_{}".format(idx) : ast_to_dict(value) for idx, value in enumerate(node)}

    return str(node)


def code_to_ast(code: str):
    tokens = javalang.tokenizer.tokenize(code)
    parser = javalang.parser.Parser(tokens)
    return parser.parse_member_declaration()
