from typing import *

from pycparser import c_ast, c_parser
import re


def comment_remover(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return ""
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)


def ast_to_dict(node: Union[c_ast.Node, List, Tuple, str, int, float]):
    if isinstance(node, c_ast.Node):
        res = {}
        for name in node.__slots__[:-2]:
            child = getattr(node, name)
            res[name] = ast_to_dict(child)
        return {node.__class__.__name__: res}

    if isinstance(node, (list, tuple)):
        return {"list_{}".format(idx) : ast_to_dict(value) for idx, value in enumerate(node)}
    
    return str(node)


def code_to_ast(code: str):
    parser = c_parser.CParser()
    return parser.parse(comment_remover(code))
