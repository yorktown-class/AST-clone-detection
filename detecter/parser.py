from typing import *

import tree_sitter
import numpy

from . import tree_tools


class ParseError(Exception):
    pass


punctuation_list = """
'";{}[],.\n
"""
punctuation_list = list(punctuation_list)

def is_punctuation(word: str) -> bool:
    global punctuation_list
    return word in punctuation_list


def parse(code: str, lang: str = "c") -> tree_tools.Tree:
    parser = tree_sitter.Parser()
    parser.set_language(tree_sitter.Language("build/lang.so", lang))
    try:
        tree = parser.parse(bytes(code, encoding="utf-8"))
    except Exception:
        raise ParseError

    nodes = list()
    parents = list()

    def walk(node: tree_sitter.Node, parent) -> None:
        nonlocal nodes, parents

        node_type = node.type

        if node_type == "comment" or is_punctuation(node_type):
            return

        nodes.append(node_type)
        parents.append(parent)
        node_id = len(nodes) - 1

        if len(node.children) == 0:
            try:
                node_text = node.text.decode("ascii")
            except UnicodeDecodeError:
                node_text = "UnicodeDecodeError"
            if node_text != node_type:
                nodes.append(node_text)
                parents.append(node_id)

        for child in node.children:
            walk(child, node_id)

    walk(tree.root_node, -1)

    return numpy.array(nodes, dtype=numpy.str), numpy.array(parents, dtype=numpy.int)

