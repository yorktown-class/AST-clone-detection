from typing import *

import tree_sitter

from . import tree_tools


class ParseError(Exception):
    pass


def is_punctuation(sent: str) -> bool:
    punctuation_list = """
    '";{}[]()\n
    """
    punctuation_list = list(punctuation_list)

    return sent in punctuation_list


def is_comment(sent: str) -> bool:
    if sent[0:2] == "//":
        return True
    if sent[0:2] == "/*":
        return True
    return False


def parse(code: str, lang: str = "c") -> tree_tools.TreeVE:
    parser = tree_sitter.Parser()
    parser.set_language(tree_sitter.Language("build/lang.so", lang))
    try:
        tree = parser.parse(bytes(code, encoding="utf-8"))
    except Exception:
        raise ParseError

    V = list()
    E = (list(), list())

    identifier_dict = dict()

    def get_identifier_id(idt_name: str):
        if idt_name not in identifier_dict:
            identifier_dict[idt_name] = len(identifier_dict)
        return identifier_dict[idt_name]

    def walk(node: tree_sitter.Node):
        if node.type == "comment" or is_punctuation(node.type):
            return None
        elif node.type == "identifier":
            desc = node.type + "_" + str(get_identifier_id(node.text.decode("utf-8")))
            V.append(desc)
            return len(V) - 1
        elif node.type[-len("literal") :] == "literal":
            desc = node.type + ": " + node.text.decode("utf-8")
            V.append(desc)
            return len(V) - 1

        V.append(node.type)
        vid = len(V) - 1
        for child in node.children:
            child_vid = walk(child)
            if child_vid is None:
                continue
            E[0].append(child_vid)
            E[1].append(vid)

        return vid

    walk(tree.root_node)
    return (V, E)


def parse_to_tensor(code: str, lang: str = "c"):
    pass
