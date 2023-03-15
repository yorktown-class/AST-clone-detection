import tree_sitter

tree_sitter.Language.build_library(
    "build/lang.so",
    [
        "build/tree-sitter-c",
    ],
)
