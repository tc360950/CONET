from typing import Callable, TypeVar

import networkx as nx

NodeType = TypeVar("NodeType")


def apply_to_nodes_in_order(
    root: NodeType, tree: nx.DiGraph, func: Callable[[NodeType], None]
):
    def __apply_dfs(node: NodeType):
        func(node)
        for child in tree.successors(node):
            __apply_dfs(child)

    __apply_dfs(root)
