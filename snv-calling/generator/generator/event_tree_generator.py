import random
from dataclasses import dataclass
from typing import Dict, List, Set, TypeVar

import networkx as nx
import numpy as np

from generator.core.event_tree import EventTree
from generator.core.types import CNEvent, EventTreeRoot, SNVEvent
from generator.generator.context import SNVGeneratorContext
from generator.generator.gen_utils import sample_conditionally_without_replacement

T = TypeVar("T")


class RandomWalkTreeSampler:
    @staticmethod
    def sample_tree(nodes: List[T]) -> nx.DiGraph:
        """
        Sample random uniform directed rooted tree built from (all) nodes in @nodes.
        Always nodes[0] is assumed to be the root.
        """
        tree = nx.DiGraph()
        tree.add_node(nodes[0])
        current_node = nodes[0]
        while (
            len(nodes) > tree.size() + 1
        ):  # nx.DiGraph.size evaluates to the number of graph edges
            new_node = random.sample(range(0, len(nodes)), 1)[0]
            if nodes[new_node] not in set(tree.nodes):
                tree.add_edge(current_node, nodes[new_node])
            current_node = nodes[new_node]
        return tree


@dataclass
class EventTreeGenerator:
    context: SNVGeneratorContext

    def generate(self, tree_size: int) -> EventTree:
        tree = nx.DiGraph()
        nodes = self.__generate_cn_event_nodes(tree_size)
        trunk_size = np.random.randint(int(0.1 * len(nodes)), int(0.4 * len(nodes)))
        trunk = nodes[0:trunk_size]
        trunk.sort(key=lambda node: node[0]* 10000 + node[1])

        for i in range(1, trunk_size):
            tree.add_edge(nodes[i-1], nodes[i])
        tree.add_edge(nodes[trunk_size -1], nodes[trunk_size])
        sub_trunk_edges = list(
            RandomWalkTreeSampler.sample_tree(nodes[trunk_size:]
        ).edges)
        for edge in sub_trunk_edges:
            tree.add_edge(edge[0], edge[1])
        return EventTree(tree, self.__generate_snvs(tree))

    def __generate_cn_event_nodes(self, tree_size: int) -> List[CNEvent]:
        return [EventTreeRoot] + list(
            random.sample(self.context.get_cn_event_candidates(), tree_size)
        )

    def __generate_snvs(self, cn_tree: nx.DiGraph) -> Dict[CNEvent, Set[SNVEvent]]:
        node_to_snvs: Dict[CNEvent, Set[SNVEvent]] = {}

        def generate_snv_events_for_node(
            node: CNEvent, ancestor_snvs: Set[SNVEvent]
        ) -> None:
            if node != EventTreeRoot:
                node_to_snvs[node] = self.__sample_snvs_for_node(node, ancestor_snvs)
                ancestor_snvs = ancestor_snvs.union(node_to_snvs[node])
            for child in cn_tree.successors(node):
                generate_snv_events_for_node(child, ancestor_snvs)

        generate_snv_events_for_node(EventTreeRoot, set())
        return node_to_snvs

    def __sample_snvs_for_node(self, node: CNEvent, ancestor_snvs: Set[SNVEvent]) -> Set[SNVEvent]:
        snv_candidates = self.context.get_snv_event_candidates()
        snv_candidates = [x for x in snv_candidates if x not in ancestor_snvs]

        snvs_in_cn_event = [x for x in snv_candidates if node[0] <= x < node[1]]

        snvs = set()
        k = self.context.sample_number_of_snvs_for_edge(
            len(snv_candidates)
        )
        if len(snv_candidates) < k:
            raise RuntimeError(f"Not enough snv candidates for node. Increase number of candidates or decrease mean "
                               f"snvs per node")

        if random.random() <= self.context.snv_in_cn_proportion() and snvs_in_cn_event:
            num = min(len(snvs_in_cn_event), k)
            snvs = sample_conditionally_without_replacement(
                k=num,
                sampler=lambda: random.sample(snvs_in_cn_event, 1)[0],
                condition=lambda x: True
            )
        if len(snvs) == k:
            return snvs

        snv_candidates = [x for x in snv_candidates if x not in snvs]

        snvs2 = sample_conditionally_without_replacement(
            k=k - len(snvs),
            sampler=lambda: random.sample(snv_candidates, 1)[0],
            condition=lambda x: True,
        )
        return snvs2.union(snvs)
