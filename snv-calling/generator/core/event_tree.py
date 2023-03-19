from dataclasses import dataclass
from typing import Dict, List, Set

import networkx as nx
import numpy as np
from generator.core.types import Bin, CNEvent, EventTreeRoot, SNVEvent
from generator.core.utils import apply_to_nodes_in_order


@dataclass
class EventTree:
    cn_event_tree: nx.DiGraph
    node_to_snvs: Dict[CNEvent, Set[SNVEvent]]

    def __copy__(self):
        return EventTree(
            cn_event_tree=self.cn_event_tree.copy(),
            node_to_snvs=self.node_to_snvs.copy(),
        )

    def add_snv_in_node(self, node: CNEvent, snv: SNVEvent) -> None:
        if node == EventTreeRoot:
            raise ValueError("Can't add SNV event to the root node!")
        self.node_to_snvs.setdefault(node, set()).add(snv)

    def delete_snv_from_node(self, node: CNEvent, snv: SNVEvent) -> None:
        self.node_to_snvs.get(node).remove(snv)

    def get_breakpoint_loci(self) -> List[Bin]:
        return sorted(
            list(
                set(
                    [x[0] for x in self.cn_event_tree.nodes if x != (0,0)]
                    + [x[1] for x in self.cn_event_tree.nodes if x != (0,0)]
                )
            )
        )

    def remove_snvs(self) -> None:
        self.node_to_snvs.clear()

    def get_node_parent(self, node: CNEvent) -> CNEvent:
        return next(self.cn_event_tree.predecessors(node))

    def snv_can_be_added_in_node(self, snv: SNVEvent, node: CNEvent) -> bool:
        """
        Check if snv @snv can be added to edge leading to node @node.
        Addition of snv is forbidden if it's already present either in the subtree of @node
        or on edges leading from the node to the root.
        """
        return snv not in self.__gather_snvs_in_subtree(
            node
        ) and snv not in self.__gather_snvs_on_path_to_root(node)

    def mark_bins_in_descendant_events(
        self, node: CNEvent, bin_bit_map: np.ndarray
    ) -> np.ndarray:
        """
        Set @bin_bit_map[bin] to True if CN in the bin is changed in the subtree of node.
        Node is excluded.
        """

        def __mark_overlapping_bins(event: CNEvent) -> None:
            bin_bit_map[
                [b for b in range(event[0], event[1]) if node[0] <= b < node[1]]
            ] = True

        for descendant in nx.descendants(self.cn_event_tree, node):
            if descendant != node:
                __mark_overlapping_bins(descendant)
        return bin_bit_map

    def mark_bins_with_snv(self, node, bin_bit_map: np.ndarray) -> np.ndarray:
        bin_bit_map[list(self.__gather_snvs_on_path_to_root(node))] = True
        return bin_bit_map

    def mark_bins_with_cn_change_after_alteration(
        self, node: CNEvent, bin_bit_map: np.ndarray
    ) -> np.ndarray:
        bit_map = np.full(bin_bit_map.shape, 0, dtype=int)
        BIN_IN_SNV = 1
        CN_ALTERATION_AFTER_SNV = 2

        for n in self.__get_path_from_root(node):
            bit_map[list(self.node_to_snvs.get(n, set()))] = BIN_IN_SNV
            bit_map[
                [b for b in range(n[0], n[1]) if bit_map[b] == BIN_IN_SNV]
            ] = CN_ALTERATION_AFTER_SNV

        bin_bit_map[bit_map == CN_ALTERATION_AFTER_SNV] = True
        return bit_map

    def __gather_snvs_in_subtree(self, node: CNEvent) -> Set[SNVEvent]:
        snvs = []
        apply_to_nodes_in_order(
            node,
            self.cn_event_tree,
            lambda n: snvs.extend(self.node_to_snvs.get(n, set())),
        )
        return set(snvs)

    def __gather_snvs_on_path_to_root(self, node: CNEvent) -> Set[SNVEvent]:
        snvs = []
        for n in self.__get_path_from_root(node):  # gather snvs from path to the root
            snvs.extend(self.node_to_snvs.get(n, set()))
        return set(snvs)

    def __get_path_from_root(self, node: CNEvent) -> List[CNEvent]:
        return (
            []
            if node == EventTreeRoot
            else next(
                nx.all_simple_paths(
                    source=EventTreeRoot, target=node, G=self.cn_event_tree
                )
            )
        )
