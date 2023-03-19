import logging
from typing import Generator, Tuple

import numpy as np

import networkx as nx
from generator.core.model_data import CellsData
from generator.core.types import CNEvent, EventTreeRoot
from generator.core.utils import apply_to_nodes_in_order
from generator.generator.context import SNVGeneratorContext, ContextException
from generator.generator.event_tree_generator import EventTreeGenerator
from generator.generator.model import CellDataSampler, SNVModel

CCMatrix = np.ndarray

logger = logging.getLogger(__name__)
def _deterministic_attachment(model: SNVModel) -> Generator[CNEvent, CNEvent, None]:
    nodes = [n for n in model.tree.cn_event_tree.nodes]
    while True:
        for n in nodes:
            yield n


def sample_cells_data(
    clusters: int,
    cluster_size: int,
    model: SNVModel,
    ctxt: SNVGeneratorContext,
    random_attachment: bool,
) -> Tuple[CellsData, CCMatrix]:
    cluster_to_cell_data = {}
    cell_sampler = CellDataSampler(ctxt=ctxt)
    attach = (
        cell_sampler.generate_cell_attachment(model)
        if random_attachment
        else _deterministic_attachment(model)
    )
    for c in range(0, clusters):
        node = next(attach)
        cell = cell_sampler.generate_cell_in_node(node, model)
        cc_matrix = cell.corrected_counts
        for _ in range(0, cluster_size - 1):
            cell2 = cell_sampler.generate_cell_in_node(node, model)
            cell.d = cell.d + cell2.d
            cell.b = cell.b + cell2.b
            cc_matrix = np.vstack([cc_matrix, cell2.corrected_counts])
        if cluster_size > 1:
            cc_matrix = np.mean(cc_matrix, axis=0)
        cell.corrected_counts = cc_matrix
        cluster_to_cell_data[c] = cell

    attachment = [cluster_to_cell_data[c].attachment for c in range(0, clusters)]

    b = np.vstack([cluster_to_cell_data[c].b for c in range(0, clusters)])
    d = np.vstack([cluster_to_cell_data[c].d for c in range(0, clusters)])
    cc = np.vstack(
        [cluster_to_cell_data[c].corrected_counts for c in range(0, clusters)]
    )
    return (
        CellsData(
            d=d,
            b=b,
            attachment=attachment,
            cell_cluster_sizes=[cluster_size for _ in range(0, clusters)],
        ),
        cc,
    )


class SNVModelGenerator:
    ctxt: SNVGeneratorContext

    def __init__(self, ctxt: SNVGeneratorContext):
        self.ctxt = ctxt
        self.model = SNVModel.create_empty_model(self.ctxt)

    def generate_model(self, tree_size: int) -> SNVModel:
        def __generate():
            self.model = SNVModel.create_empty_model(self.ctxt)
            self.model.tree = EventTreeGenerator(self.ctxt).generate(tree_size)
            apply_to_nodes_in_order(
                root=EventTreeRoot,
                tree=self.model.tree.cn_event_tree,
                func=lambda n: self.__generate_node_cn_profile(n)
                if n != EventTreeRoot
                else None,
            )
            apply_to_nodes_in_order(
                root=EventTreeRoot,
                tree=self.model.tree.cn_event_tree,
                func=lambda n: self.__generate_node_altered_counts_profile(n)
                if n != EventTreeRoot
                else None,
            )

        try:
            __generate()
        except ContextException:
            self.model = None
        while self.model is None or not self.__tree_is_valid(self.model, self.ctxt.number_of_bins(), self.ctxt.neutral_cn()):
            logger.error("Generated tree is not valid, trying one more time...")
            try:
                __generate()
            except ContextException:
                self.model = None
        logger.info("Model generated")
        return self.model

    def __tree_is_valid(self, model: SNVModel, bins: int, neutral_cn: int) -> bool:
        counts = np.zeros([1, bins], dtype=np.float64)
        counts.fill(neutral_cn)
        brkps = np.zeros([1, bins], dtype=np.float64)
        node_to_counts = {}
        node_to_brkps = {}
        tree = model.tree.cn_event_tree
        for node in tree.nodes:
            path = nx.shortest_path(tree, list(tree.nodes)[0], node)
            node_brkps = np.copy(brkps)
            for i in range(0, len(path)):
                ancestor = path[i]
                for j in range(ancestor[0], ancestor[1]):
                    node_brkps[0, ancestor[0]] = 1
                    node_brkps[0, ancestor[1]] = 1
            node_to_counts[node] = model.node_to_cn_profile[node]
            node_to_brkps[node] = node_brkps
            for i in range(node_to_brkps[node].shape[1]):
                if node_to_brkps[node][0, i] == 1 and node_to_counts[node][i] == node_to_counts[node][i - 1]:
                    logger.error("Non identifiable breakpoint")
                    return False
        return True
    def __generate_node_cn_profile(self, node: CNEvent) -> None:
        self.model.node_to_cn_profile[node] = self.model.node_to_cn_profile[
            self.model.tree.get_node_parent(node)
        ].copy()
        bitmap = self.model.tree.mark_bins_in_descendant_events(
            node, np.full((self.ctxt.number_of_bins()), False)
        )
        cn_change = self.ctxt.sample_cn_change(
            node,
            self.model.node_to_cn_profile[self.model.tree.get_node_parent(node)],
            bitmap,
        )
        for bin in range(node[0], node[1]):
            self.model.node_to_cn_profile[node][bin] += cn_change
            if self.model.node_to_cn_profile[node][bin] < 0:
                raise RuntimeError(f"{cn_change} {self.model.node_to_cn_profile[self.model.tree.get_node_parent(node)]} \n{bitmap} \n {node}")

    def __generate_node_altered_counts_profile(self, node: CNEvent) -> None:
        alterations = self.model.node_to_altered_counts_profile[
            self.model.tree.get_node_parent(node)
        ].copy()
        node_cn_profile = self.model.node_to_cn_profile[node]

        for snv in self.model.tree.node_to_snvs[node]:
            if (
                self.model.node_to_cn_profile[self.model.tree.get_node_parent(node)][
                    snv
                ]
                > 0
            ):
                alterations[snv] = 1

        for bin in [bin for bin in range(node[0], node[1]) if alterations[bin] > 0]:
            alterations[bin] = self.ctxt.get_number_of_alterations(
                cn_before=self.model.node_to_cn_profile[
                    self.model.tree.get_node_parent(node)
                ][bin],
                cn_after=node_cn_profile[bin],
                parent_altered_counts=alterations[bin],
            )

        self.model.node_to_altered_counts_profile[node] = alterations
