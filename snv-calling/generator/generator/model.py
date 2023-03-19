import random
from dataclasses import dataclass
from math import exp, log
from typing import Any, Dict, Generator

import networkx as nx
import numpy as np
from generator.core.types import CNEvent, EventTreeRoot
from generator.generator.context import SNVGeneratorContext
from generator.generator.corrected_counts_generator import CNSampler, CountsGenerator
from generator.generator.event_tree_generator import EventTree


@dataclass
class CellData:
    d: np.ndarray
    b: np.ndarray
    attachment: CNEvent
    corrected_counts: np.ndarray


@dataclass
class SNVModel:
    tree: EventTree
    node_to_cn_profile: Dict[CNEvent, np.ndarray]
    node_to_altered_counts_profile: Dict[CNEvent, np.ndarray]

    @classmethod
    def create_empty_model(cls, ctxt: SNVGeneratorContext) -> "SNVModel":
        tree = nx.DiGraph()
        tree.add_node(EventTreeRoot)
        return cls(
            tree=EventTree(cn_event_tree=tree, node_to_snvs={}),
            node_to_cn_profile={
                EventTreeRoot: np.full(
                    (ctxt.number_of_bins()), fill_value=ctxt.neutral_cn()
                )
            },
            node_to_altered_counts_profile={
                EventTreeRoot: np.zeros((ctxt.number_of_bins()))
            },
        )


class CellDataSampler:
    ctxt: SNVGeneratorContext
    counts_gen: CountsGenerator

    def __init__(self, ctxt: SNVGeneratorContext):
        self.ctxt = ctxt
        self.counts_gen = CountsGenerator(CNSampler.create_default_sampler())

    def generate_cell_in_node(self, node: CNEvent, model: SNVModel) -> CellData:
        d = np.zeros(self.ctxt.number_of_bins())
        b = np.zeros(self.ctxt.number_of_bins())
        cc = self.counts_gen.add_noise_to_counts(model.node_to_cn_profile[node])
        for bin in range(0, d.shape[0]):
            d[bin] = self.__sample_read_in_bin(model.node_to_cn_profile[node][bin])
            b[bin] = self.__sample_altered_reads_in_bin(
                model.node_to_altered_counts_profile[node][bin],
                d[bin],
                model.node_to_cn_profile[node][bin],
            )

        return CellData(d=d, b=b, attachment=node, corrected_counts=cc)

    def generate_cell_attachment(
        self, model: SNVModel
    ) -> Generator[CNEvent, CNEvent, Any]:
        if model.tree is None:
            raise RuntimeError("Can't generate cells for empty tree!")

        probs = self.__create_node_attachment_probabilities(model)
        while True:
            yield random.choices(list(probs), k=1, weights=list(probs.values()))[0]

    def __create_node_attachment_probabilities(
        self, model: SNVModel
    ) -> Dict[CNEvent, float]:
        """
        Probability of cell being attached to given node is proportional
        to  e^{0.1 * depth} where depth is node's depth in the tree.
        """
        depths = nx.shortest_path_length(model.tree.cn_event_tree, source=EventTreeRoot)
        depths = dict([(key, np.exp(0.1 * val)) for key, val in depths.items()])
        sum_depths = sum(list(depths.values()))
        for key, value in depths.items():
            depths[key] = np.exp(0.1 * depths[key]) / sum_depths
        return depths

    def __sample_read_in_bin(self, cn: int) -> int:
        log_mean = log(
            self.ctxt.sequencing_error() + cn * self.ctxt.per_allele_coverage()
        )
        num_failures = exp(
            log(1.0 - self.ctxt.read_success_prob())
            + log_mean
            - log(self.ctxt.read_success_prob())
        )
        return np.random.negative_binomial(
            num_failures, 1 - self.ctxt.read_success_prob()
        )

    def __sample_altered_reads_in_bin(
        self, altered_counts: int, total_reads: int, cn: int
    ) -> int:
        if total_reads == 0:
            return 0
        if altered_counts == 0 or cn == 0:
            return np.random.binomial(total_reads, self.ctxt.sequencing_error())
        return np.random.binomial(total_reads, altered_counts / cn)
