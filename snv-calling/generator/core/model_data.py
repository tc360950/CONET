from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from generator.core.event_tree import EventTree
from generator.core.types import Cell, CNEvent


@dataclass
class Parameters:
    m: float  # per allele coverage
    e: float  # sequencing error
    q: float  # read success probablity

    def __str__(self) -> str:
        return (
            f"Per allele coverage: {self.m}, "
            f"sequencing error: {self.e}, "
            f"read success probability: {self.q}."
        )


@dataclass
class EventTreeWithCounts:
    tree: EventTree
    node_to_cn_profile: Dict[CNEvent, np.ndarray]


@dataclass
class CellsData:
    d: np.ndarray
    b: np.ndarray
    attachment: List[CNEvent]
    # Each cell represents a cluster of given size
    cell_cluster_sizes: List[int]

    def get_cells_attached_to_node(self, node: CNEvent) -> List[Cell]:
        return [c for c in range(0, len(self.attachment)) if self.attachment[c] == node]

    def get_cells(self) -> List[Cell]:
        return list(range(0, len(self.attachment)))


def create_cn_matrix(cells: CellsData, tree: EventTreeWithCounts) -> np.ndarray:
    matrix = np.zeros(cells.d.shape)
    for cell in range(0, cells.d.shape[0]):
        matrix[
            cell,
        ] = tree.node_to_cn_profile[cells.attachment[cell]]
    return matrix.astype(int)
