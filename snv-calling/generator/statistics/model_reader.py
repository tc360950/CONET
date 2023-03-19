import pickle
from pathlib import Path

import numpy
import networkx as nx


class ModelReader:
    def __init__(self, dir):
        self.tree_path = Path(dir) / Path("event_tree")
        self.attachment_path = Path(dir) / Path("attachment")
        self.cn_path = Path(dir) / Path("cn")
        self._tree = None
        self._attachment = None
        self._cn = None
        self._snvs = None

    @property
    def cell_snv_pairs(self):
        result = []
        for cell, node in enumerate(self._attachment):
            path = nx.shortest_path(self.tree, (0, 0), node)
            for n in path:
                for snv in self._snvs.get(n, set()):
                    result.append((cell, snv))
        return result

    @property
    def tree(self):
        return self._tree

    @property
    def attachment(self):
        return self._attachment

    @property
    def cn(self):
        return self._cn

    @property
    def snvs(self):
        return self._snvs

    def load(self):
        tree = self._load_tree()
        self._tree = tree.tree.cn_event_tree
        self._snvs = tree.tree.node_to_snvs
        self._attachment = self._load_attachment()
        self._cn = self._load_cn()

    def _load_cn(self):
        return numpy.transpose(numpy.loadtxt(str(self.cn_path), delimiter=' ', dtype=int))

    def _load_attachment(self) -> list:
        with self.attachment_path.open(mode="rb") as f:
            return pickle.load(f)

    def _load_tree(self):
        with self.tree_path.open(mode="rb") as f:
            tree = pickle.load(f)
            return tree


if __name__ == "__main__":
    c = ModelReader("../../tmp")
    c.load()
    print("x")
