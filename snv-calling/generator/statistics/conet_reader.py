from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy
import pandas


class ConetReader:
    def __init__(self, dir, cc_path: Path, postfix: str):
        self.cc_path = cc_path
        self.tree_path = Path(dir) / Path(f"inferred_tree")
        self.attachment_path = Path(dir) / Path(f"inferred_attachment")
        self.snvs_path = Path(dir) / Path(f"inferred_snvs2")
        self.cn_path = Path(dir) / Path(f"inferred_counts")
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
        self._tree = self._load_tree()
        self._attachment = self._load_attachment()
        self._snvs = self._load_snvs()
        self._cn = self._load_cn()

    def _load_cn(self):
        return numpy.loadtxt(str(self.cn_path), delimiter=';', dtype=int)

    def _load_snvs(self):
        result = defaultdict(set)
        def line_to_edge(line):
            line = line.replace('1_', '')
            parent, child, rest = line.split(';')
            return int(parent), int(child), int(rest)

        with self.snvs_path.open(mode="r") as f:
            data = f.read().split('\n')
            for d in data:
                if d == '':
                    continue
                parent, child, snv = line_to_edge(d)
                result[(parent, child)].add(snv)
        return result

    def _load_attachment(self):
        with self.attachment_path.open(mode="r") as f:
            data = f.read().split('\n')
            result = []
            for d in data:
                if d == '':
                    continue
                d = d.replace('1_', '')
                cell, num, loci1, loci2 = d.split(';')
                loci1 = int(loci1)
                loci2 = int(loci2)
                if loci2 == loci1:
                    loci2 = 0
                    loci1 = 0
                result.append((loci1, loci2))
        return result

    def _load_tree(self) -> nx.DiGraph:
        with self.tree_path.open(mode="r") as f:
            data = f.read()

            def line_to_edge(line):
                line = line.replace('1_', '')
                parent, child = line.split('-')
                parent = eval(parent)
                child = eval(child)
                return parent, child

            lines = data.split('\n')
            edges = []
            for line in lines:
                if line == '':
                    continue
                edges.append(line_to_edge(line))
            graph = nx.DiGraph()
            graph.add_edges_from(edges)
            return graph


if __name__ == "__main__":
    c = ConetReader("../../tmp/out")
    c.load()
    print("x")
