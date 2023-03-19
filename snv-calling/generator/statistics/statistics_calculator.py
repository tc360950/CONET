from pathlib import Path

import numpy

from generator.statistics.conet_reader import ConetReader
from generator.statistics.model_reader import ModelReader


class StatisticsCalculator:
    def __init__(self, dir: str, prefix:str, postfix: str=None):
        self.conet = ConetReader(Path(dir) / Path("out"), Path(dir) / Path("cc"), postfix)
        self.model = ModelReader(dir)
        self.conet.load()
        self.model.load()
        self.prefix=prefix

    def get_stat_names(self) -> str:
        return ';'.join([
            "inferred_tree_size",
            "real_tree_size",
            "inferred_snvs",
            "real_snvs",
            "cn_node_recall",
            "cn_node_precision",
            "cn_edge_recall",
            "cn_edge_precision",
            "snv_recall",
            "snv_precision",
            "cell_snv_recall",
            "cell_snv_precision",
            "cn_prob",
            "non_trivial_cns"
        ])

    def calculate(self) -> str:
        result = f"{self.prefix};"
        inferred_nodes = set(self.conet.tree.nodes)
        real_nodes = set(self.model.tree.nodes)
        inferred_snvs = {x for y in self.conet.snvs.values() for x in y}
        real_snvs = {x for y in self.model.snvs.values() for x in y}

        result += f"{len(inferred_nodes)};{len(real_nodes)};{len(inferred_snvs)};{len(real_snvs)};"
        result += f"{len(inferred_nodes.intersection(real_nodes)) / len(real_nodes)};"
        result += f"{len(inferred_nodes.intersection(real_nodes)) / len(inferred_nodes)};"

        inferred_edges = set(self.conet.tree.edges)
        real_edges = set(self.model.tree.edges)

        result += f"{len(inferred_edges.intersection(real_edges)) / len(real_edges)};"
        result += f"{len(inferred_edges.intersection(real_edges)) / len(inferred_edges)};"

        result += f"{len(inferred_snvs.intersection(real_snvs)) / (len(real_snvs) if real_snvs else 1.0)};"
        result += f"{len(inferred_snvs.intersection(real_snvs)) / (len(inferred_snvs) if inferred_snvs else 1.0)};"

        inferred_cell_snv = set(self.conet.cell_snv_pairs)
        real_cell_snv = set(self.model.cell_snv_pairs)

        result += f"{len(inferred_cell_snv.intersection(real_cell_snv)) / (len(real_cell_snv) if real_cell_snv else 1.0)};"
        result += f"{len(inferred_cell_snv.intersection(real_cell_snv)) / (len(inferred_cell_snv) if inferred_cell_snv else 1.0)};"

        inferred_cn = self.conet.cn
        real_cn = self.model.cn

        cn_percent = numpy.sum((inferred_cn - real_cn) == 0) / inferred_cn.size
        result += f"{cn_percent};{numpy.sum(real_cn == 2) / inferred_cn.size}"

        return result


if __name__ == "__main__":
    calc = StatisticsCalculator("../../tmp")
    print(calc.calculate())
